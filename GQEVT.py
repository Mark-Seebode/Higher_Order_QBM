"""
Generalized Quantum Eigenvalue Transformation (GQEVT)

This module provides the :class:GQEVT, which implements functionality for applying generalized quantum signal processing (GQSP) to a
block-encoded operator (for some Hamiltonian H).

Main steps
----------
1. Generate a Chebyshev expansion for a target function (e.g. e^{βx})
   on x ∈ [-1, 1].
2. Convert the polynomial coefficients into quantum signal processing
   angles using either PennyLane's built-in methods or gradient-based routine.
3. Build a block-encoding for H.
4. Assemble the full GQSP circuit and simulate it either as a
   PennyLane circuit, a dense matrix product, or via a precomputed
   block-encoded inverse.

The resulting unitary can be used as a building block for Gibbs-state preparation.
"""




import numpy as np
import torch
from torch.nn.functional import conv1d, pad
from torch.fft import fft
from torchaudio.transforms import FFTConvolve
import time
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from helper import *
from scipy.stats import unitary_group
import itertools





from pyqsp.poly import PolyOneOverX
from pyqsp.poly import PolyCosineTX
import scipy
from pyqsp.poly import PolyTaylorSeries   # built-in since v0.1.4




class GQEVT:
    """
    This class builds a GQEVT circuit for a block-encoded operator. It
    internally:
    - constructs a Chebyshev polynomial approximation to a target function
      (by default e^{βx}),
    - converts the polynomial coefficients into GQSP angles,
    - assembles the corresponding quantum circuit in PennyLane.

    Typical usage
    -------------
    >>> gqevt = GQEVT(n=30, pre_fac=-1.0, method="GQSP")
    >>> gqevt.build(H)                 # provide matrix representation of H
    >>> final_state = gqevt.measure_qnode(sim_method="pennylane")

    Parameters
    ----------
    n : int, optional
        Degree of the polynomial approximation (number of Chebyshev
        coefficients - 1). Defaults to 30.
    pre_fac : float, optional
        Prefactor β used for the exponential e^{βx} 
        
    method : {"rootfinding", "gradient"}, optional
        Method used to compute QSP angles:
        - "rootfinding": use PennyLane's ``qml.poly_to_angles`` with scheme "GQSP".
        - "gradient": uses gradient-based routines defined in
          ``helper.py``.
        Defaults to "rootfinding" which is more accurate.
    scaling : float, optional
        Additional scaling applied to ``pre_fac`` before building the
        Chebyshev expansion. Useful when rescaling the spectrum of H.
        Defaults to 1.

    Attributes
    ----------
    n : int
        Polynomial degree.
    pre_fac : float
        Prefactor β used in the exponential e^{βx}.
    poly : np.ndarray
        Chebyshev coefficients of the target function.
    scale : float
        Overall scaling factor returned from the polynomial generator and
        normalized by the Fourier norm of the polynomial.
    p_scale : torch.Tensor
        Max-norm of the polynomial in the Fourier domain, used for scaling.
    method : str
        Angle-computation method ("rootfinding" or "gradient").
    angles : tuple
        QSP / GQSP angles used to build the circuit.
    H : np.ndarray
        Hamiltonian matrix passed to :meth:`build`.
    U : np.ndarray
        Block-encoding unitary of H.
    A : qml.operation.ControlledQubitUnitary
        Controlled version of the block-encoding (signal operator).
    block_block_inv : torch.Tensor or np.ndarray
        Block-encoding of e^{-H}, used when ``sim_method="matrix"``.
    dim : int
        Number of signal qubits.
    target_wires : list[int]
        Wire indices for the signal register.
    norm : float
        Normalization factor returned by :class:`qml.BlockEncode` for the
        e^{-H} block-encoding.
    """
    
    
    
    
    def __init__(self, n=30, pre_fac=-1.0, method='rootfinding',scaling=1,seed=32, sim_level=0):

        def generate_coeff_exp(n=30, beta=-1.0):
            """
            Chebyshev coefficients for a numerically-stable normalized exponential on [-1,1]:
                f(x) = exp(beta*x - |beta|)
            This guarantees |f(x)| <= 1 and avoids overflow when |beta| is large.
            """
            samples = 2 * n

            poly = PolyTaylorSeries()
            c = poly.taylor_series(
                func=lambda x: np.exp(beta * x - abs(beta)),  # <-- stable normalization
                degree=n,
                chebyshev_basis=True,
                cheb_samples=samples,
                return_scale=False
            )
            return c.coef, 1

        torch.manual_seed(seed)

        self.n=n
        self.pre_fac=pre_fac
        self.sim_level=sim_level

        if self.sim_level != 0:
            self.poly,self.scale=generate_coeff_exp(n,pre_fac*scaling)
            #self.poly=torch.from_numpy(self.poly)
            self.method = method

            self._compute_angles(method)

            polynomial=torch.tensor(self.poly).clone()


            ft = fft(polynomial)

                # Normalize P
            P_norms = ft.abs()
            self.p_scale=torch.max(P_norms)




            self.scale=self.scale/self.p_scale

    
    
    
    def generate_coeff(self, kappa=3,epsilon=0.1):
        '''
    
        Generate coefficients in the Chebychev basis for the function 1/x.
        '''
        
        poly_generator = PolyOneOverX(verbose=False)
        #poly_generator = PolyCosineTX(verbose=False)
      
    
        coefficients,scale = poly_generator.generate(kappa=kappa, epsilon=epsilon, return_scale=True,chebyshev_basis=True,return_coef=True)
        return coefficients,scale

    
    def _BlockEncode(self,A,n):
        
        
        
        encoding=qml.BlockEncode(A, wires=range(n))
        
        return encoding
    
    def _compute_angles(self, method):
        
        """
        Compute GQSP angles for the current polynomial.

        Depending on "method", this uses either:
        - "gradient": gradient-based routine from "helper.py"
          
        - "rootfinding": PennyLane's :func:`qml.poly_to_angles`/

        The result is stored in :attr:'angles' as a tuple of torch tensors.

        Parameters
        ----------
        method : str
            Angle computation strategy ("gradient" or "rootfinding").
        """
        
        
        
        if method == 'gradient':
            self.poly=torch.tensor(self.poly)
            p_poly = P_polynomial(self.poly)
            q_poly, _ = Q_polynomial(self.poly)

            S = torch.stack([
                p_poly.to(torch.float64),
                q_poly.to(torch.float64)
            ])

            self.angles = Compute(S, len(self.poly) - 1)
        else:
            from numpy.polynomial.chebyshev import chebval

            xs = np.linspace(-1.0, 1.0, 2001)
            vals = chebval(xs, self.poly)

            m = np.max(np.abs(vals))
            if not np.isfinite(m) or m == 0:
                raise ValueError(f"Invalid GQSP polynomial max norm: {m}")

            # PennyLane's GQSP angle solver is fragile exactly at |P| = 1.
            # Keep a small margin inside the valid region
            self.poly = self.poly * (0.999 / (m + 1e-12))
            nonzero_coeffs = np.flatnonzero(np.abs(self.poly) > 1e-14)
            if len(nonzero_coeffs) > 0:
                self.poly = self.poly[: nonzero_coeffs[-1] + 1]

            angles = qml.poly_to_angles(self.poly, "GQSP")
            
            self.angles= [tuple(torch.from_numpy(angles[0])),tuple(torch.from_numpy(angles[1])),torch.tensor(angles[2][0])]



    def circuit(self):
       """
        Build the GQEVT circuit for a given set of GQSP rotation angles.

        Parameters
        ----------
        angles : tuple(theta, phi, lamb) of GQSP angles.
        
        sim_level : {0,1,2}, optional
            - "2 (Full circuit)": Build the GQSP unitary using basic PennyLane
            operations layer by layer(controlled block-encoding + reflections).
            - "1": Explicitly construct the full matrix
              representation of the GQSP unitary using individual rotation layers and embed the final result
              as a single 'qml.QubitUnitary'.
            - "0 ": use a precomputed block-encoding of e^{-H}. 
              stored in :attr:'block_block_inv'.

        Notes
        -----
        This method only defines the circuit on the qnode;
        it does not execute it. 
        """
      
       if self.sim_level==2:

            theta, phi, lamb = self.angles

            modified_phi = tuple(p + torch.pi for p in phi[:-1]) + (phi[-1],)

            qml.QubitUnitary(TorchRotation(theta[0], modified_phi[0], lamb), wires=0)

            for i in range(1, len(theta)):
                qml.PauliX(wires=0)
                qml.QubitUnitary(self.A.matrix(), wires=self.A.wires, id='Block')
                qml.PauliX(wires=0)

                diag = torch.ones(2 ** (self.dim + 1), dtype=torch.complex128)
                diag[:2 ** (self.dim - 1)] *= -1
                qml.DiagonalQubitUnitary(diag, wires=range(self.dim + 1), id='Refl')

                qml.QubitUnitary(TorchRotation(theta[i], modified_phi[i], torch.tensor(0.0)), wires=0)

       elif self.sim_level==1:

            theta, phi, lamb = self.angles

            modified_phi = tuple(p + torch.pi for p in phi[:-1]) + (phi[-1],)

            R=torch.kron(TorchRotation(theta[0], modified_phi[0], lamb), torch.eye(2**self.dim))

            X= torch.kron(torch.tensor([[0,1],[1,0]],dtype=torch.complex128),torch.eye(2**self.dim))

            final_matrix=R
            for i in range(1, len(theta)):
                final_matrix=torch.matmul(R,X)

                final_matrix=torch.matmul(final_matrix,torch.tensor(self.A.matrix()))
                final_matrix=torch.matmul(final_matrix,X)

                diag = torch.ones(2 ** (self.dim + 1), dtype=torch.complex128)
                diag[:2 ** (self.dim - 1)] *= -1

                final_matrix=torch.matmul(final_matrix,torch.diag(diag))

                R1=TorchRotation(theta[i], modified_phi[i], torch.tensor(0.0))
                final_matrix=torch.matmul(final_matrix,torch.kron(R1,torch.eye(2**self.dim)))

            qml.QubitUnitary(final_matrix,wires=range(self.dim+1))
       elif self.sim_level==0:
          
           qml.QubitUnitary(self.block_block_inv,wires=range(self.dim+1))

    def measure_qnode(self,sim_level=0):
        self.dev = qml.device("default.qubit", wires=len(self.A.wires))
        
      
        @qml.qnode(self.dev, interface="torch")
        def measurement_circuit(angles):
            self.circuit(angles,sim_level)
            return qml.state()   

        return measurement_circuit(self.angles)
        
        

    
    def get_circuit(self):
        return self.circuit(self.angles)

    def plot_circuit(self):
        qml.draw_mpl(self.circuit)(self.angles)

    def build(self,H):
        """
        Build block-encodings for H and e^{-H}(used incase of sim_level=0) and set up internal operators.

        Given a Hamiltonian matrix H, this method:
        1. Constructs a block-encoding U of H via :class:'qml.BlockEncode'.
        2. Computes e^{-H} using :func:'scipy.linalg.expm' if sim_level=0.
        3. Builds a controlled version of U, stored in :attr:'A', which
           plays the role of the "signal" operator in QSP / GQSP.
        4. Builds a block-encoding of e^{-H} and stores its matrix
           representation in :attr:'block_block_inv', along with the
           normalization factor :attr:'norm'.

        Parameters
        ----------
        H : array-like
            Hermitian matrix representing the Hamiltonian. Its dimension
            must be a power of two so that log2(dim) is an integer.

        Notes
        -----
        After calling :meth:'build', the object is ready to generate and
        simulate the corresponding GQEVT circuit.
        """
        self.H = H
        dim_H = int(np.log2(len(H[0])))

        U = self._BlockEncode(H, dim_H + 1)
        BlockU = qml.matrix(U)

        self.U = BlockU
        self.dim = int(torch.log2(torch.tensor(len(BlockU), dtype=torch.float64)).item())
        self.target_wires = list(range(0, self.dim + 1))
        self.A = qml.ControlledQubitUnitary(self.U, wires=self.target_wires)

        if self.sim_level == 0:
            inv = scipy.linalg.expm(-self.H)
            self.block_block_inv = self._BlockEncode(inv, dim_H + 2)
            self.norm = self.block_block_inv.hyperparameters["norm"]
            self.block_block_inv = qml.matrix(self.block_block_inv)
