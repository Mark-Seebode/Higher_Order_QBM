import numpy as np

import torch
from torch.nn.functional import conv1d, pad
from torch.fft import fft
from torchaudio.transforms import FFTConvolve
import time
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from IPython.display import display, Math


def latex_matrix(matrix):
    latex_str = "\\begin{bmatrix}\n"
    for row in matrix:
        latex_str += " & ".join(map(str, row)) + " \\\\\n"
    latex_str += "\\end{bmatrix}"
    display(Math(latex_str))



def TorchRotation(theta,phi,lamb):
    try:
        exp_term1 = torch.exp(1j * (phi + lamb))
        exp_term2 = torch.exp(1j * phi)
        exp_term3 = torch.exp(1j * lamb)


        R = torch.stack([
            torch.stack([exp_term1 * torch.cos(theta), torch.sin(theta) * exp_term2]),
            torch.stack([exp_term3 * torch.sin(theta), -torch.cos(theta)])])

        return R
    except Exception as e:
        raise ValueError(f"Error in rotation matrix calculation: {e}")

def Rotation(theta, phi, lamb):
    '''
    Function that returns the general SU(2) Rotation matrix.
    Args:
    
    theta
    phi
    lamb

    Returns: 
    ndarray , Rotation Matrix
    '''
    
    # Return a rotation matrix 
    try:
        R = np.array([[np.exp(1.0j * (phi + lamb)) * np.cos(theta), np.sin(theta) * np.exp(1.j * phi)],
                       [np.exp(1.0j * lamb) * np.sin(theta), -np.cos(theta)]], dtype=np.complex128)
        return R
    except Exception as e:
        raise ValueError(f"Error in rotation matrix calculation: {e}")



def Compute(S,d):
    a=(S[0][d])
    b=(S[1][d])

    


    magnitude = torch.abs(b) / torch.abs(a)
    theta = torch.arctan2(torch.abs(b),torch.abs(a))  # or torch.atan(magnitude)

    phi = torch.angle(a/b) 


    #print(d)
    #print('theta: ', theta)
    #print('phi:',phi)
    if d==0:
        lamb=torch.angle(b)
        return (theta,),(phi,),(lamb)
    
    
    #print(TorchRotation(theta,phi,torch.tensor(0.0)))
    S=torch.matmul(TorchRotation(theta,phi,torch.tensor(0.0)).conj().T,S.to(torch.complex128))
    #print(S)
    S = torch.stack([
    S[0, 1:],  # Shift the first row
    S[1, :-1]])
    
  
    theta_1,phi_1,lamb_1=Compute(S,d-1)
    
    
    return (theta_1+(theta,)),(phi_1+(phi,)),(lamb_1)



def Compute_parameters(S,d:int):
    '''
    Function that computes the list of rotation parameters in increasing order given the matrix S.

    Args:

    S (ndarray): (2,n) Matrix containing coefficients of polynomials P and Q. 
    d(int) : index of rotation parameters.

    Returns:

    (theta's,.,. )(phi's,.,.)(lamda)
    '''
    
    
    
    a=np.complex128(S[0][d])
    
    b=np.complex128(S[1][d])
    
    theta=np.arctan2(np.abs(b),np.abs(a))
    t=np.complex128(a/b)
    phi=np.angle(t)
    if d==0:
        lamb=np.angle(b)
        return (theta,),(phi,),(lamb)
    
    
    S=np.matmul(Rotation(theta,phi,0).conj().T,S)
    S = np.array([
    S[0, 1:],  # Shift the first row
    S[1, :-1]])
    
    theta_1,phi_1,lamb_1=Compute_parameters(S,d-1)
    
    
    return (theta_1+(theta,)),(phi_1+(phi,)),(lamb_1)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def Q_polynomial(poly: torch.tensor):

    '''
    Function that returns a Q polynomial that satisfies the constraints of theorem 3 in 'https://arxiv.org/abs/2308.01501 , Generalized Quantum Signal Processing (2024)'
    Args:
    poly (torch.tensor) : coefficents for the polynomial to be applied.

    Returns
    Q (torch.tensor)
    '''
    print('Calculating Q..')
    def objective_torch(x, P):
        x.requires_grad = True
    
        # Compute loss using squared distance function
        loss = torch.norm(P - FFTConvolve("full").forward(x, torch.flip(x, dims=[0])))**2
        return loss
    
    times = []
    final_vals = []
    num_iterations = []
    
    polynomial=poly.clone()
    N = len(polynomial)
    #poly = torch.randn(N, device=device)
    
    granularity = 2 ** 25
    P = pad(polynomial, (0, granularity - polynomial.shape[0]))
    ft = fft(P)
    
        # Normalize P
    P_norms = ft.abs()
    polynomial = polynomial/ torch.max(P_norms)
    print('Max P(z) = ', torch.max(P_norms))
    conv_p_negative = FFTConvolve("full").forward(polynomial, torch.flip(polynomial, dims=[0]))* -1
    conv_p_negative[polynomial.shape[0] - 1] = 1 - torch.norm(polynomial) ** 2
    
        # Initializing Q randomly to start with
    initial = torch.randn(polynomial.shape[0], device=device, requires_grad=True)
    initial = (initial / torch.norm(initial)).clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.LBFGS([initial], max_iter=9000,tolerance_change=1e-16)
    
    t0 = time.time()
    
    def closure():
        optimizer.zero_grad()
        loss = objective_torch(initial, conv_p_negative)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    t1 = time.time()
    
    total = t1-t0
    
    print(f'N: {N}')
    print(f'Time: {total}')
    print(f'Final: {closure().item()}')
    print(f"# Iterations: {optimizer.state[optimizer._params[0]]['n_iter']}")
    print("-----------------------------------------------------")

    return initial.detach().clone(), closure().item()


def P_polynomial(poly):
    '''
    Function to return a normalized version of the polynomial. Such that the generalized QSP sequence could be applied.'
    Args:
    poly (torch.tensor)

    Returns:
    
    '''
    
    
    #poly = torch.randn(N, device=device)
    polynomial=poly.clone()
    granularity = 2 ** 25
    Pad = pad(polynomial, (0, granularity - polynomial.shape[0]))
    ft = fft(Pad)
    
        # Normalize P
    P_norms = ft.abs()
    polynomial = polynomial/ torch.max(P_norms)
    return polynomial.detach().clone()


