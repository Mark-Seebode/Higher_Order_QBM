"""
Quantum Boltzmann Machine (QBM) Model with GQEVT
================================================================================

This module implements discriminative Quantum Boltzmann Machine models:



A QBM models the conditional distribution P(y|x) using a parameterized
Ising-type Hamiltonian constructed via :class:ModelHamiltonian. Training is
performed using an approximate Gibbs state prepared by the Generalized Quantum
Eigenvalue Transformation (GQEVT), which implements an e^{-βH} via a
block-encoding construction.

Each training iteration requires computing:

1. The **x-clamped Hamiltonian**  H(x)
2. The **(x, y)-clamped Hamiltonian**  H(x, y)
3. Expectation values under both effective Hamiltonians:
       E[O | x]       and       E[O | x, y]
4. An update: Δθ ∝ E[O | x, y] – E[O | x]

where O runs over all Pauli-term operators in the Hamiltonian.

Labels are encoded on Ny output qubits. Inputs occupy the remaining visible
qubits. Hidden qubits mediate higher-order correlations.

This file provides:
- full training loop with batching
- weight updates
- plotting utilities for 2D toy problems
- high-level prediction and accuracy_list functions
"""

from hamiltonian import *
from GQEVT import *
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import copy
from sklearn.metrics import accuracy_score


class QBM:
    """
    Discriminative Quantum Boltzmann Machine.

    This class implements the parameterized Ising-type Hamiltonian defined via ModelHamiltonian with a GQEVT-based Gibbs state preparation to 
    implement a supervised learning based classifier.

    Visible qubits are split into:
    - Ny output (label) qubits,
    - the remaining visible qubits that encode the input features x.

    Training uses a contrastive rule:
    - build the x-clamped Hamiltonian H(x),
    - build the (x, y)-clamped Hamiltonian H(x, y),
    - approximate the Gibbs state via GQEVT and compute expectations,
    - update parameters proportional to E_{model}[· | x, y] - E_{model}[· | x].

    
    """
    def __init__(
        self,
        n_hidden,
        n_visible,
        n_output,
        terms=None,
        connectivity='all',
        n=12,
        beta=3.0,
        seed=12,
        gamma_visible=0.5,
        allow_visible_x=True,
        sim_level=0,
        h_norm_trigger=14.0,
        h_norm_target=10.0,
        track_nll=True,
        gqevt_refresh_mode='threshold',
    ):
        """
         Internally this creates:
        - ModelHamiltonian object with the specified visible/hidden layout,
        - GQEVT object for the chosen β and polynomial degree,
        - a flattened parameter vector (self.weights).
        """
        
        
        self.hamiltonian=ModelHamiltonian(n_hidden,n_visible,terms=terms,n_output=n_output,connectivity=connectivity,seed=seed, allow_visible_x=allow_visible_x, gamma_visible=gamma_visible)
        
        self.n = n
        self.seed = seed
        self.beta = beta
        self.sim_level = sim_level
        self.h_norm_trigger = h_norm_trigger
        self.h_norm_target = h_norm_target
        self.track_nll = track_nll
        self.gqevt_refresh_mode = gqevt_refresh_mode
        self.alpha_safety = 1.05
        self.scaling = 1
        self.gqevt = GQEVT(n, -beta, 'rootfinding', scaling=self.scaling, seed=seed, sim_level=sim_level)
       
        self.n_hidden=self.hamiltonian.n_hidden
        self.n_visible=self.hamiltonian.n_visible
        
        self.n_qubits=self.n_hidden+self.n_visible
        self.Ny=self.hamiltonian.Ny
      

        self.weights=self.hamiltonian.get_weights()
        self.connectivity=connectivity

    def update_model_params(self, weights):
        # reorder weights for simplicity

        i = 0
        for term, values in self.hamiltonian.params.items():
            mask = self.hamiltonian.trainable_mask[term]
            m = np.sum(mask)
            if m > 0:
                new_vals = values.copy()
                new_vals[mask] = weights[i:i + m]
                self.hamiltonian.params[term] = new_vals
                i += m

    def _output_positive_configs(self, x, y):

        """
        Compute expectations for x-clamped and (x, y)-clamped Hamiltonians.

        Parameters
        ----------
        x : array-like
            Input data for the visible input qubits.
        y : array-like
            Output / label bitstring (in {0, 1}^Ny).

        Returns
        -------
        x_clamped : np.ndarray
            Normalized expectation values for the x-clamped Hamiltonian H(x).
        xy_clamped : np.ndarray
            Normalized expectation values for the (x, y)-clamped Hamiltonian
            H(x, y).

        Notes
        -----
        Internally, both Hamiltonians are built via hamiltonian.build_hamiltonians. Expectations are then computed and divided by the success
        probability (the first component) to condition on successful ancilla projection.
        """

        self.H_xy = self.hamiltonian.build_hamiltonians(x, y)

        xy_clamped = self._compute_expectation(self.H_xy)
        xy_clamped = np.array(xy_clamped) / xy_clamped[0]

        return xy_clamped

    def _energy_fully_clamped(self, x, y_pm1):
        """
        Exact scalar energy E(x,y) for the fully visible / fully clamped case.

        Assumes:
        - no hidden qubits
        - only Z-type terms matter
        - y_pm1 is already in {-1, +1}^Ny
        """
        if self.n_hidden != 0:
            raise ValueError("Exact fully-clamped energy helper only implemented for n_hidden=0.")

        vis = np.concatenate([y_pm1, x])  # order matches build_hamiltonians: [y, x]
        E = 0.0

        for term, conns in self.hamiltonian.connections.items():
            # only Z-only terms should contribute in this case
            if any(p != "Z" for p in term):
                continue

            params = self.hamiltonian.params[term]
            for k, conn in enumerate(conns):
                prod = 1.0
                for label in conn:
                    if label[0] != "v":
                        raise ValueError("Found hidden qubit in fully visible energy computation.")
                    vidx = int(label[1:])
                    prod *= vis[vidx]
                E += -params[k] * prod

        return float(E)

    def _logZ_given_x_exact(self, x):
        """
        Exact log Z(x) by summing over all label configurations.
        Only for n_hidden = 0.
        """
        if self.n_hidden != 0:
            raise ValueError("Exact logZ helper only implemented for n_hidden=0.")

        n_states = 2 ** self.Ny
        vals = []

        for s in range(n_states):
            bits = np.array(list(np.binary_repr(s, width=self.Ny)), dtype=int)
            y_pm1 = 2 * bits - 1
            E = self._energy_fully_clamped(x, y_pm1)
            vals.append(-self.beta * E)

        vals = np.asarray(vals, dtype=float)
        m = np.max(vals)
        return float(m + np.log(np.sum(np.exp(vals - m))))

    def _sample_nll_exact(self, x, y):
        """
        Exact NLL = log Z(x) - log exp(-beta E(x,y))
                  = log Z(x) + beta E(x,y)

        y is expected in {0,1} encoding.
        """
        y_pm1 = 2 * np.asarray(y) - 1
        E_xy = self._energy_fully_clamped(x, y_pm1)
        logZx = self._logZ_given_x_exact(x)
        return float(logZx + self.beta * E_xy)
            

    def _output_negative_configs(self, x):

        """
        Compute expectations for x-clamped and (x, y)-clamped Hamiltonians.

        Parameters
        ----------
        x : array-like
            Input data for the visible input qubits.
        y : array-like
            Output / label bitstring (in {0, 1}^Ny).

        Returns
        -------
        x_clamped : np.ndarray
            Normalized expectation values for the x-clamped Hamiltonian H(x).
        xy_clamped : np.ndarray
            Normalized expectation values for the (x, y)-clamped Hamiltonian
            H(x, y).

        Notes
        -----
        Internally, both Hamiltonians are built via hamiltonian.build_hamiltonians. Expectations are then computed and divided by the success
        probability (the first component) to condition on successful ancilla projection.
        """

        self.H_x=self.hamiltonian.build_hamiltonians(x, y=None)

        x_clamped=self._compute_expectation(self.H_x)
        x_clamped=np.array(x_clamped)/x_clamped[0]   

        return x_clamped

    def _rebuild_gqevt_after_error(self, error):
        print("GQEVT evaluation failed; rebuilding angles/state.")

        self.gqevt = GQEVT(
            self.n,
            -self.beta,
            "rootfinding",
            scaling=1,
            seed=self.seed,
            sim_level=self.sim_level,
        )

    def _compute_expectation(self, H, prediction_mode=False):

        """
        Apply GQEVT to H and measure expectation values.

        Parameters
        ----------
        H : qml.Hamiltonian
            PennyLane Hamiltonian representing either H(x) or H(x, y).
        prediction_mode : bool, optional
            If False (default), measure projector-augmented versions of all
            operators in H. If True, only measure PauliZ on the output
            label qubits, suitable for prediction.

        Returns
        -------
        list[float]
            A list of expectation values. The first entry corresponds to
            the success probability of the ancilla projector, and the
            remaining entries correspond either to all operators in H
            or only the label-qubit PauliZ operators (in prediction mode).

        Notes
        -----
        The Hamiltonian wires are re-mapped to leave the first two wires
        for ancilla/projector qubits. 
        """
        
        
        new_wires=range(2,self.n_qubits+2)
        old_wires=range(0,self.n_qubits)
        
        
        H_new = H.map_wires(dict(zip(old_wires,new_wires)))
        proj = qml.Projector( [0] * 2,wires=range(0,2))
        
        if prediction_mode==False:
        
            ops=H_new.ops
        else:
            ## Measure only output qubits for prediction
            ops=[qml.PauliZ(i) for i in range(self.n_hidden+2,self.n_hidden+self.Ny+2)]
        
        new_ops=[proj]+[proj@op for op in ops]
        
        H_matrix = qml.matrix(H_new)

        def _run_expectation(matrix):
            dev_xy = qml.device("default.qubit", wires=2*(len(H.wires))+2)
            self.gqevt.build(matrix)

            @qml.qnode(dev_xy, interface="autograd")
            def _compute():

                for wire in range(2,len(H.wires)+2):
                    qml.Hadamard(wire)
                    qml.CNOT([wire,wire+len(H.wires)])
                
               
                self.gqevt.circuit()
                #return qml.expval(qml.PauliZ(0))
                #return qml.state()
                return [qml.expval(op) for op in new_ops]

            return _compute()

        try:
            return _run_expectation(H_matrix)
        except Exception as error:
            if self.sim_level == 0 or self.gqevt_refresh_mode != 'on_error':
                raise

            self._rebuild_gqevt_after_error(error)
            return _run_expectation(H_matrix)

    def _calculate_sample_nll(self, x, y):
        """
        Calculates the NLL for a single sample: log Z(x) - log Z(x, y)
        Note: Success probability [0] from _compute_expectation is proportional to Z.
        """
        y_pm1 = 2 * np.asarray(y) - 1

        h_xy = self.hamiltonian.build_hamiltonians(x, y_pm1)
        z_xy = self._compute_expectation(h_xy)[0]

        h_x = self.hamiltonian.build_hamiltonians(x, y=None)
        z_x = self._compute_expectation(h_x)[0]

        nll = np.log(z_x + 1e-12) - np.log(z_xy + 1e-12)
        return nll

    def _xy_clamped_exact_expectations(self, x_vec, y_vec_pm1):
        """
        Exact expectations for the (x,y)-clamped phase under semirestricted (no hidden-hidden) + vh_only.

        Returns:
            expvals: np.ndarray aligned with self.hamiltonian.connections[] iteration order.
        """
        ham = self.hamiltonian
        nH = ham.n_hidden

        vis = np.concatenate([y_vec_pm1, x_vec])  # shape (n_visible,)

        hx = np.zeros(nH, dtype=float)
        hy = np.zeros(nH, dtype=float)
        hz = np.zeros(nH, dtype=float)

        def _hid_index(label):
            return int(label[1:])

        def _vis_index(label):
            return int(label[1:])

        for term, conns in ham.connections.items():
            params = ham.params[term]
            for k, conn in enumerate(conns):
                w = params[k]

                hidden_positions = [j for j, lab in enumerate(conn) if lab[0] == "h"]
                visible_positions = [j for j, lab in enumerate(conn) if lab[0] == "v"]

                pref = 1.0
                for j in visible_positions:
                    pref *= vis[_vis_index(conn[j])]

                if len(hidden_positions) == 0:
                    continue

                if len(hidden_positions) > 1:
                    raise ValueError(
                        "Found hidden-hidden interaction in clamped phase; exact factorized formula no longer applies.")

                hi = _hid_index(conn[hidden_positions[0]])
                pauli_on_hidden = term[hidden_positions[0]]

                if pauli_on_hidden == "X":
                    hx[hi] += w * pref
                elif pauli_on_hidden == "Y":
                    hy[hi] += w * pref
                elif pauli_on_hidden == "Z":
                    hz[hi] += w * pref
                else:
                    raise ValueError(f"Unknown Pauli {pauli_on_hidden}")

        beta = self.beta
        expX = np.zeros(nH, dtype=float)
        expY = np.zeros(nH, dtype=float)
        expZ = np.zeros(nH, dtype=float)

        for i in range(nH):
            r = np.sqrt(hx[i] ** 2 + hy[i] ** 2 + hz[i] ** 2)
            if r < 1e-12:
                expX[i] = expY[i] = expZ[i] = 0.0
            else:
                t = np.tanh(beta * r)
                expX[i] = (hx[i] / r) * t
                expY[i] = (hy[i] / r) * t
                expZ[i] = (hz[i] / r) * t

        expvals = []
        for term, conns in ham.connections.items():
            for conn in conns:
                hidden_positions = [j for j, lab in enumerate(conn) if lab[0] == "h"]
                visible_positions = [j for j, lab in enumerate(conn) if lab[0] == "v"]

                pref = 1.0
                for j in visible_positions:
                    pref *= vis[_vis_index(conn[j])]

                if len(hidden_positions) == 0:
                    # fully clamped visible product
                    expvals.append(pref)
                    continue

                if len(hidden_positions) > 1:
                    raise ValueError(
                        "Hidden-hidden interaction encountered; cannot use factorized exact clamped phase.")

                hi = _hid_index(conn[hidden_positions[0]])
                p = term[hidden_positions[0]]
                if p == "X":
                    expvals.append(pref * expX[hi])
                elif p == "Y":
                    expvals.append(pref * expY[hi])
                elif p == "Z":
                    expvals.append(pref * expZ[hi])
                else:
                    raise ValueError(f"Unknown Pauli {p}")

        return np.array(expvals, dtype=float)

    def _flat_trainable_mask(self):
        """Boolean mask aligned with order_match_configs output ordering."""
        mask = []
        for term, conns in self.hamiltonian.connections.items():
            tmask = self.hamiltonian.trainable_mask.get(term, None)
            if tmask is None:
                mask.extend([True] * len(conns))
            else:
                mask.extend(list(tmask))
        return np.array(mask, dtype=bool)
        
        
    def update_weights(self,x_batch,y_batch,learning_rate):
        
        
        """
        Perform one parameter update step on a batch.

        Parameters
        ----------
        x_batch : np.ndarray
            Batch of input samples with shape (batch_size, n_x).
        y_batch : np.ndarray
            Batch of corresponding labels encoded as bitstrings of shape
            (batch_size, Ny) with entries in {0, 1}.
        learning_rate : float
            Learning rate for the parameter update.

        Returns
        -------
        np.ndarray
            The averaged error vector used to update the weights.
        """
        
        errors=0 
        
      
        for i,x_vector in enumerate(x_batch):
            
           
            y_vector=y_batch[i]
            y_vector=2*y_vector-1
            
           
            if self.connectivity == 'vh_only':
                xy_clamped=self._xy_clamped_exact_expectations(x_vector,y_vector)
            else:
                xy_clamped=self._output_positive_configs(x_vector,y_vector)
                xy_clamped = xy_clamped[1:]

            x_clamped = self._output_negative_configs(x_vector)
            x_clamped=x_clamped[1:]


            if self.connectivity == 'vh_only':
                dummy = np.zeros_like(x_clamped)
                new_x_clamped, _ = self.hamiltonian.order_match_configs(
                    x_clamped, dummy, x_vector, y_vector,
                    self.H_x.terms()[-1], self.H_x.terms()[-1]
                )
                new_xy_clamped = xy_clamped
            else:
                new_x_clamped,new_xy_clamped= self.hamiltonian.order_match_configs(x_clamped,xy_clamped,x_vector,y_vector,self.H_x.terms()[-1],self.H_xy.terms()[-1])
            
           
            
            errors+= (new_xy_clamped - new_x_clamped)

        errors /= x_batch.shape[0]

        train_mask = self._flat_trainable_mask()

        if errors.shape[0] != train_mask.shape[0]:
            raise ValueError(f"Mismatch: errors has {errors.shape[0]} entries but train_mask has {train_mask.shape[0]}")

        errors_train = errors[train_mask]

        self.weights = self.weights + learning_rate * errors_train

        self.update_model_params(self.weights)

        return errors_train

    def predict(self,x):
        
        """
        Predict the class label for a single input sample.

        Parameters
        ----------
        x : array-like
            Input feature vector

        Returns
        -------
        output_vals : np.ndarray
            Expectation values of PauliZ on the output qubits (after
            conditioning on ancilla success).
        prediction : int
            Predicted class index obtained by thresholding the output
            expectations to bits and decoding them as a binary integer.
        """
        
        
        
        H_x=self.hamiltonian.build_hamiltonians(x, y=None)
       
        
    
        ops=[qml.PauliZ(i) for i in range(self.n_hidden,self.n_hidden+self.Ny)]
        
        x_clamped = self._compute_expectation(H_x,prediction_mode=True) 
        
        x_clamped=np.array(x_clamped)/x_clamped[0]

        #indices=[H_x.terms()[-1].index(op) for op in ops]

        output_vals=np.array(x_clamped[1:])

        if self.Ny == 1:
            prediction = int((np.sign(output_vals[0]) + 1) / 2)
        else:
            prediction = int(np.argmax(output_vals))

        return output_vals, prediction
     
    def predict_test(self,X_test,y_test):
        predictions=[] 
        for x_vector in X_test:
            
             exp_y,y_predict=self.predict(x_vector)
             predictions.append(y_predict)

        acc_sklearn = accuracy_score(y_test, predictions)
        return acc_sklearn
    
    def predict_test_labels(self,X_test,y_test,plot=False):
        predictions=[] 
        for x_vector in X_test:
            exp_y, y_predict = self.predict(x_vector)
            predictions.append(y_predict)

        def to_cat(Y):
            Y = np.asarray(Y)
            if Y.ndim == 1:
                return Y.astype(int)
            if Y.shape[1] == 1:
                return Y.reshape(-1).astype(int)
            return np.argmax(Y, axis=1).astype(int)

        y_cat = to_cat(y_test)

        acc_sklearn = accuracy_score(y_cat, predictions)

        if plot == True:

            if self.Ny == 1:
                x_min = np.min(X_test[:, 0]) - 0.5
                x_max = np.max(X_test[:, 0]) + 0.5
                y_min = np.min(X_test[:, 1]) - 0.5
                y_max = np.max(X_test[:, 1]) + 0.5
                h = 0.3
                xx, yy = np.meshgrid(
                    np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h)
                )

                Z = np.zeros(xx.size, dtype=int)
                for i, (gx, gy) in enumerate(np.c_[xx.ravel(), yy.ravel()]):
                    Z[i] = self.predict(np.array(np.array([gx, gy])))[1]

                Z = Z.reshape(xx.shape)

            plt.figure()
            plt.subplot(1, 2, 1)
            plt.title('Predictions')
            plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='viridis', marker='x', label="Test")

            if self.Ny == 1:
                plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)

            plt.subplot(1, 2, 2)
            plt.title('Ground truth')
            plt.scatter(X_test[:, 0], X_test[:, 1], c=to_cat(y_test), cmap='viridis', marker='x', label="Test")
            plt.show()

        return acc_sklearn

    def train_model(self, X, y_data, X_test, y_test, batch_size=8, learning_rate=0.005, epochs=3, plot=False):
        """
        Training loop for the QBM tracking both Gradient Norm and Negative Log-Likelihood.

        Returns
        -------
        losses : list[list[float]]
            Per-epoch list of L2 norms of the weight updates (gradient norm).
        nll_history : list[float]
            Average Negative Log-Likelihood per epoch.
        """
        self.accuracy_list = []
        self.nll_history = []
        self.epochs = epochs

        if len(X[0]) + y_data.shape[1] != self.n_visible:
            raise ValueError(f"Insufficient visible nodes for dataset")

        data = X
        batch_num = data.shape[0] // batch_size
        diff = data.shape[0] % batch_size
        self.batch_size = batch_size

        if diff:
            data_main = data[:-diff]
            y_data_main = y_data[:-diff]
            last_batch = data[-diff:]
            last_ybatch = y_data[-diff:]
            x_batches = np.vsplit(data_main, batch_num)
            y_batches = np.vsplit(y_data_main, batch_num)
            x_batches.append(last_batch)
            y_batches.append(last_ybatch)
        else:
            x_batches = np.vsplit(data, batch_num)
            y_batches = np.vsplit(y_data, batch_num)

        losses = []

        for e in tqdm(range(1, self.epochs + 1), desc="Epochs", leave=True):
            errors_epoch = []
            nll_epoch_accumulator = []

            for i, x_batch in tqdm(enumerate(x_batches), desc=f"Batch (Epoch {e})", total=len(x_batches), leave=False):
                y_batch = y_batches[i]

                if self.track_nll:
                    batch_nlls = []
                    for sample_idx in range(x_batch.shape[0]):
                        x_sample = x_batch[sample_idx]
                        y_sample = y_batch[sample_idx]

                        if self.n_hidden == 0:
                            val = self._sample_nll_exact(x_sample, y_sample)
                        else:
                            val = self._calculate_sample_nll(x_sample, y_sample)

                        batch_nlls.append(val)

                    nll_epoch_accumulator.append(np.mean(batch_nlls))

                errors = self.update_weights(x_batch, y_batch, learning_rate)
                errors_epoch.append(np.linalg.norm(errors))

                if self.gqevt_refresh_mode == 'threshold' and self.gqevt.sim_level == 2:
                    self.H_norm = np.linalg.norm(self.weights)
                    if self.H_norm <= self.h_norm_trigger:
                        continue
                    print("recalculating GQSP angles")
                    rescale_factor = self.H_norm / self.h_norm_target

                    self.weights = self.weights / rescale_factor
                    self.update_model_params(self.weights)

                    self.scaling *= rescale_factor
                    self.gqevt = GQEVT(
                        self.n,
                        -self.beta,
                        'rootfinding',
                        scaling=self.scaling,
                        seed=self.seed,
                        sim_level=self.sim_level,
                    )

            avg_nll = np.mean(nll_epoch_accumulator) if self.track_nll else np.nan
            self.nll_history.append(float(avg_nll))
            losses.append(errors_epoch)

            acc = self.predict_test_labels(X_test, y_test, plot)
            self.accuracy_list.append(acc)

            print(f"Epoch {e}/{self.epochs} | NLL: {avg_nll:.4f} | Test Acc: {acc:.4f}")

        return losses, self.nll_history







