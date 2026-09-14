"""
This module defines the class, ``ModelHamiltonian`` which generate PennyLane ``qml.Hamiltonian`` objects for
a QBM with visible (data) units, hidden units, and optional output units.

Qubits are labeled logically as:

- Hidden units :  "h0", "h1", ..., "h{n_hidden-1}"
- Visible units:  "v0", "v1", ..., "v{n_visible-1}"

On the PennyLane device, these logical labels are mapped to wire indices as:

- Hidden wires : 0 ... n_hidden - 1
- Visible wires: n_hidden ... n_hidden + n_visible - 1

For discriminative models with ``n_output > 0``, the first ``n_output`` visible
qubits are treated as output (label) qubits and can be clamped separately from
the remaining visible "input" qubits.

Main functionality
------------------

-Generate all allowed Pauli-word "connections" between hidden/visible qubits
  for a given list of Pauli terms (e.g. ``["Z", "ZZ", "X", "XZ"]``).
- Randomly initialize trainable parameters (couplings) for each connection.
- Build input-/label-dependent effective Hamiltonians for
  - x-clamped circuits (only data clamped),
  - xy-clamped circuits (data and labels clamped).
- Reduce operators when some wires are clamped and re-order expectation values
  across different clamping configurations, to construct the weight update
  vector used in QBM training.

The non-Z operators
(“X”, “Y”) only act on hidden units, and visible-only couplings are disabled for
a subset of visible nodes (useful to distinguish input vs. output qubits).

Usage
-------------



    from hamiltonian import ModelHamiltonian

    # Create model with 3 hidden, 4 visible (incl. 1 output), and ZZ/Z couplings
    ham_builder = ModelHamiltonian(
        n_hidden=3,
        n_visible=4,
        n_output=1,
        terms=["Z", "ZZ"],
        connectivity="all",
    )

    x = np.array([+2, -1, +1])      # input configuration (Ising spins)
    y = np.array([+1])              # label configuration (optional)

    H_x  = ham_builder.build_hamiltonians(x)       # x-clamped Hamiltonian
    H_xy = ham_builder.build_hamiltonians(x, y)    # xy-clamped Hamiltonian

These Hamiltonians can then be used inside PennyLane QNodes to evaluate model
expectations, log-likelihoods, and gradients.
"""

import itertools
import numpy as np
import pennylane as qml
import random
operator_list = {'X': qml.PauliX, 'Y': qml.PauliY, 'Z': qml.PauliZ}

class ModelHamiltonian:
    
    '''
    This class provides functionality to build arbitrary Hamiltonians for the QBM circuit.

    Attributes:
        
        n_hidden (int): Number of hidden qubits.
        n_visible (int): Number of visible qubits. (Equal to num_features+ log2(num_class_labels))
        
        Ny (int): Number of output qubits.
        
        terms (List[str]): Pauli terms to include in Hamiltonian (e.g., 'Z', 'ZZ').
        
        params (dict): Dictionary of trainable parameters per term.
        
        connections (dict): qubit combinations for each Pauli word.
    
    Notes:
    -----
    * Non-Z operators (X, Y) are **not** allowed on visible qubits. They may
      only act on hidden units. This is enforced in
      :meth:`get_valid_combinations`.
    * For Z-type terms, visible qubits corresponding to unclamped inputs or
      outputs are “absorbed” into a classical prefactor (their current ±1
      configuration), and the remaining quantum operator is reduced to the
      unclamped subset.
    * The resulting Hamiltonians are automatically simplified using
      ``qml.Hamiltonian.simplify``.
    
    Example:
        
        >>> model = ModelHamiltonian(n_hidden=3, n_visible=4, terms=['Z', 'ZZ'])
        
        >>> hamiltonian = model.build_hamiltonians(x,y)

    '''
    
    def __init__(self, n_hidden: int, n_visible: int, terms: list[str] = None, n_output: int = 1, connectivity: str = 'all',seed=12, allow_visible_x: bool = True,
        gamma_visible=0.5,):
        
        
        
        self.terms = terms if terms is not None else ['Z', 'ZZ']
         
        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.Ny=n_output
        self.output_wires=range(self.n_hidden,self.n_hidden+self.Ny)
        self.allow_visible_x = allow_visible_x

        if allow_visible_x:
            gv = np.asarray(gamma_visible, dtype=float)
            if gv.ndim == 0:
                gv = np.full(self.Ny, float(gv))  # ONLY for label qubits
            if gv.shape != (self.Ny,):
                raise ValueError("gamma_visible must be scalar or shape (Ny,)")

            self.gamma_visible = gv
        self.generate_connections(connectivity=connectivity)
        
        self.initialize_params(seed=seed)
    
    
    def generate_connections(self, connectivity='all'):
        """
        Generates valid qubit connections as a dictionary for each term in the Hamiltonian.
        For each Pauli word in :attr:`terms`, this method: Enumerates all combinations of hidden and visible qubits of the
        appropriate length, Filters out combinations where a non-Z operator (X or Y) would act on
        a visible qubit.
        """
        qubits_hidden = ['h' + str(i) for i in range(self.n_hidden)]
        qubits_visible = ['v' + str(i) for i in range(self.n_visible)]
        qubits = qubits_hidden + qubits_visible

        self.connections = {}
        for term in self.terms:
            m = len(term)

            # Default: non-Z terms act only on hidden
            if 'Z' in term:
                options = qubits
            else:
                if self.allow_visible_x and term in ("X", "Y") and len(term) == 1:
                    # allow X only on hidden + LABEL visibles
                    label_qubits = ['v' + str(i) for i in range(self.Ny)]
                    options = qubits_hidden + label_qubits
                else:
                    options = qubits_hidden

            combinations = list(itertools.combinations(options, m))
            valid_combinations = self.get_valid_combinations(combinations, term, connectivity=connectivity)
            self.connections[term] = valid_combinations

    def get_valid_combinations(self, combinations, term, connectivity="all"):
        """
        Filters combinations/permutations:
        - existing rule: non-Z operators cannot act on visible qubits
        - NEW rule for connectivity="vh_only":
            * multi-qubit terms: at most one hidden qubit in the whole connection
            * (optional but recommended) multi-qubit terms must include exactly one hidden,
              i.e. enforce visible-hidden-only interactions
        """
        valid_combinations = []

        non_Z_indices = [i for i, letter in enumerate(term) if letter != "Z"]

        term_permutations = list(itertools.permutations(term))
        index_duplicates = [i for i, t in enumerate(term_permutations) if t in term_permutations[:i]]

        # Your existing "reject visible-only between input nodes" rule for Z-only terms
        reject_terms = ["v" + str(i) for i in range(self.Ny, self.n_visible)]

        for combi in combinations:
            # Existing: remove connections between input nodes for Z-only terms
            if non_Z_indices == []:
                if set(combi).intersection(set(reject_terms)) == set(combi):
                    continue

            # NEW: vh_only = semirestricted = "no hidden-hidden"
            if connectivity == "vh_only":
                hidden_count = sum(1 for q in combi if q[0] == "h")

                # For multi-qubit interactions, disallow >1 hidden qubit
                # (this forbids hidden-hidden and any term that couples multiple h's)
                if len(combi) > 1 and hidden_count > 1:
                    continue

                    # (visible_count is then >= 1 automatically)
                else:
                    # single-body terms: allow h0 (hidden field), allow v0 (visible bias) if term is Z
                    # but if you want to disallow visible-only single body when using vh_only, uncomment:
                    # if visible_count == 1: continue
                    pass

            permutations = list(itertools.permutations(combi))
            for k, permutation in enumerate(permutations):
                if k in index_duplicates:
                    continue

                # Non-Z on visible is only allowed for single-body X/Y terms (transverse field)
                if non_Z_indices:
                    if not (self.allow_visible_x and len(term) == 1 and term in ("X", "Y")):
                        ok = True
                        for idx in non_Z_indices:
                            if permutation[idx][0] == "v":
                                ok = False
                                break
                        if not ok:
                            continue

                valid_combinations.append(permutation)

        return valid_combinations

    def initialize_params(self, seed):
        np.random.seed(seed)
        random.seed(seed)

        self.params = {}
        self.trainable_mask = {}

        for term in self.terms:
            n = len(self.connections[term])

            p = np.random.rand(n)
            p = p - np.mean(p) / 2

            mask = np.ones(n, dtype=bool)

            # Freeze visible X/Y single-body terms (labels only)
            if self.allow_visible_x and term in ("X", "Y") and len(term) == 1:
                for k, conn in enumerate(self.connections[term]):
                    q = conn[0]
                    if q[0] == "v":  # visible
                        vidx = int(q[1:])
                        if vidx < self.Ny:  # label qubit
                            p[k] = self.gamma_visible[vidx]
                            mask[k] = False  # not trainable

            self.params[term] = p
            self.trainable_mask[term] = mask
    
            
    def qubit_number(self, qubit):
        
        if qubit[0] == 'h':
            return int(qubit[1])
        else:
            return int(qubit[1]) + self.n_hidden

    def operators(self, term, connection):
        
        assert len(term) == len(connection)
        for i, op in enumerate(term):
            num = self.qubit_number(connection[i])
            
            if i == 0:
                a = operator_list[op](num)
            else:
                a = a @ operator_list[op](num)
        return a
    
    def fix_visible(self,connection,x,y):

        for label in connection:
            if 'v' in label:
                pass        
        
    
    def build_hamiltonians(self,x,y=None):
        """
            Build the parameterized Hamiltonian based on input (and optionally output) data.
            
            Args:
                x: array-like, visible unit (input nodes) configuration 
                y: array-like or None, output data (class data)
                
            Returns:
                
                PennyLane Hamiltonian object
        """
       
        if y is None:        
            
            X=x
            
            reject_terms=['v'+str(i) for i in range(self.Ny,self.n_visible)]
            
        else:    
            X=np.concatenate((y,x))
            reject_terms=['v'+str(i) for i in range(self.n_visible)]
        
   
        coeffs, ops = [], []
    
        # Assigning pauli word for each connection, substituting values for visible and output qubits. 
      
        for pauli_term in self.terms:
         
            
            if 'Z' not in pauli_term:
                
                for i,coefficient in enumerate(self.params[pauli_term]):
                
                    coeffs.append(-coefficient)
                    ops.append(self.operators(pauli_term,self.connections[pauli_term][i]))
        
            else:
                m=len(pauli_term)
                # --- ADD THIS helper ONCE per pauli_term loop (or per connection loop) ---
                if y is None:
                    # X is inputs only (length = n_visible - Ny); vNy.. maps to X[vidx-Ny]
                    def get_val(label):
                        vidx = int(label[1:])
                        return X[vidx - self.Ny]
                else:
                    # X is full visible assignment [y, x] (length = n_visible); v0.. maps to X[vidx]
                    def get_val(label):
                        vidx = int(label[1:])
                        return X[vidx]
               
                for i,connection in enumerate(self.connections[pauli_term]):
                         
                    factor=1
                    count=0
                    new_connection=[]
                    for label in connection:
                        
                        
                        if any(sub in label for sub in reject_terms):

                            factor *= get_val(label)
                            count+=1
                           
                        else:
                            new_connection.append(label)
                            
                          
                                
                    
                    if count==0:
                        coeffs.append(-self.params[pauli_term][i])
                        ops.append(self.operators(pauli_term,self.connections[pauli_term][i]))
                    else:
                        
                        new_term=pauli_term[count:]
                        
                        if new_term != '':    
                        
                            coeffs.append(-self.params[pauli_term][i]*factor)
                           
                            
                            new_operator=self.operators(new_term,new_connection)
                            ops.append(new_operator)
                       
        
        hamiltonian=qml.Hamiltonian(coeffs,ops)        
          
        
        
        return hamiltonian.simplify()


    def remove_operators_on_wires(self,operator_terms, wires_to_remove):
        
        """
        Remove the action of an operator on a given set of wires.

        This is used when some qubits are clamped to classical values: operator
        factors acting on those wires are removed from the quantum operator and
        their contribution is tracked separately via a classical prefactor.

        Parameters
        ----------
        operator_terms : list[qml.operation.Operator]
            List of multi-qubit operators to be reduced.
        wires_to_remove : Iterable[int]
            Wire indices whose Pauli factors should be removed.

        Returns
        -------
        new_ops : list[qml.operation.Operator]
            Reduced operators that only act on the remaining wires (or
            ``qml.Identity`` if all wires were removed).
        removed_wires_list : list[list[int]]
            For each input operator, the list of wires that were removed.
        """
        
        #wires_to_remove = set(wires_to_remove)
        removed_wires_list = []
        new_ops = []
        
        for word in operator_terms:
            wires=word.wires.tolist()
            new_word_ops=[]
            new_word=word
           
            
            if any(x in wires_to_remove for x in wires):
                new_word= qml.Identity(wires[0])
               # removed_wires= list(set(wires) & set(wires_to_remove))
                
                for i,wire in enumerate(wires):
                   
                    
                    if wire not in wires_to_remove:
                      
                        new_word_ops.append(word[i])
                
            
                for i,k in enumerate(new_word_ops):
                    new_word=new_word@k
            
            removed_wires= list(set(wires) & set(wires_to_remove))
            
            new_ops.append(new_word.simplify())
    
            removed_wires_list.append(removed_wires)
        return new_ops,removed_wires_list
            
    
    def get_weights(self):
        
        """
            Retreive array of weights from the parameter dictionary .
            
            Args:
               
                
            Returns:
                
            nd.array
        """
        weights = []
        for term, values in self.params.items():
            mask = self.trainable_mask[term]
            weights.append(values[mask])
        return np.concatenate(weights) if weights else np.array([])
    
    def order_match_configs(self,x_clamped,xy_clamped,x,y,terms_x,terms_xy):

        """
        Reorder expectation values from x- and xy-clamped circuits.

        Given lists of expectation values and corresponding operators from
        separate circuits (x-clamped and xy-clamped), this method:

        * Reduces each operator according to which qubits are clamped.
        * Matches reduced operators between the two circuits.
        * Multiplies matched expectations by the appropriate classical
          prefactors (products of clamped spins).
        * Returns expectation arrays ordered according to
          :attr:`connections`, ready to be used in the gradient / weight
          update rule.

        Parameters
        ----------
        x_clamped : np.ndarray
            Expectation values from the x-clamped circuit.
        xy_clamped : np.ndarray
            Expectation values from the xy-clamped circuit.
        x : array-like
            Input data configuration (visible inputs).
        y : array-like
            Output / label configuration.
        terms_x : list[qml.operation.Operator]
            List of operators measured in the x-clamped circuit.
        terms_xy : list[qml.operation.Operator]
            List of operators measured in the xy-clamped circuit.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - Reordered x-clamped expectations.
            - Reordered xy-clamped expectations.

        These arrays are aligned with each parameter in :attr:`params` and can
        be plugged directly into the chosen learning rule.
        """
           
        
        X=np.concatenate((y,x))
        self.visible_wires=range(self.n_hidden,self.n_hidden+self.n_visible)
        self.input_wires=range(self.n_hidden+self.Ny,self.n_hidden+self.n_visible)
        new_x_clamped=[]
        new_xy_clamped=[]
        for term,connections in self.connections.items():
            for connection in connections:
                
                
                operator=self.operators(term,connection)
            
                
                reduced_operator_x,removed_wires_x=self.remove_operators_on_wires([operator],self.input_wires)
                reduced_operator_xy, removed_wires_xy=self.remove_operators_on_wires([operator],self.visible_wires)
                prefactor_x=1
                prefactor_xy=1
               
                ## Assign values for clamped qubits 
                for wire in removed_wires_x[0]:
                    prefactor_x*= X[wire-self.n_hidden]
                for wire in removed_wires_xy[0]:
                    prefactor_xy*=X[wire-self.n_hidden]
               
            
                try:
                    index_x=terms_x.index(reduced_operator_x[0])
                   
                    value_x= x_clamped[index_x]*prefactor_x
                except:
                    index_x=None
                    value_x=prefactor_x
                
                
                
                try:
                    
                    index_xy=terms_xy.index(reduced_operator_xy[0])
                    
                    value_xy= xy_clamped[index_xy]*prefactor_xy
                
                except:
                    index_xy=None
                    value_xy=prefactor_xy

                new_x_clamped.append(value_x)
                new_xy_clamped.append(value_xy)
        return np.array(new_x_clamped),np.array(new_xy_clamped)
               
                

class modelHamiltonian_old:
    def __init__(self, n_hidden, n_visible, terms=['Z','ZZ'], n_output=1,connectivity='all'):
        self.terms = terms
        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.Ny=n_output
        self.output_wires=range(self.n_hidden,self.n_hidden+self.Ny)
        if connectivity == 'all':
            self.generate_connections()
        self.initialize_params()
    def generate_connections(self):
        qubits_hidden = ['h'+str(i) for i in range(self.n_hidden)]
        qubits_visible = ['v'+str(i) for i in range(self.n_visible)]
        qubits = qubits_hidden + qubits_visible
        self.connections = {}
        for term in self.terms:
            m = len(term)
            options = qubits if 'Z' in term else qubits_hidden
            combinations = list(itertools.combinations(options, m))
            self.connections[term] = combinations

    def initialize_params(self):
        self.params = {}
        for term in self.terms:
            self.params[term] = np.random.rand(len(self.connections[term]))

    def qubit_number(self, qubit):
        if qubit[0] == 'h':
            return int(qubit[1])
        else:
            return int(qubit[1]) + self.n_hidden

    def operators(self, term, connection):
        assert len(term) == len(connection)
        for i, op in enumerate(term):
            num = self.qubit_number(connection[i])
            
            if i == 0:
                a = operator_list[op](num)
            else:
                a = a @ operator_list[op](num)
        return a
    
    def fix_visible(self,connection,x,y):

        for label in connection:
            if 'v' in label:
                pass        
        
    
    def build_hamiltonians(self,x,y=None):
        
       
        if y is None:        
            X=x
            reject_terms=['v'+str(i) for i in range(self.Ny,self.n_visible)]
            
        else:    
            X=np.concatenate((x,y))
            reject_terms=['v'+str(i) for i in range(self.n_visible)]
        
   
        coeffs, ops = [], []
    
        # transverse-x‐field term: − ∑ Γx_a σ^x_a
      
        for pauli_term in self.terms:
         
           
            if 'Z' not in pauli_term:
                
                for i,coefficient in enumerate(self.params[pauli_term]):
                
                    coeffs.append(-coefficient)
                    ops.append(self.operators(pauli_term,self.connections[pauli_term][i]))
        
            else:
                m=len(pauli_term)
               
                for i,connection in enumerate(self.connections[pauli_term]):
                         
                    factor=1
                    count=0
                    new_connection=[]
                    for label in connection:
                        
                        
                        if any(sub in label for sub in reject_terms):
                           
                           
                            factor*=X[int(label[1])-self.Ny]
                            count+=1
                           
                        else:
                            new_connection.append(label)
                            
                          
                                
                    
                    if count==0:
                        coeffs.append(-self.params[pauli_term][i])
                        ops.append(self.operators(pauli_term,self.connections[pauli_term][i]))
                    else:
                        
                        new_term=pauli_term[count:]
                        
                        if new_term != '':    
                        
                            coeffs.append(-self.params[pauli_term][i]*factor)
                           
                            
                            new_operator=self.operators(new_term,new_connection)
                            ops.append(new_operator)
                       
        hamiltonian=qml.Hamiltonian(coeffs,ops)        
          
       
        
        return hamiltonian.simplify()


    def remove_operators_on_wires(self,operator_terms, wires_to_remove):
        """Remove terms from an operator that act on any of the given wires."""
        #wires_to_remove = set(wires_to_remove)
        removed_wires_list = []
        new_ops = []
        
        for word in operator_terms:
            wires=word.wires.tolist()
            new_word_ops=[]
            new_word=word
           
            
            if any(x in wires_to_remove for x in wires):
                new_word= qml.Identity(wires[0])
               # removed_wires= list(set(wires) & set(wires_to_remove))
                
                for i,wire in enumerate(wires):
                   
                    
                    if wire not in wires_to_remove:
                      
                        new_word_ops.append(word[i])
                
            
                for i,k in enumerate(new_word_ops):
                    new_word=new_word@k
            
            removed_wires= list(set(wires) & set(wires_to_remove))
            
            new_ops.append(new_word.simplify())
    
            removed_wires_list.append(removed_wires)
        return new_ops,removed_wires_list
            
    
    def get_weights(self):
        weights=np.array([])
        for term,values in self.params.items():
            weights=np.concatenate((weights,values))

        return weights
    
    def order_match_configs(self,x_clamped,xy_clamped,x,y,terms_x,terms_xy):

        X=np.concatenate((y,x))
        self.visible_wires=range(self.n_hidden,self.n_hidden+self.n_visible)
        self.input_data_wires=range(self.n_hidden+self.Ny,self.n_hidden+self.n_visible)
        new_x_clamped=[]
        new_xy_clamped=[]
        for term,connections in self.connections.items():
            for connection in connections:
                
                operator=self.operators(term,connection)
            
                reduced_operator_x,removed_wires_x=self.remove_operators_on_wires([operator],self.input_data_wires)
                reduced_operator_xy, removed_wires_xy=self.remove_operators_on_wires([operator],self.visible_wires)
                prefactor_x=1
                prefactor_xy=1
               
                for wire in removed_wires_x[0]:
                    prefactor_x*= X[wire-self.n_hidden]
                for wire in removed_wires_xy[0]:
                    prefactor_xy*=X[wire-self.n_hidden]
               
            
                try:
                    index_x=terms_x.index(reduced_operator_x[0])
                   
                    value_x= x_clamped[index_x]*prefactor_x
                except:
                    index_x=None
                    value_x=prefactor_x
                
                try:
                    index_xy=terms_xy.index(reduced_operator_xy[0])
                    value_xy= xy_clamped[index_xy]*prefactor_xy
                
                except:
                    index_xy=None
                    value_xy=prefactor_xy

                new_x_clamped.append(value_x)
                new_xy_clamped.append(value_xy)
        return np.array(new_x_clamped),np.array(new_xy_clamped)
               
