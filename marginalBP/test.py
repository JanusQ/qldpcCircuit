import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
import itertools

class QLDPC_BP_Marginals:
    def __init__(self, D_prime, D_L, s_prime, weights):
        """
        Initialize BP for QLDPC logical syndrome marginal estimation
        
        Args:
            D_prime: Syndrome constraint matrix (m x n)
            D_L: Logical syndrome constraint matrix (k x n) 
            s_prime: Observed syndrome vector (length m)
            weights: Log-likelihood weights w_j for each bit (length n)
        """
        self.D_prime = csr_matrix(D_prime)
        self.D_L = csr_matrix(D_L)
        self.s_prime = s_prime
        self.weights = weights
        
        self.n_vars = D_prime.shape[1]  # number of error bits
        self.n_syndromes = D_prime.shape[0]  # number of syndrome constraints
        self.n_logical = D_L.shape[0]  # number of logical syndrome bits
        
        # Build factor graph connectivity
        self._build_factor_graph()
        
        # Initialize messages
        self._initialize_messages()
    
    def _build_factor_graph(self):
        """Build factor graph connectivity lists"""
        # Variable to syndrome check connections
        self.var_to_syn_checks = defaultdict(list)
        self.syn_check_to_vars = defaultdict(list)
        
        for i in range(self.n_syndromes):
            vars_in_check = self.D_prime.getrow(i).nonzero()[1]
            self.syn_check_to_vars[i] = list(vars_in_check)
            for j in vars_in_check:
                self.var_to_syn_checks[j].append(i)
        
        # Variable to logical check connections  
        self.var_to_log_checks = defaultdict(list)
        self.log_check_to_vars = defaultdict(list)
        
        for i in range(self.n_logical):
            vars_in_check = self.D_L.getrow(i).nonzero()[1]
            self.log_check_to_vars[i] = list(vars_in_check)
            for j in vars_in_check:
                self.var_to_log_checks[j].append(i)
    
    def _initialize_messages(self):
        """Initialize all messages"""
        # Messages from variables to syndrome checks
        self.msg_var_to_syn = defaultdict(lambda: defaultdict(lambda: np.array([0.5, 0.5])))
        
        # Messages from syndrome checks to variables  
        self.msg_syn_to_var = defaultdict(lambda: defaultdict(lambda: np.array([0.5, 0.5])))
        
        # Messages from logical syndrome variables to logical checks
        self.msg_sL_to_log = defaultdict(lambda: defaultdict(lambda: np.array([0.5, 0.5])))
        
        # Messages from logical checks to logical syndrome variables
        self.msg_log_to_sL = defaultdict(lambda: defaultdict(lambda: np.array([0.5, 0.5])))
        
        # Messages from variables to logical checks (through logical syndrome variables)
        self.msg_var_to_log = defaultdict(lambda: defaultdict(lambda: np.array([0.5, 0.5])))
        
        # Messages from logical checks to variables (through logical syndrome variables)
        self.msg_log_to_var = defaultdict(lambda: defaultdict(lambda: np.array([0.5, 0.5])))
        
        # Initialize variable messages with prior
        for j in range(self.n_vars):
            prior = np.array([1.0, np.exp(-self.weights[j])])
            prior = prior / np.sum(prior)
            
            for i in self.var_to_syn_checks[j]:
                self.msg_var_to_syn[j][i] = prior.copy()
            
            for i in self.var_to_log_checks[j]:
                self.msg_var_to_log[j][i] = prior.copy()
        
        # Initialize logical syndrome variable messages (uniform prior)
        for i in range(self.n_logical):
            uniform = np.array([0.5, 0.5])
            self.msg_sL_to_log[i][i] = uniform.copy()
    
    def _update_syndrome_check_messages(self):
        """Update messages from syndrome checks to variables"""
        for i in range(self.n_syndromes):
            vars_in_check = self.syn_check_to_vars[i]
            target_parity = self.s_prime[i]
            
            for j in vars_in_check:
                # Compute message from check i to variable j
                other_vars = [v for v in vars_in_check if v != j]
                
                if len(other_vars) == 0:
                    # Only one variable in check - direct constraint
                    if target_parity == 0:
                        self.msg_syn_to_var[i][j] = np.array([1.0, 0.0])
                    else:
                        self.msg_syn_to_var[i][j] = np.array([0.0, 1.0])
                    continue
                
                # Compute probability that other variables have even/odd parity
                prob_even = 0.0
                prob_odd = 0.0
                
                # Use efficient convolution for parity computation
                parity_dist = np.array([1.0, 0.0])  # Start with even parity
                
                for var_idx in other_vars:
                    var_prob = self.msg_var_to_syn[var_idx][i]
                    new_parity_dist = np.zeros(2)
                    
                    # Convolution for XOR operation
                    new_parity_dist[0] = parity_dist[0] * var_prob[0] + parity_dist[1] * var_prob[1]
                    new_parity_dist[1] = parity_dist[0] * var_prob[1] + parity_dist[1] * var_prob[0]
                    
                    parity_dist = new_parity_dist
                
                prob_even = parity_dist[0]
                prob_odd = parity_dist[1]
                
                # Message constrains variable j based on target parity
                if target_parity == 0:
                    # Need even total parity
                    self.msg_syn_to_var[i][j] = np.array([prob_even, prob_odd])
                else:
                    # Need odd total parity  
                    self.msg_syn_to_var[i][j] = np.array([prob_odd, prob_even])
                
                # Normalize
                norm = np.sum(self.msg_syn_to_var[i][j])
                if norm > 0:
                    self.msg_syn_to_var[i][j] /= norm
    
    def _update_logical_check_messages(self):
        """Update messages from logical checks"""
        for i in range(self.n_logical):
            vars_in_check = self.log_check_to_vars[i]
            
            # Message from logical check to logical syndrome variable
            # This computes P(s_L_i | error variables in check)
            
            # Compute parity distribution of all variables in the check
            parity_dist = np.array([1.0, 0.0])  # Start with even parity
            
            for var_idx in vars_in_check:
                var_prob = self.msg_var_to_log[var_idx][i]
                new_parity_dist = np.zeros(2)
                
                # Convolution for XOR operation
                new_parity_dist[0] = parity_dist[0] * var_prob[0] + parity_dist[1] * var_prob[1]
                new_parity_dist[1] = parity_dist[0] * var_prob[1] + parity_dist[1] * var_prob[0]
                
                parity_dist = new_parity_dist
            
            # This is the marginal distribution for s_L_i
            self.msg_log_to_sL[i][i] = parity_dist / np.sum(parity_dist)
            
            # Messages from logical check to variables
            for j in vars_in_check:
                # Compute message from logical check i to variable j
                other_vars = [v for v in vars_in_check if v != j]
                
                if len(other_vars) == 0:
                    # Only one variable in check - message depends on s_L_i belief
                    s_L_belief = self.msg_sL_to_log[i][i]
                    self.msg_log_to_var[i][j] = s_L_belief.copy()
                    continue
                
                # Compute expected parity constraint from s_L_i and other variables
                # This is more complex - we need to marginalize over s_L_i
                
                # Get current belief about s_L_i
                s_L_belief = self.msg_sL_to_log[i][i]
                
                # Compute parity distribution of other variables
                other_parity_dist = np.array([1.0, 0.0])
                
                for var_idx in other_vars:
                    var_prob = self.msg_var_to_log[var_idx][i]
                    new_parity_dist = np.zeros(2)
                    
                    new_parity_dist[0] = other_parity_dist[0] * var_prob[0] + other_parity_dist[1] * var_prob[1]
                    new_parity_dist[1] = other_parity_dist[0] * var_prob[1] + other_parity_dist[1] * var_prob[0]
                    
                    other_parity_dist = new_parity_dist
                
                # Message to variable j
                msg = np.zeros(2)
                
                # For each possible value of s_L_i
                for s_L_val in [0, 1]:
                    s_L_prob = s_L_belief[s_L_val]
                    
                    # For each possible value of variable j
                    for j_val in [0, 1]:
                        # Required parity of other variables
                        required_other_parity = s_L_val ^ j_val
                        
                        # Probability that other variables have required parity
                        prob_other = other_parity_dist[required_other_parity]
                        
                        msg[j_val] += s_L_prob * prob_other
                
                self.msg_log_to_var[i][j] = msg / np.sum(msg)
    
    def _update_variable_messages(self):
        """Update messages from variables to checks"""
        for j in range(self.n_vars):
            # Prior belief
            prior = np.array([1.0, np.exp(-self.weights[j])])
            
            # Update messages to syndrome checks
            for i in self.var_to_syn_checks[j]:
                belief = prior.copy()
                
                # Multiply by messages from other syndrome checks
                for other_i in self.var_to_syn_checks[j]:
                    if other_i != i:
                        belief *= self.msg_syn_to_var[other_i][j]
                
                # Multiply by messages from logical checks
                for log_i in self.var_to_log_checks[j]:
                    belief *= self.msg_log_to_var[log_i][j]
                
                self.msg_var_to_syn[j][i] = belief / np.sum(belief)
            
            # Update messages to logical checks
            for i in self.var_to_log_checks[j]:
                belief = prior.copy()
                
                # Multiply by messages from syndrome checks
                for syn_i in self.var_to_syn_checks[j]:
                    belief *= self.msg_syn_to_var[syn_i][j]
                
                # Multiply by messages from other logical checks
                for other_i in self.var_to_log_checks[j]:
                    if other_i != i:
                        belief *= self.msg_log_to_var[other_i][j]
                
                self.msg_var_to_log[j][i] = belief / np.sum(belief)
    
    def _update_logical_syndrome_messages(self):
        """Update messages from logical syndrome variables"""
        for i in range(self.n_logical):
            # Uniform prior for logical syndrome bits
            # (or could incorporate prior knowledge if available)
            self.msg_sL_to_log[i][i] = np.array([0.5, 0.5])
    
    def run_belief_propagation(self, max_iterations=50, tolerance=1e-6):
        """
        Run belief propagation to convergence
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            converged: Whether BP converged
            iterations: Number of iterations run
        """
        for iteration in range(max_iterations):
            # Store old messages for convergence check
            old_syn_messages = {}
            for i in range(self.n_syndromes):
                old_syn_messages[i] = {}
                for j in self.syn_check_to_vars[i]:
                    old_syn_messages[i][j] = self.msg_syn_to_var[i][j].copy()
            
            # BP update steps
            self._update_syndrome_check_messages()
            self._update_logical_check_messages()
            self._update_variable_messages()
            self._update_logical_syndrome_messages()
            
            # Check convergence
            converged = True
            for i in range(self.n_syndromes):
                for j in self.syn_check_to_vars[i]:
                    if np.linalg.norm(self.msg_syn_to_var[i][j] - old_syn_messages[i][j]) > tolerance:
                        converged = False
                        break
                if not converged:
                    break
            
            if converged:
                return True, iteration + 1
        
        return False, max_iterations
    
    def compute_logical_syndrome_marginals(self):
        """
        Compute marginal probabilities P(s_L_i | s') for each logical syndrome bit
        
        Returns:
            marginals: Array of shape (n_logical, 2) where marginals[i, j] = P(s_L_i = j | s')
        """
        marginals = np.zeros((self.n_logical, 2))
        
        for i in range(self.n_logical):
            # The marginal is computed from the logical check message
            marginals[i] = self.msg_log_to_sL[i][i]
        
        return marginals
    
    def find_most_likely_logical_syndrome(self):
        """
        Find the most likely logical syndrome based on marginals
        
        Returns:
            most_likely_s_L: Most probable logical syndrome (componentwise MAP)
            marginals: Marginal probabilities for each bit
        """
        marginals = self.compute_logical_syndrome_marginals()
        
        # Componentwise MAP estimation
        most_likely_s_L = np.argmax(marginals, axis=1)
        
        return most_likely_s_L, marginals

# Example usage
if __name__ == "__main__":
    # Example: Small QLDPC code
    np.random.seed(42)
    from qldpcdecoding.codes import gen_BB_code, gen_HP_ring_code, get_benchmark_code
    import numpy as np
    css_code = gen_BB_code(72) 
    D = np.hstack([np.eye(css_code.hz.shape[0]), css_code.hz])
    lz = np.hstack([np.zeros((css_code.lz.shape[0], css_code.hz.shape[0])), css_code.lz])
    print("col sparsity of lz:", np.bincount(np.sum(lz, axis=0).astype(int)))
    # error = np.random.choice(2, D.shape[1], replace=True, p=[0.001, 0.999])
    logical_errs = 0
    p_error = 0.003
    for trial in range(5000):
        error = np.zeros(D.shape[1])
        for i in range(D.shape[1]):
            if np.random.rand() < p_error:
                error[i] = 1
        if np.sum(error) == 0:
            continue
        syndrome = np.mod(D @ error, 2)
        true_logical_syndrome = np.mod(lz @ error, 2)
        # Example syndrome and weights
    
        weights = np.array([np.log((1-p_error)/p_error)] * D.shape[1])
        
        # Run BP
        bp = QLDPC_BP_Marginals(D, lz, syndrome, weights)
        
        # print("Running belief propagation...")
        converged, iterations = bp.run_belief_propagation(max_iterations=100)
        
        # if converged:
        #     print(f"Converged after {iterations} iterations")
        # else:
        #     print(f"Did not converge after {iterations} iterations")
        
        # Get marginal probabilities
        marginals = bp.compute_logical_syndrome_marginals()
        most_likely_s_L, _ = bp.find_most_likely_logical_syndrome()
        if not np.allclose(most_likely_s_L, true_logical_syndrome):
            logical_errs += 1
    print(f"Logical error rate: {logical_errs/1000}")