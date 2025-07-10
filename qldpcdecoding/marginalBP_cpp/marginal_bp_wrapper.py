import numpy as np
from .MarginalBPDecoder import QLDPC_BP_Marginals as CppQLDPC_BP_Marginals

class QLDPC_BP_Marginals:
    """
    Python wrapper for the C++ QLDPC Belief Propagation Marginal Decoder
    
    This class provides the same interface as the original Python implementation
    but uses the optimized C++ backend for better performance.
    """
    
    def __init__(self, D_prime, D_L, s_prime, weights):
        """
        Initialize BP for QLDPC logical syndrome marginal estimation
        
        Args:
            D_prime: Syndrome constraint matrix (m x n)
            D_L: Logical syndrome constraint matrix (k x n) 
            s_prime: Observed syndrome vector (length m)
            weights: Log-likelihood weights w_j for each bit (length n)
        """
        # Convert inputs to numpy arrays if they aren't already
        D_prime = np.asarray(D_prime, dtype=np.int32)
        D_L = np.asarray(D_L, dtype=np.int32)
        s_prime = np.asarray(s_prime, dtype=np.int32)
        weights = np.asarray(weights, dtype=np.float64)
        
        # Create the C++ decoder instance
        self._cpp_decoder = CppQLDPC_BP_Marginals(D_prime, D_L, s_prime, weights)
        
        # Store dimensions for convenience
        self.n_vars = D_prime.shape[1]
        self.n_syndromes = D_prime.shape[0]
        self.n_logical = D_L.shape[0]
    
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
        return self._cpp_decoder.run_belief_propagation(max_iterations, tolerance)
    
    def compute_logical_syndrome_marginals(self):
        """
        Compute marginal probabilities P(s_L_i | s') for each logical syndrome bit
        
        Returns:
            marginals: Array of shape (n_logical, 2) where marginals[i, j] = P(s_L_i = j | s')
        """
        return self._cpp_decoder.compute_logical_syndrome_marginals()
    
    def find_most_likely_logical_syndrome(self):
        """
        Find the most likely logical syndrome based on marginals
        
        Returns:
            most_likely_s_L: Most probable logical syndrome (componentwise MAP)
            marginals: Marginal probabilities for each bit
        """
        return self._cpp_decoder.find_most_likely_logical_syndrome()

# Example usage and testing
if __name__ == "__main__":
    # Test with a simple example
    print("Testing C++ Marginal BP Decoder...")
    
    # Simple test case
    D_prime = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32)
    D_L = np.array([[1, 0, 1]], dtype=np.int32)
    s_prime = np.array([1, 0], dtype=np.int32)
    weights = np.array([0.1, 0.2, 0.1], dtype=np.float64)
    
    # Create decoder
    bp = QLDPC_BP_Marginals(D_prime, D_L, s_prime, weights)
    
    # Run BP
    converged, iterations = bp.run_belief_propagation(max_iterations=10)
    print(f"BP converged: {converged}, iterations: {iterations}")
    
    # Get marginals
    marginals = bp.compute_logical_syndrome_marginals()
    print(f"Marginals shape: {marginals.shape}")
    print(f"Marginals: {marginals}")
    
    # Get most likely logical syndrome
    most_likely_s_L, marginals = bp.find_most_likely_logical_syndrome()
    print(f"Most likely logical syndrome: {most_likely_s_L}")
    
    print("Test completed successfully!") 