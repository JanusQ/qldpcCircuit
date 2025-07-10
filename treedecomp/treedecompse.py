import numpy as np
import networkx as nx
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional
import itertools
from dataclasses import dataclass
import time
from qldpcdecoding.codes import gen_BB_code, gen_HP_ring_code, get_benchmark_code
@dataclass
class TreeDecomposition:
    """Tree decomposition data structure"""
    bags: Dict[int, Set[int]]  # node_id -> set of syndrome bits
    tree: nx.Graph  # tree structure
    root: int
    
    def children(self, node):
        return list(self.tree.neighbors(node))
    
    def is_leaf(self, node):
        return self.tree.degree(node) <= 1
    
    def treewidth(self):
        return max(len(bag) for bag in self.bags.values()) - 1

class TreewidthSolver:
    """
    Treewidth-based solver for quantum error correction optimization
    
    Solves: argmin_{l ∈ {0,1}^K} ∑_{u ∈ {0,1}^m} P(e_0 ⊕ D_x*u ⊕ L_x*l)
    """
    
    def __init__(self, D_x: np.ndarray, L_x: np.ndarray, e_0: np.ndarray, 
                 weights: Optional[np.ndarray] = None, probability_type: str = 'exponential'):
        """
        Initialize solver
        
        Args:
            D_x: Check matrix (n x m)
            L_x: Logical operators (n x K) 
            e_0: Initial error (n,)
            weights: Error weights (n,) - w_j = ln((1-p_j)/p_j) for exponential
            probability_type: 'exponential' or 'product'
        """
        self.D_x = D_x.astype(int)
        self.L_x = L_x.astype(int)
        self.e_0 = e_0.astype(int)
        self.weights = weights
        self.probability_type = probability_type
        
        self.n, self.m = D_x.shape
        self.K = L_x.shape[1]
        
        # Build syndrome interaction graph
        self.syndrome_graph = self._build_syndrome_graph()
        
        # Compute tree decomposition
        self.tree_decomp = self._compute_tree_decomposition()
        
        print(f"Problem size: n={self.n}, m={self.m}, K={self.K}")
        print(f"Syndrome graph: {self.syndrome_graph.number_of_nodes()} nodes, {self.syndrome_graph.number_of_edges()} edges")
        print(f"Treewidth: {self.tree_decomp.treewidth()}")
    
    def _build_syndrome_graph(self) -> nx.Graph:
        """Build syndrome interaction graph"""
        G = nx.Graph()
        
        # Add all syndrome bits as nodes
        for i in range(self.m):
            G.add_node(i)
        
        # Add edges between syndromes that share physical qubits
        for i in range(self.m):
            for j in range(i+1, self.m):
                col_i = self.D_x[:, i]
                col_j = self.D_x[:, j]
                # Check if columns share any physical qubit
                if np.any(col_i & col_j):
                    G.add_edge(i, j)
        
        return G
    
    def _compute_tree_decomposition(self) -> TreeDecomposition:
        """Compute tree decomposition using minimum degree heuristic"""
        if self.syndrome_graph.number_of_nodes() == 0:
            # Empty graph case
            return TreeDecomposition(bags={0: set()}, tree=nx.Graph([(0, 0)]), root=0)
        
        # Use minimum degree elimination
        H = self.syndrome_graph.copy()
        elimination_order = []
        bags = {}
        
        node_id = 0
        while H.nodes:
            # Find vertex with minimum degree
            if len(H.nodes) == 1:
                min_vertex = list(H.nodes)[0]
            else:
                min_vertex = min(H.nodes, key=lambda v: H.degree(v))
            
            elimination_order.append(min_vertex)
            
            # Create bag containing min_vertex and its neighbors
            neighbors = set(H.neighbors(min_vertex))
            bag = {min_vertex} | neighbors
            bags[node_id] = bag
            
            # Make neighbors into a clique
            for u in neighbors:
                for v in neighbors:
                    if u != v and not H.has_edge(u, v):
                        H.add_edge(u, v)
            
            # Remove min_vertex
            H.remove_node(min_vertex)
            node_id += 1
        
        # Build tree structure
        tree = nx.Graph()
        for i in range(len(bags)):
            tree.add_node(i)
        
        # Connect bags that share vertices (simplified)
        for i in range(len(bags) - 1):
            tree.add_edge(i, i + 1)
        
        root = 0 if bags else 0
        print(f"bags number: {len(bags)}")
        print(f"Treewidth: {np.max([len(bag) for bag in bags.values()])-1}")
        return TreeDecomposition(bags=bags, tree=tree, root=root)
    
    def _compute_probability(self, error_pattern: np.ndarray) -> float:
        """Compute probability P(error_pattern)"""
        if self.weights is None:
            return 1.0
        
        if self.probability_type == 'exponential':
            # P(e) = exp(-∑ w_j e_j)
            log_prob = -np.sum(self.weights * error_pattern)
            return np.exp(log_prob)
        elif self.probability_type == 'product':
            # P(e) = ∏ (1 + (z_j - 1) * e_j) where z_j = p_j/(1-p_j)
            prob = 1.0
            for j in range(len(error_pattern)):
                if error_pattern[j] == 1:
                    prob *= self.weights[j]  # Assume weights[j] = z_j
            return prob
        else:
            raise ValueError(f"Unknown probability type: {self.probability_type}")
    
    def _enumerate_bag_assignments(self, bag: Set[int]):
        """Enumerate all 2^|bag| assignments to syndrome bits in bag"""
        bag_list = list(sorted(bag))
        for assignment_int in range(2**len(bag_list)):
            assignment = {}
            for i, syndrome_bit in enumerate(bag_list):
                assignment[syndrome_bit] = (assignment_int >> i) & 1
            yield assignment
    
    def _is_compatible(self, parent_assignment: Dict[int, int], 
                      child_assignment: Dict[int, int], 
                      separator: Set[int]) -> bool:
        """Check if assignments agree on separator"""
        for syndrome_bit in separator:
            if (syndrome_bit in parent_assignment and 
                syndrome_bit in child_assignment and
                parent_assignment[syndrome_bit] != child_assignment[syndrome_bit]):
                return False
        return True
    
    def _compute_subtree_contribution(self, tree_node: int, 
                                    bag_assignment: Dict[int, int],
                                    current_error: np.ndarray,
                                    memo: Dict) -> float:
        """Compute contribution from subtree rooted at tree_node"""
        
        # Create memoization key
        assignment_tuple = tuple(sorted(bag_assignment.items()))
        memo_key = (tree_node, assignment_tuple)
        
        if memo_key in memo:
            return memo[memo_key]
        
        bag = self.tree_decomp.bags[tree_node]
        
        # Compute error pattern for this bag assignment
        syndrome_pattern = np.zeros(self.m, dtype=int)
        for syndrome_bit, value in bag_assignment.items():
            syndrome_pattern[syndrome_bit] = value
        
        # Error contribution from this syndrome assignment
        error_from_syndrome = (self.D_x @ syndrome_pattern) % 2
        total_error = (current_error ^ error_from_syndrome) % 2
        
        if self.tree_decomp.is_leaf(tree_node):
            # Base case: leaf node
            prob = self._compute_probability(total_error)
            memo[memo_key] = prob
            return prob
        
        # Recursive case: sum over compatible child assignments
        result = 0.0
        children = [child for child in self.tree_decomp.tree.neighbors(tree_node)]
        
        if not children:
            # No children (shouldn't happen in proper tree)
            prob = self._compute_probability(total_error)
            memo[memo_key] = prob
            return prob
        
        # For simplicity, handle single child case (can be extended)
        child = children[0]
        child_bag = self.tree_decomp.bags[child]
        separator = bag & child_bag
        
        for child_assignment in self._enumerate_bag_assignments(child_bag):
            if self._is_compatible(bag_assignment, child_assignment, separator):
                child_contrib = self._compute_subtree_contribution(
                    child, child_assignment, total_error, memo
                )
                result += child_contrib
        
        memo[memo_key] = result
        return result
    
    def compute_objective_for_logical(self, l: np.ndarray) -> float:
        """Compute objective for a specific logical operator l"""
        # Compute base error with logical correction
        base_error = (self.e_0 ^ (self.L_x @ l)) % 2
        
        # Use tree decomposition to compute syndrome sum
        root = self.tree_decomp.root
        root_bag = self.tree_decomp.bags[root]
        
        total_sum = 0.0
        memo = {}
        
        # Handle empty bag case
        if not root_bag:
            # No syndromes, just return probability of base error
            return self._compute_probability(base_error)
        
        # Enumerate all assignments to root bag
        for root_assignment in self._enumerate_bag_assignments(root_bag):
            contribution = self._compute_subtree_contribution(
                root, root_assignment, base_error, memo
            )
            total_sum += contribution
        
        return total_sum
    
    def solve_optimal(self) -> Tuple[np.ndarray, float]:
        """
        Find optimal logical operator
        
        Returns:
            (optimal_l, optimal_objective)
        """
        print(f"Solving optimization with treewidth {self.tree_decomp.treewidth()}")
        
        if self.tree_decomp.treewidth() > 15:
            print("Warning: Large treewidth detected. This may be slow.")
        
        best_l = None
        best_objective = float('inf')
        
        start_time = time.time()
        
        # Enumerate all logical operators
        for l_int in range(2**self.K):
            # Convert integer to bit vector
            l = np.array([(l_int >> i) & 1 for i in range(self.K)], dtype=int)
            
            # Compute objective
            objective = self.compute_objective_for_logical(l)
            
            if objective < best_objective:
                best_objective = objective
                best_l = l.copy()
            
            # Progress reporting
            if l_int % max(1, 2**(self.K-4)) == 0:
                elapsed = time.time() - start_time
                progress = (l_int + 1) / (2**self.K) * 100
                print(f"Progress: {progress:.1f}%, Current best: {best_objective:.6f}, Time: {elapsed:.2f}s")
        
        total_time = time.time() - start_time
        print(f"Optimization completed in {total_time:.2f}s")
        print(f"Optimal objective: {best_objective:.6f}")
        print(f"Optimal logical operator: {best_l}")
        
        return best_l, best_objective
    
    def solve_greedy(self) -> Tuple[np.ndarray, float]:
        """
        Greedy local search for large K
        """
        print("Using greedy local search for large K")
        
        # Start with zero logical operator
        current_l = np.zeros(self.K, dtype=int)
        current_obj = self.compute_objective_for_logical(current_l)
        
        improved = True
        iteration = 0
        
        while improved:
            improved = False
            iteration += 1
            print(f"Iteration {iteration}, Current objective: {current_obj:.6f}")
            
            for i in range(self.K):
                # Try flipping bit i
                test_l = current_l.copy()
                test_l[i] = 1 - test_l[i]
                
                test_obj = self.compute_objective_for_logical(test_l)
                
                if test_obj < current_obj:
                    current_l = test_l
                    current_obj = test_obj
                    improved = True
                    print(f"  Improved by flipping bit {i}: {current_obj:.6f}")
                    break
        
        print(f"Greedy search completed. Final objective: {current_obj:.6f}")
        return current_l, current_obj

# Example usage and testing
def create_example_problem():
    """Create a small example problem for testing"""


    css_code = gen_BB_code(72) 
    D = np.hstack([np.eye(css_code.hx.shape[0]), css_code.hx])
    lz = np.hstack([np.zeros((css_code.lz.shape[0], css_code.hx.shape[0])), css_code.lz])
    print("col sparsity of lz:", np.bincount(np.sum(lz, axis=0).astype(int)))
    # Small problem: 8 physical qubits, 4 syndromes, 2 logical operators
    m, n, K = D.shape[0], D.shape[1], lz.shape[0]
    # Create sparse D_x (each column has at most 3 non-zeros)
    # Random initial error
    e_0 = np.random.randint(0, 2, n)
    
    # Random weights
    weights = np.random.uniform(0.1, 2.0, n)
    
    return D, lz, e_0, weights

def test_solver():
    """Test the solver on a small example"""
    print("Creating example problem...")
    D_x, L_x, e_0, weights = create_example_problem()
    
    print(f"D_x shape: {D_x.shape}")
    print(f"L_x shape: {L_x.shape}")
    print(f"e_0 shape: {e_0.shape}")
    print(f"D_x sparsity: {np.sum(D_x)} / {D_x.size}")
    print(f"L_x sparsity: {np.sum(L_x)} / {L_x.size}")
    
    # Create solver
    solver = TreewidthSolver(D_x, L_x, e_0, weights, probability_type='exponential')
    
    # Solve
    if solver.K <= 10:
        optimal_l, optimal_obj = solver.solve_optimal()
    else:
        optimal_l, optimal_obj = solver.solve_greedy()
    
    return optimal_l, optimal_obj

if __name__ == "__main__":
    test_solver()