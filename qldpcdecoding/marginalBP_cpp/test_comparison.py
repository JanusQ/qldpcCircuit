#!/usr/bin/env python3
"""
Test script to compare C++ and Python implementations of the marginal BP decoder
"""

import numpy as np
import time
import sys
import os

# Add the parent directory to the path to import the original Python implementation
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from marginalBP.test import QLDPC_BP_Marginals as PythonQLDPC_BP_Marginals
    from marginal_bp_wrapper import QLDPC_BP_Marginals as CppQLDPC_BP_Marginals
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to build the C++ extension first using 'make' or 'python setup.py build_ext --inplace'")
    sys.exit(1)

def create_test_case():
    """Create a simple test case for comparison"""
    # Small QLDPC code example
    D_prime = np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1]
    ], dtype=np.int32)
    
    D_L = np.array([
        [1, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 1]
    ], dtype=np.int32)
    
    s_prime = np.array([1, 0, 1, 0], dtype=np.int32)
    weights = np.array([0.1, 0.2, 0.1, 0.3, 0.2, 0.1], dtype=np.float64)
    
    return D_prime, D_L, s_prime, weights

def test_correctness():
    """Test that both implementations give the same results"""
    print("Testing correctness...")
    
    D_prime, D_L, s_prime, weights = create_test_case()
    
    # Create both decoders
    python_bp = PythonQLDPC_BP_Marginals(D_prime, D_L, s_prime, weights)
    cpp_bp = CppQLDPC_BP_Marginals(D_prime, D_L, s_prime, weights)
    
    # Run BP with same parameters
    max_iterations = 20
    tolerance = 1e-6
    
    # Python implementation
    python_converged, python_iterations = python_bp.run_belief_propagation(max_iterations, tolerance)
    python_marginals = python_bp.compute_logical_syndrome_marginals()
    python_most_likely, _ = python_bp.find_most_likely_logical_syndrome()
    
    # C++ implementation
    cpp_converged, cpp_iterations = cpp_bp.run_belief_propagation(max_iterations, tolerance)
    cpp_marginals = cpp_bp.compute_logical_syndrome_marginals()
    cpp_most_likely, _ = cpp_bp.find_most_likely_logical_syndrome()
    
    # Compare results
    print(f"Convergence: Python={python_converged}, C++={cpp_converged}")
    print(f"Iterations: Python={python_iterations}, C++={cpp_iterations}")
    
    # Compare marginals
    marginals_diff = np.abs(python_marginals - cpp_marginals).max()
    print(f"Max marginals difference: {marginals_diff}")
    
    # Compare most likely logical syndromes
    logical_diff = np.abs(python_most_likely - cpp_most_likely).max()
    print(f"Logical syndrome difference: {logical_diff}")
    
    # Check if results are close enough
    tolerance = 1e-4  # Increased tolerance for numerical differences
    if marginals_diff < tolerance and logical_diff == 0:
        print("✓ Correctness test PASSED")
        return True
    else:
        print("✗ Correctness test FAILED")
        return False

def test_performance():
    """Test performance comparison"""
    print("\nTesting performance...")
    
    D_prime, D_L, s_prime, weights = create_test_case()
    
    # Python implementation timing
    python_bp = PythonQLDPC_BP_Marginals(D_prime, D_L, s_prime, weights)
    
    start_time = time.time()
    python_bp.run_belief_propagation(max_iterations=50)
    python_time = time.time() - start_time
    
    # C++ implementation timing
    cpp_bp = CppQLDPC_BP_Marginals(D_prime, D_L, s_prime, weights)
    
    start_time = time.time()
    cpp_bp.run_belief_propagation(max_iterations=50)
    cpp_time = time.time() - start_time
    
    print(f"Python time: {python_time:.4f} seconds")
    print(f"C++ time: {cpp_time:.4f} seconds")
    print(f"Speedup: {python_time / cpp_time:.2f}x")
    
    if cpp_time < python_time:
        print("✓ Performance test PASSED")
        return True
    else:
        print("✗ Performance test FAILED")
        return False

def test_larger_case():
    """Test with a larger case"""
    print("\nTesting larger case...")
    
    # Generate a larger random test case
    np.random.seed(42)
    n_vars = 100
    n_syndromes = 50
    n_logical = 10
    
    # Random sparse matrices
    D_prime = np.random.choice([0, 1], size=(n_syndromes, n_vars), p=[0.9, 0.1])
    D_L = np.random.choice([0, 1], size=(n_logical, n_vars), p=[0.8, 0.2])
    s_prime = np.random.choice([0, 1], size=n_syndromes)
    weights = np.random.uniform(0.1, 1.0, size=n_vars)
    
    try:
        # Test C++ implementation with larger case
        cpp_bp = CppQLDPC_BP_Marginals(D_prime, D_L, s_prime, weights)
        
        start_time = time.time()
        converged, iterations = cpp_bp.run_belief_propagation(max_iterations=30)
        cpp_time = time.time() - start_time
        
        marginals = cpp_bp.compute_logical_syndrome_marginals()
        most_likely, _ = cpp_bp.find_most_likely_logical_syndrome()
        
        print(f"Larger case - C++ time: {cpp_time:.4f} seconds")
        print(f"Converged: {converged}, Iterations: {iterations}")
        print(f"Marginals shape: {marginals.shape}")
        print(f"Most likely logical syndrome: {most_likely}")
        print("✓ Larger case test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Larger case test FAILED: {e}")
        return False

def main():
    """Run all tests"""
    print("QLDPC Marginal BP Decoder - C++ vs Python Comparison")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_correctness():
        tests_passed += 1
    
    if test_performance():
        tests_passed += 1
    
    if test_larger_case():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests PASSED! The C++ implementation is working correctly.")
    else:
        print("❌ Some tests FAILED. Please check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    main() 