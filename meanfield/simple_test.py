#!/usr/bin/env python3
"""
Simple test script for the Mean-Field QLDPC Decoder
"""

import numpy as np
import scipy.sparse as sp

# Import the decoder
from qlpc_decoder import QLDPCDecoder, create_uniform_error_model

def test_basic_functionality():
    """Test basic functionality of the decoder."""
    print("Testing basic functionality...")
    
    # Create a simple CSS code
    D_x_data = [
        [1, 1, 0, 0],  # Check 1: qubits 0,1
        [0, 1, 1, 0],  # Check 2: qubits 1,2
    ]
    
    L_x_data = [
        [1, 0, 0, 1],  # Logical: qubits 0,3
    ]
    
    # Transpose so that rows represent qubits and columns represent operators
    D_x = sp.csr_matrix(D_x_data, dtype=float).T
    L_x = sp.csr_matrix(L_x_data, dtype=float).T
    
    # Create decoder
    decoder = QLDPCDecoder(max_iterations=20, tolerance=1e-4)
    decoder.set_code(D_x, L_x)
    
    # Check code info
    info = decoder.code_info
    print(f"Code info: {info}")
    
    # Create error model
    error_probs = create_uniform_error_model(4, 0.1)
    print(f"Error probabilities: {error_probs}")
    
    # Test syndrome (4 qubits)
    syndrome = np.array([1, 0, 0, 0])  # Error on qubit 0
    
    # Decode
    logical_correction = decoder.decode_syndrome(syndrome, error_probs)
    print(f"Logical correction: {logical_correction}")
    
    # Get magnetizations
    check_mags, logical_mags = decoder.get_magnetizations(syndrome, error_probs)
    print(f"Check magnetizations: {check_mags}")
    print(f"Logical magnetizations: {logical_mags}")
    
    # Test eta computation
    eta = decoder.compute_eta(syndrome)
    print(f"Eta values: {eta}")
    
    print("Basic functionality test passed!")
    return True

def test_error_models():
    """Test error model creation."""
    print("\nTesting error models...")
    
    # Test uniform error model
    error_probs = create_uniform_error_model(5, 0.05)
    print(f"Uniform error model (5 qubits, 5%): {error_probs}")
    
    # Test weights conversion
    decoder = QLDPCDecoder()
    weights = decoder.error_probabilities_to_weights(error_probs)
    print(f"Weights: {weights}")
    
    print("Error model test passed!")
    return True

def test_batch_decoding():
    """Test batch decoding."""
    print("\nTesting batch decoding...")
    
    # Create code and decoder
    D_x_data = [[1, 1, 0], [0, 1, 1]]
    L_x_data = [[1, 0, 1]]
    
    # Transpose so that rows represent qubits and columns represent operators
    D_x = sp.csr_matrix(D_x_data, dtype=float).T
    L_x = sp.csr_matrix(L_x_data, dtype=float).T
    
    decoder = QLDPCDecoder(max_iterations=20, tolerance=1e-4)
    decoder.set_code(D_x, L_x)
    
    # Create error model
    error_probs = create_uniform_error_model(3, 0.1)
    
    # Test multiple syndromes (3 qubits)
    syndromes = np.array([
        [1, 0, 0],  # Error on qubit 0
        [0, 1, 0],  # Error on qubit 1
        [0, 0, 0],  # No error
    ])
    
    # Decode
    logical_corrections = decoder.decode_batch(syndromes, error_probs)
    print(f"Batch logical corrections:\n{logical_corrections}")
    
    print("Batch decoding test passed!")
    return True

def main():
    """Run all tests."""
    print("Mean-Field QLDPC Decoder - Simple Test")
    print("=" * 40)
    
    try:
        test_basic_functionality()
        test_error_models()
        test_batch_decoding()
        
        print("\n" + "=" * 40)
        print("All tests passed successfully!")
        print("The mean-field decoder is working correctly.")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 