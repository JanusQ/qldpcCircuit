#!/usr/bin/env python3
"""
Simple tests for the Mean-Field QLDPC Decoder

This module provides basic tests to verify the decoder functionality.
"""

import numpy as np
import scipy.sparse as sp
import pytest

# Try to import the decoder
try:
    from qlpc_decoder import QLDPCDecoder, create_uniform_error_model
    DECODER_AVAILABLE = True
except ImportError:
    DECODER_AVAILABLE = False


def create_test_code():
    """Create a simple test CSS code."""
    # Small test code: 4 qubits, 2 checks, 1 logical operator
    D_x_data = [
        [1, 1, 0, 0],  # Check 1: qubits 0,1
        [0, 1, 1, 0],  # Check 2: qubits 1,2
    ]
    
    L_x_data = [
        [1, 0, 0, 1],  # Logical: qubits 0,3
    ]
    
    D_x = sp.csr_matrix(D_x_data, dtype=float)
    L_x = sp.csr_matrix(L_x_data, dtype=float)
    
    return D_x, L_x


@pytest.mark.skipif(not DECODER_AVAILABLE, reason="Decoder not available")
class TestQLDPCDecoder:
    """Test cases for the QLDPC decoder."""
    
    def test_initialization(self):
        """Test decoder initialization."""
        decoder = QLDPCDecoder(max_iterations=50, tolerance=1e-6)
        assert decoder.max_iterations == 50
        assert decoder.tolerance == 1e-6
        
        # Check initial state
        info = decoder.code_info
        assert info["status"] == "No code set"
    
    def test_code_setup(self):
        """Test setting up a code."""
        decoder = QLDPCDecoder()
        D_x, L_x = create_test_code()
        
        decoder.set_code(D_x, L_x)
        
        info = decoder.code_info
        assert info["num_qubits"] == 4
        assert info["num_checks"] == 2
        assert info["num_logical"] == 1
        assert info["check_matrix_shape"] == (4, 2)
        assert info["logical_matrix_shape"] == (4, 1)
    
    def test_error_probabilities_to_weights(self):
        """Test conversion of error probabilities to weights."""
        decoder = QLDPCDecoder()
        
        # Test uniform error model
        error_probs = np.array([0.1, 0.1, 0.1, 0.1])
        weights = decoder.error_probabilities_to_weights(error_probs)
        
        expected_weight = np.log((1 - 0.1) / 0.1)
        np.testing.assert_allclose(weights, expected_weight)
        
        # Test edge cases
        error_probs = np.array([0.5, 0.5, 0.5, 0.5])
        weights = decoder.error_probabilities_to_weights(error_probs)
        np.testing.assert_allclose(weights, 0.0)
        
        # Test invalid probabilities
        with pytest.raises(ValueError):
            decoder.error_probabilities_to_weights(np.array([-0.1, 0.5, 0.5, 0.5]))
        
        with pytest.raises(ValueError):
            decoder.error_probabilities_to_weights(np.array([1.1, 0.5, 0.5, 0.5]))
    
    def test_single_decoding(self):
        """Test single syndrome decoding."""
        decoder = QLDPCDecoder(max_iterations=20, tolerance=1e-4)
        D_x, L_x = create_test_code()
        decoder.set_code(D_x, L_x)
        
        # Create error model
        error_probs = create_uniform_error_model(4, 0.1)
        
        # Test syndrome
        syndrome = np.array([1, 0])  # Error on qubit 0 or 1
        
        # Decode
        logical_correction = decoder.decode_syndrome(syndrome, error_probs)
        
        # Check output format
        assert logical_correction.shape == (1,)
        assert logical_correction.dtype == int
        assert np.all((logical_correction == 0) | (logical_correction == 1))
    
    def test_batch_decoding(self):
        """Test batch decoding."""
        decoder = QLDPCDecoder(max_iterations=20, tolerance=1e-4)
        D_x, L_x = create_test_code()
        decoder.set_code(D_x, L_x)
        
        # Create error model
        error_probs = create_uniform_error_model(4, 0.1)
        
        # Test multiple syndromes
        syndromes = np.array([
            [1, 0],  # Error on qubit 0 or 1
            [0, 1],  # Error on qubit 1 or 2
            [0, 0],  # No error
        ])
        
        # Decode
        logical_corrections = decoder.decode_batch(syndromes, error_probs)
        
        # Check output format
        assert logical_corrections.shape == (3, 1)
        assert logical_corrections.dtype == int
        assert np.all((logical_corrections == 0) | (logical_corrections == 1))
    
    def test_magnetizations(self):
        """Test getting magnetizations."""
        decoder = QLDPCDecoder(max_iterations=20, tolerance=1e-4)
        D_x, L_x = create_test_code()
        decoder.set_code(D_x, L_x)
        
        # Create error model
        error_probs = create_uniform_error_model(4, 0.1)
        syndrome = np.array([1, 0])
        
        # Get magnetizations
        check_mags, logical_mags = decoder.get_magnetizations(syndrome, error_probs)
        
        # Check output format
        assert check_mags.shape == (2,)  # 2 checks
        assert logical_mags.shape == (1,)  # 1 logical operator
        assert check_mags.dtype == float
        assert logical_mags.dtype == float
        
        # Check magnetization bounds
        assert np.all(np.abs(check_mags) <= 1.0)
        assert np.all(np.abs(logical_mags) <= 1.0)
    
    def test_eta_computation(self):
        """Test eta computation."""
        decoder = QLDPCDecoder()
        D_x, L_x = create_test_code()
        decoder.set_code(D_x, L_x)
        
        # Test syndrome
        syndrome = np.array([1, 0, 1, 0])
        eta = decoder.compute_eta(syndrome)
        
        # Check eta values: eta_j = 1 - 2*e_{0j}
        expected_eta = np.array([-1, 1, -1, 1])  # 1 - 2*[1,0,1,0]
        np.testing.assert_array_equal(eta, expected_eta)
    
    def test_parameter_updates(self):
        """Test updating decoder parameters."""
        decoder = QLDPCDecoder(max_iterations=50, tolerance=1e-6)
        
        # Update parameters
        decoder.set_parameters(max_iterations=100, tolerance=1e-8)
        
        assert decoder.max_iterations == 100
        assert decoder.tolerance == 1e-8
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        decoder = QLDPCDecoder()
        
        # Test decoding without setting code
        with pytest.raises(ValueError, match="Code parameters not set"):
            decoder.decode_syndrome(np.array([1, 0]), np.array([0.1, 0.1]))
        
        # Set up code
        D_x, L_x = create_test_code()
        decoder.set_code(D_x, L_x)
        
        # Test wrong syndrome dimensions
        with pytest.raises(ValueError, match="Syndrome must have"):
            decoder.decode_syndrome(np.array([1, 0, 0]), np.array([0.1, 0.1, 0.1, 0.1]))
        
        # Test wrong weights dimensions
        with pytest.raises(ValueError, match="Weights must have"):
            decoder.decode_syndrome(np.array([1, 0]), np.array([0.1, 0.1, 0.1]))
        
        # Test missing error information
        with pytest.raises(ValueError, match="Either error_probs or weights must be provided"):
            decoder.decode_syndrome(np.array([1, 0]))


def test_uniform_error_model():
    """Test uniform error model creation."""
    if not DECODER_AVAILABLE:
        pytest.skip("Decoder not available")
    
    from qlpc_decoder import create_uniform_error_model
    
    error_probs = create_uniform_error_model(5, 0.1)
    
    assert error_probs.shape == (5,)
    assert np.all(error_probs == 0.1)


def test_depolarizing_error_model():
    """Test depolarizing error model creation."""
    if not DECODER_AVAILABLE:
        pytest.skip("Decoder not available")
    
    from qlpc_decoder import create_depolarizing_error_model
    
    error_probs = create_depolarizing_error_model(4, 0.05, 0.03, 0.02)
    
    assert error_probs.shape == (4,)
    expected_prob = 0.05 + 0.03 + 0.02
    assert np.all(error_probs == expected_prob)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"]) 