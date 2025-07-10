#!/usr/bin/env python3
"""
High-level Python interface for the Mean-Field QLDPC Decoder

This module provides an easy-to-use interface for decoding CSS quantum LDPC codes
using mean-field theory. It handles the conversion of error probabilities to
weights and provides convenient functions for common decoding tasks.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Union
import warnings

try:
    from . import meanfield_decoder
except ImportError:
    # Try direct import if not installed as package
    import meanfield_decoder


class QLDPCDecoder:
    """
    High-level interface for QLDPC decoding using mean-field theory.
    
    This class provides a convenient interface for decoding CSS quantum LDPC codes
    using the mean-field approximation. It handles the conversion of error
    probabilities to weights and provides methods for both single-shot and
    batch decoding.
    
    Attributes:
        max_iterations (int): Maximum number of iterations for convergence
        tolerance (float): Convergence tolerance for magnetization updates
        decoder: The underlying C++ mean-field decoder
    """
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Initialize the QLDPC decoder.
        
        Args:
            max_iterations: Maximum number of iterations for convergence
            tolerance: Convergence tolerance for magnetization updates
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.decoder = meanfield_decoder.MeanFieldDecoder(max_iterations, tolerance)
        
        # Store matrices for validation
        self._D_x = None
        self._L_x = None
        self._num_qubits = None
        self._num_checks = None
        self._num_logical = None
        
    def set_code(self, D_x: sp.spmatrix, L_x: sp.spmatrix) -> None:
        """
        Set the CSS code parameters.
        
        Args:
            D_x: Check matrix (syndrome to qubit mapping) in sparse format
            L_x: Logical operator matrix (logical operators to qubit mapping) in sparse format
            
        Raises:
            ValueError: If matrices have incompatible dimensions
        """
        # Convert to CSC format for efficiency
        D_x_csc = D_x.tocsc()
        L_x_csc = L_x.tocsc()
        
        # Store dimensions
        self._num_qubits = D_x_csc.shape[0]
        self._num_checks = D_x_csc.shape[1]
        self._num_logical = L_x_csc.shape[1]
        
        # Set matrices in decoder
        self.decoder.set_check_matrix(D_x_csc)
        self.decoder.set_logical_matrix(L_x_csc)
        
        # Store for reference
        self._D_x = D_x_csc
        self._L_x = L_x_csc
        
    def error_probabilities_to_weights(self, error_probs: np.ndarray) -> np.ndarray:
        """
        Convert error probabilities to weights for the decoder.
        
        The weight for qubit j is computed as w_j = ln((1-p_j)/p_j),
        where p_j is the error probability for qubit j.
        
        Args:
            error_probs: Array of error probabilities for each qubit
            
        Returns:
            Array of weights for each qubit
            
        Raises:
            ValueError: If error probabilities are not in [0, 1]
        """
        error_probs = np.asarray(error_probs)
        
        if np.any((error_probs < 0) | (error_probs > 1)):
            raise ValueError("Error probabilities must be in [0, 1]")
        
        # Avoid division by zero and log(0)
        eps = 1e-15
        error_probs = np.clip(error_probs, eps, 1 - eps)
        
        weights = np.log((1 - error_probs) / error_probs)
        return weights
    
    def decode_syndrome(self, 
                       syndrome: np.ndarray, 
                       error_probs: Optional[np.ndarray] = None,
                       weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decode a syndrome to find the optimal logical correction.
        
        Args:
            syndrome: Binary syndrome vector (0s and 1s)
            error_probs: Error probabilities for each qubit (optional if weights provided)
            weights: Pre-computed weights for each qubit (optional if error_probs provided)
            
        Returns:
            Binary logical correction vector (0s and 1s)
            
        Raises:
            ValueError: If neither error_probs nor weights provided, or if dimensions don't match
        """
        if self._D_x is None or self._L_x is None:
            raise ValueError("Code parameters not set. Call set_code() first.")
        print(self._D_x.shape)
        print(self._L_x.shape)
        
        
        # Get weights
        if weights is not None:
            weights = np.asarray(weights)
        elif error_probs is not None:
            weights = self.error_probabilities_to_weights(error_probs)
        else:
            raise ValueError("Either error_probs or weights must be provided")
        
        # Set parameters in decoder
        self.decoder.set_error_weights(weights)
        self.decoder.set_initial_syndrome(syndrome)
        
        # Perform decoding
        logical_correction = self.decoder.decode()
        
        return np.array(logical_correction, dtype=int)
    
    def decode_batch(self, 
                    syndromes: np.ndarray, 
                    error_probs: Optional[np.ndarray] = None,
                    weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decode multiple syndromes in batch.
        
        Args:
            syndromes: Array of syndromes, shape (num_syndromes, num_qubits)
            error_probs: Error probabilities for each qubit (optional if weights provided)
            weights: Pre-computed weights for each qubit (optional if error_probs provided)
            
        Returns:
            Array of logical corrections, shape (num_syndromes, num_logical)
        """
        syndromes = np.asarray(syndromes)
        
        if syndromes.ndim == 1:
            # Single syndrome case
            return self.decode_syndrome(syndromes, error_probs, weights)
        
        if syndromes.ndim != 1:
            raise ValueError("Syndromes must be 1D (single syndrome) or 2D (batch)")
        
        num_syndromes = len(syndromes)
        logical_corrections = np.zeros((num_syndromes, self._num_logical), dtype=int)
        
        for i in range(num_syndromes):
            logical_corrections[i] = self.decode_syndrome(
                syndromes[i], error_probs, weights
            )
        
        return logical_corrections
    
    def get_magnetizations(self, 
                          syndrome: np.ndarray, 
                          error_probs: Optional[np.ndarray] = None,
                          weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the final magnetizations for analysis.
        
        Args:
            syndrome: Binary syndrome vector
            error_probs: Error probabilities for each qubit
            weights: Pre-computed weights for each qubit
            
        Returns:
            Tuple of (check_magnetizations, logical_magnetizations)
        """
        if self._D_x is None or self._L_x is None:
            raise ValueError("Code parameters not set. Call set_code() first.")
        
        # Set up decoder (reuse decode_syndrome logic)
        syndrome = np.asarray(syndrome, dtype=int)
        
        if weights is not None:
            weights = np.asarray(weights)
        elif error_probs is not None:
            weights = self.error_probabilities_to_weights(error_probs)
        else:
            raise ValueError("Either error_probs or weights must be provided")
        
        self.decoder.set_error_weights(weights)
        self.decoder.set_initial_syndrome(syndrome)
        
        # Get magnetizations
        check_mags, logical_mags = self.decoder.get_magnetizations()
        
        return np.array(check_mags), np.array(logical_mags)
    
    def compute_eta(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Compute eta_j = (-1)^e_{0j} = 1 - 2*e_{0j} for a given syndrome.
        
        Args:
            syndrome: Binary syndrome vector
            
        Returns:
            Array of eta values
        """
        syndrome = np.asarray(syndrome, dtype=int)
        self.decoder.set_initial_syndrome(syndrome)
        return self.decoder.compute_eta()
    
    def set_parameters(self, max_iterations: int, tolerance: float) -> None:
        """
        Update convergence parameters.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.decoder.set_parameters(max_iterations, tolerance)
    
    @property
    def code_info(self) -> dict:
        """Get information about the current code."""
        if self._D_x is None:
            return {"status": "No code set"}
        
        return {
            "num_qubits": self._num_qubits,
            "num_checks": self._num_checks,
            "num_logical": self._num_logical,
            "check_matrix_shape": self._D_x.shape,
            "logical_matrix_shape": self._L_x.shape,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
        }


def create_uniform_error_model(num_qubits: int, error_prob: float) -> np.ndarray:
    """
    Create a uniform error model for all qubits.
    
    Args:
        num_qubits: Number of qubits
        error_prob: Error probability for each qubit
        
    Returns:
        Array of error probabilities
    """
    return np.full(num_qubits, error_prob)


def create_depolarizing_error_model(num_qubits: int, 
                                  p_x: float, 
                                  p_y: float, 
                                  p_z: float) -> np.ndarray:
    """
    Create a depolarizing error model.
    
    The total error probability for each qubit is p_x + p_y + p_z.
    
    Args:
        num_qubits: Number of qubits
        p_x: X error probability
        p_y: Y error probability  
        p_z: Z error probability
        
    Returns:
        Array of total error probabilities
    """
    total_prob = p_x + p_y + p_z
    return np.full(num_qubits, total_prob)


def decode_css_code(D_x: sp.spmatrix, 
                   L_x: sp.spmatrix, 
                   syndrome: np.ndarray, 
                   error_probs: np.ndarray,
                   max_iterations: int = 100,
                   tolerance: float = 1e-6) -> np.ndarray:
    """
    Convenience function for one-shot decoding of a CSS code.
    
    Args:
        D_x: Check matrix in sparse format
        L_x: Logical operator matrix in sparse format
        syndrome: Binary syndrome vector
        error_probs: Error probabilities for each qubit
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence tolerance
        
    Returns:
        Binary logical correction vector
    """
    decoder = QLDPCDecoder(max_iterations, tolerance)
    decoder.set_code(D_x, L_x)
    return decoder.decode_syndrome(syndrome, error_probs) 