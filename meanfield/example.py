#!/usr/bin/env python3
"""
Example usage of the Mean-Field QLDPC Decoder

This script demonstrates how to use the mean-field decoder with a simple
CSS code example. It shows how to:
1. Set up a CSS code with check and logical matrices
2. Create error models
3. Generate syndromes and decode them
4. Analyze the results
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from qlpc_decoder import QLDPCDecoder, create_uniform_error_model, decode_css_code
from qldpcdecoding.codes import gen_BB_code, gen_HP_ring_code, get_benchmark_code


if __name__ == "__main__":
    D_x_data = gen_BB_code(144).hx
    L_x_data = gen_BB_code(144).lx
    
    # Convert to sparse matrices
    D_x = sp.csr_matrix(D_x_data, dtype=int)
    L_x = sp.csr_matrix(L_x_data, dtype=int)
    print(f"Code: {D_x.shape[0]} qubits, {D_x.shape[1]} checks, {L_x.shape[1]} logical operators")
    
    # Create decoder
    decoder = QLDPCDecoder(max_iterations=50, tolerance=1e-6)
    decoder.set_code(D_x, L_x)
    DZ= gen_BB_code(144).hz
    LZ= gen_BB_code(144).lz
    # Create error model
    p = 0.001
    num_trials = 1000
    success = 0
    for j in range(num_trials):
        error_pattern = np.zeros(DZ.shape[1])
        for i in range(DZ.shape[1]):
            if np.random.rand() < p:
                error_pattern[i] = 1
        syndrome = (DZ @ error_pattern) % 2
        randomstablizer = D_x[np.random.randint(0, D_x.shape[0])]
        init_error_pattern = (error_pattern + randomstablizer) % 2
        ## flatten the error pattern
        init_error_pattern = init_error_pattern.reshape(-1)
        logical_correction = decoder.decode_syndrome(init_error_pattern.tolist(), error_probs=np.ones(DZ.shape[1]) * p)
        true_logical_correction =  (LZ @ error_pattern) % 2
        if np.array_equal(true_logical_correction, logical_correction):
            success += 1
        else:
            print(f"Error pattern: {error_pattern}")
            print(f"Logical correction: {logical_correction}")
            print(f"True logical correction: {true_logical_correction}")
    print(f"Success rate: {success / num_trials}")
    