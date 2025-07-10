#!/usr/bin/env python3
"""
Scaling analysis for the BP decoder with accumulated likelihood.
Tests multiple code sizes to analyze the scaling of logical error rates.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time
import json
from datetime import datetime
import multiprocessing as mp
from functools import partial
from qldpcdecoding.codes import gen_BB_code


def run_trials_for_error_rate(args):
    """Run trials for a single error rate (for parallel processing)."""
    code_size, p_error, num_trials, bp_iterations, seed = args
    
    try:
        from BPdecoder import likelihoodDecoder
    except ImportError:
        return None
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate code
    css_code = gen_BB_code(code_size)
    D = np.hstack([np.eye(css_code.hz.shape[0]), css_code.hz])
    DL = np.hstack([np.zeros((css_code.lz.shape[0], css_code.hz.shape[0])), css_code.lz])
    
    # Create decoder
    error_probs = [p_error] * D.shape[1]
    decoder = likelihoodDecoder(D, DL, error_probs, bp_iterations)
    
    # Run trials
    logical_errors = 0
    decoding_failures = 0
    
    for trial in range(num_trials):
        # Generate error
        error = np.zeros(D.shape[1]).astype(np.int32)
        for q in range(D.shape[1]):
            if np.random.rand() < p_error:
                error[q] = 1
        
        # Calculate syndrome
        syndrome = np.mod(D @ error, 2).astype(np.int32)
        
        # Decode
        decoded_logical = decoder.decode(syndrome)
        
        # Check results
        actual_logical = np.mod(DL @ error, 2).astype(np.int32)
        
        if not np.array_equal(decoded_logical, actual_logical):
            decoding_failures += 1
            if np.any(actual_logical != 0):
                logical_errors += 1
    
    return {
        'logical_error_rate': logical_errors / num_trials,
        'decoding_failure_rate': decoding_failures / num_trials,
        'num_trials': num_trials
    }

def estimate_threshold(code_sizes: List[int], 
                      p_errors: List[float],
                      num_trials: int = 1000,
                      bp_iterations: int = 20,
                      num_processes: int = None) -> Dict:
    """
    Estimate error threshold by testing multiple code sizes.
    
    Args:
        code_sizes: List of code sizes to test
        p_errors: List of physical error rates
        num_trials: Number of trials per (code_size, p_error) pair
        bp_iterations: Number of BP iterations
        num_processes: Number of parallel processes (None for auto)
    
    Returns:
        Dictionary with results for all code sizes
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"=== Scaling Analysis for BP Decoder ===")
    print(f"Code sizes: {code_sizes}")
    print(f"Error rates: {p_errors}")
    print(f"Trials per point: {num_trials}")
    print(f"BP iterations: {bp_iterations}")
    print(f"Using {num_processes} parallel processes")
    print()
    
    all_results = {}
    
    for code_size in code_sizes:
        print(f"\nProcessing code size n={code_size}...")
        start_time = time.time()
        
        # Prepare arguments for parallel processing
        args_list = []
        for i, p_error in enumerate(p_errors):
            # Different seed for each combination
            seed = 42 + code_size * 100 + i
            args_list.append((code_size, p_error, num_trials, bp_iterations, seed))
        
        # Run in parallel
        with mp.Pool(num_processes) as pool:
            results = pool.map(run_trials_for_error_rate, args_list)
        
        # Check for import errors
        if any(r is None for r in results):
            print("Error: Failed to import BPdecoder module")
            return None
        
        # Store results
        logical_rates = [r['logical_error_rate'] for r in results]
        decode_fail_rates = [r['decoding_failure_rate'] for r in results]
        
        all_results[code_size] = {
            'p_errors': p_errors,
            'logical_error_rates': logical_rates,
            'decoding_failure_rates': decode_fail_rates,
            'runtime': time.time() - start_time
        }
        
        print(f"Completed in {all_results[code_size]['runtime']:.1f} seconds")
        
        # Print summary
        print(f"{'p_error':>10} | {'Logical Rate':>12} | {'Decode Fail':>12}")
        print("-" * 40)
        for i, p in enumerate(p_errors):
            print(f"{p:>10.4f} | {logical_rates[i]:>12.6f} | {decode_fail_rates[i]:>12.6f}")
    
    return all_results

def plot_scaling_results(results: Dict, save_prefix: str = "scaling"):
    """Create plots for scaling analysis."""
    if results is None:
        return
    
    # Plot 1: Logical error rate vs physical error rate for different code sizes
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for i, (code_size, data) in enumerate(sorted(results.items())):
        p_errors = data['p_errors']
        logical_rates = data['logical_error_rates']
        
        # Skip points with zero error rate for log plot
        non_zero = [(p, l) for p, l in zip(p_errors, logical_rates) if l > 0]
        if non_zero:
            p_plot, l_plot = zip(*non_zero)
            ax1.loglog(p_plot, l_plot, 'o-', color=colors[i], 
                      label=f'n={code_size}', markersize=8, linewidth=2)
    
    # Add reference line
    p_ref = np.array([1e-4, 1e-1])
    ax1.loglog(p_ref, p_ref, 'k--', alpha=0.5, label='p_L = p')
    
    ax1.set_xlabel('Physical Error Rate (p)', fontsize=12)
    ax1.set_ylabel('Logical Error Rate (p_L)', fontsize=12)
    ax1.set_title('Scaling of Logical Error Rate with Code Size', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_error_rates.png', dpi=150)
    plt.show()
    
    # Plot 2: Finite-size scaling analysis
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # For each error rate, plot logical rate vs code size
    unique_p_errors = results[list(results.keys())[0]]['p_errors']
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(unique_p_errors)))
    
    for i, p_error in enumerate(unique_p_errors):
        code_sizes = []
        logical_rates = []
        
        for code_size, data in sorted(results.items()):
            idx = data['p_errors'].index(p_error)
            if data['logical_error_rates'][idx] > 0:  # Skip zero rates
                code_sizes.append(code_size)
                logical_rates.append(data['logical_error_rates'][idx])
        
        if code_sizes:
            ax2.semilogy(code_sizes, logical_rates, 'o-', color=colors2[i],
                        label=f'p={p_error:.4f}', markersize=8, linewidth=2)
    
    ax2.set_xlabel('Code Size (n)', fontsize=12)
    ax2.set_ylabel('Logical Error Rate (p_L)', fontsize=12)
    ax2.set_title('Finite-Size Scaling Analysis', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_finite_size.png', dpi=150)
    plt.show()
    
    # Plot 3: Threshold estimation
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    # Calculate effective logical error rate p_L/p for threshold estimation
    for i, (code_size, data) in enumerate(sorted(results.items())):
        p_errors = np.array(data['p_errors'])
        logical_rates = np.array(data['logical_error_rates'])
        
        # Calculate ratio where both are non-zero
        mask = (logical_rates > 0) & (p_errors > 0)
        if np.any(mask):
            p_plot = p_errors[mask]
            ratio = logical_rates[mask] / p_errors[mask]
            
            ax3.semilogx(p_plot, ratio, 'o-', color=colors[i],
                        label=f'n={code_size}', markersize=8, linewidth=2)
    
    ax3.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Physical Error Rate (p)', fontsize=12)
    ax3.set_ylabel('p_L / p', fontsize=12)
    ax3.set_title('Threshold Estimation', fontsize=14)
    ax3.legend(loc='best')
    ax3.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_threshold.png', dpi=150)
    plt.show()
    
    print(f"\nPlots saved with prefix: {save_prefix}")

def estimate_threshold_value(results: Dict) -> float:
    """Estimate the error threshold from scaling data."""
    if not results:
        return None
    
    # Simple estimation: find where p_L/p crosses 1 for largest code
    largest_code = max(results.keys())
    data = results[largest_code]
    
    p_errors = np.array(data['p_errors'])
    logical_rates = np.array(data['logical_error_rates'])
    
    # Find crossing point
    for i in range(len(p_errors) - 1):
        if logical_rates[i] > 0 and logical_rates[i+1] > 0:
            ratio1 = logical_rates[i] / p_errors[i]
            ratio2 = logical_rates[i+1] / p_errors[i+1]
            
            if ratio1 > 1 and ratio2 < 1:
                # Linear interpolation in log space
                log_p1, log_p2 = np.log(p_errors[i]), np.log(p_errors[i+1])
                log_r1, log_r2 = np.log(ratio1), np.log(ratio2)
                
                # Find where log(ratio) = 0
                log_p_th = log_p1 - log_r1 * (log_p2 - log_p1) / (log_r2 - log_r1)
                return np.exp(log_p_th)
    
    return None

def main():
    """Run scaling analysis."""
    # Test parameters
    code_sizes = [72, 90, 144, 288]  # Different code sizes
    p_errors = [ 0.0005,0.0008, 0.001, 0.003, 0.005, 0.008]
    num_trials = 10000  # Reduced for faster testing with multiple sizes
    bp_iterations = 100
    
    # Run analysis
    results = estimate_threshold(
        code_sizes=code_sizes,
        p_errors=p_errors,
        num_trials=num_trials,
        bp_iterations=bp_iterations
    )
    
    if results:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'scaling_analysis_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create plots
        plot_scaling_results(results, save_prefix=f'scaling_{timestamp}')
        
        # Estimate threshold
        threshold = estimate_threshold_value(results)
        if threshold:
            print(f"\nEstimated error threshold: p_th ≈ {threshold:.5f}")
        else:
            print("\nCould not estimate threshold from data")
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Code sizes tested: {code_sizes}")
        print(f"Total runtime: {sum(data['runtime'] for data in results.values()):.1f} seconds")

if __name__ == "__main__":
    main() 