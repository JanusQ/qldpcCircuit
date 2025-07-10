#!/usr/bin/env python3
"""
Test script to measure the logical error rate of the BP decoder.
This script runs Monte Carlo simulations at different physical error rates
to determine the logical error rate.
"""
from enumlikelydecoder import EnumDecoder
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
import json
from datetime import datetime
from qldpcdecoding.codes import gen_BB_code

def run_single_trial(decoder, D, DL, p_error):
    """
    Run a single decoding trial.
    
    Returns:
        (bool, bool): (decoding_success, logical_error_corrected)
    """
    n = D.shape[1]
    
    # Generate random error
    error = np.zeros(n).astype(int)
    for q in range(n):
        if np.random.rand() < p_error:
            error[q] = 1
    
    # Calculate syndrome
    syndrome = np.mod(D @ error, 2).astype(np.int32)
    
    # Decode
    decoded_logical = decoder.decode(syndrome)
    
    # Calculate actual logical syndrome
    actual_logical = np.mod(DL @ error, 2).astype(np.int32)
    
    # Check if decoding is correct
    decoding_success = np.array_equal(decoded_logical, actual_logical)
    
    # Check if there's a logical error (non-zero logical syndrome)
    has_logical_error = np.any(actual_logical != 0)
    
    return decoding_success, has_logical_error, np.sum(error)

def estimate_logical_error_rate(code_distance: int, 
                               p_errors: List[float], 
                               num_trials: int = 1000,
                               bp_iterations: int = 10) -> Dict:
    """
    Estimate logical error rates at different physical error rates.
    
    Args:
        code_distance: The code distance (or size parameter)
        p_errors: List of physical error probabilities to test
        num_trials: Number of Monte Carlo trials per error rate
        bp_iterations: Number of BP iterations
        
    Returns:
        Dictionary with results
    """
    try:
        from BPdecoder import likelihoodDecoder
    except ImportError:
        print("Error: BPdecoder module not found!")
        print("Please build the module first by running: make build")
        return None
    
    
    
    
    # Generate code
    print(f"Generating QLDPC code with n={code_distance}...")
    css_code = gen_BB_code(code_distance)  # Fixed seed for reproducibility
    # D = css_code.hz
    # DL = css_code.lz
    D = np.hstack([np.eye(css_code.hz.shape[0]), css_code.hz])
    DL = np.hstack([np.zeros((css_code.lz.shape[0], css_code.hz.shape[0])), css_code.lz])
    print(f"Code parameters:")
    print(f"  - Number of qubits: {D.shape[1]}")
    print(f"  - Number of checks: {D.shape[0]}")
    print(f"  - Number of logical qubits: {DL.shape[0]}")
    print(f"  - Average check weight: {np.mean(np.sum(D, axis=1)):.2f}")
    print(f"  - Average qubit degree: {np.mean(np.sum(D, axis=0)):.2f}")
    
    results = {
        'code_distance': code_distance,
        'num_qubits': D.shape[1],
        'num_checks': D.shape[0],
        'num_logical': DL.shape[0],
        'num_trials': num_trials,
        'bp_iterations': bp_iterations,
        'p_errors': p_errors,
        'logical_error_ratesBP': [],
        'decoding_failure_ratesBP': [],
        'logical_error_ratesEnum': [],
        'decoding_failure_ratesEnum': [],
        'avg_error_weights': [],
        'std_error_weights': [],
        'runtimeBP': [],
        'runtimeEnum': []
    }
    
    print(f"\nRunning {num_trials} trials for each error rate...")
    print("-" * 70)
    print(f"{'p_error':>10} |{'Method':>10} | {'Logical Err Rate':>16} | {'Decode Fail Rate':>16} | {'Avg Weight':>10} | {'Time (s)':>8}")
    print("-" * 70)
    
    for p_error in p_errors:
        # Create decoder for this error rate
        error_probs = [p_error] * D.shape[1]
        decoderBP = likelihoodDecoder(D, DL, error_probs, bp_iterations)
        decoderEnum = EnumDecoder(D, DL, error_probs, 4)
        
        # Run trials
        logical_errorsBP = 0
        decoding_failuresBP = 0
        decoding_failuresEnum = 0
        logical_errorsEnum = 0

        error_weights = []
        
        start_time = time.time()
        
        for trial in range(num_trials):
  
            successBP, has_logical_error, error_weightBP = run_single_trial(decoderBP, D, DL, p_error)
            
            if not successBP:
                decoding_failuresBP += 1
                if has_logical_error:
                    logical_errorsBP += 1

            error_weights.append(error_weightBP)
        elapsed_time = time.time() - start_time
        start_timeEnum = time.time()
        for trial in range(num_trials):
            successEnum, has_logical_errorEnum, error_weightEnum = run_single_trial(decoderEnum, D, DL, p_error)

            if not successEnum:
                decoding_failuresEnum += 1
                if has_logical_errorEnum:
                    logical_errorsEnum += 1

        
        elapsed_timeEnum = time.time() - start_timeEnum
        # Calculate rates
        logical_error_rateBP = logical_errorsBP / num_trials
        decoding_failure_rateBP = decoding_failuresBP / num_trials
        logical_error_rateEnum = logical_errorsEnum / num_trials
        decoding_failure_rateEnum = decoding_failuresEnum / num_trials
        avg_weight = np.mean(error_weights)
        std_weight = np.std(error_weights)
        
        # Store results
        results['logical_error_ratesBP'].append(logical_error_rateBP)
        results['decoding_failure_ratesBP'].append(decoding_failure_rateBP)
        results['logical_error_ratesEnum'].append(logical_error_rateEnum)
        results['decoding_failure_ratesEnum'].append(decoding_failure_rateEnum)
        results['avg_error_weights'].append(avg_weight)
        results['std_error_weights'].append(std_weight)
        results['runtimeBP'].append(elapsed_time)
        results['runtimeEnum'].append(elapsed_timeEnum)
        # Print progress
        print(f"{p_error:>10.4f} |{'BP':>10} | {logical_error_rateBP:>16.6f} | {decoding_failure_rateBP:>16.6f} | {avg_weight:>10.2f} | {elapsed_time:>8.2f}")
        print(f"{p_error:>10.4f} |{'Enum':>10} | {logical_error_rateEnum:>16.6f} | {decoding_failure_rateEnum:>16.6f} | {avg_weight:>10.2f} | {elapsed_timeEnum:>8.2f}")
    
    print("-" * 70)
    
    return results

def plot_results(results: Dict, save_path: str = None):
    """Plot the logical error rates vs physical error rates."""
    if results is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Error rates
    p_errors = results['p_errors']
    logical_ratesBP = results['logical_error_ratesBP']
    decode_fail_ratesBP = results['decoding_failure_ratesBP']
    logical_ratesEnum = results['logical_error_ratesEnum']
    decode_fail_ratesEnum = results['decoding_failure_ratesEnum']
    
    ax1.semilogy(p_errors, logical_ratesBP, 'o-', label='Logical Error Rate', markersize=8)
    ax1.semilogy(p_errors, decode_fail_ratesBP, 's--', label='Decoding Failure Rate', markersize=8)
    ax1.semilogy(p_errors, logical_ratesEnum, 'o-', label='Logical Error Rate', markersize=8)
    ax1.semilogy(p_errors, decode_fail_ratesEnum, 's--', label='Decoding Failure Rate', markersize=8)
    ax1.semilogy(p_errors, p_errors, ':', color='gray', label='Physical Error Rate')
    
    ax1.set_xlabel('Physical Error Rate')
    ax1.set_ylabel('Error Rate')
    ax1.set_title(f'BP Decoder Performance (n={results["num_qubits"]}, {results["num_trials"]} trials)')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    # Plot 2: Average error weight
    avg_weights = results['avg_error_weights']
    std_weights = results['std_error_weights']
    
    ax2.errorbar(p_errors, avg_weights, yerr=std_weights, fmt='o-', capsize=5)
    ax2.set_xlabel('Physical Error Rate')
    ax2.set_ylabel('Average Error Weight')
    ax2.set_title('Average Number of Errors per Trial')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()

def save_results(results: Dict, filename: str = None):
    """Save results to JSON file."""
    if results is None:
        return
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logical_error_rate_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")

def main():
    """Main function to run logical error rate tests."""
    print("=== Logical Error Rate Test for BP Decoder ===\n")
    
    # Test parameters
    code_distance = 72  # Number of qubits
    p_errors = [ 0.0005,0.0008, 0.001, 0.003, 0.005, 0.008]
    num_trials = 10000  # More trials for better statistics
    bp_iterations = 100  # More iterations for better decoding
    
    # Run tests
    results = estimate_logical_error_rate(
        code_distance=code_distance,
        p_errors=p_errors,
        num_trials=num_trials,
        bp_iterations=bp_iterations
    )
    
    if results:
        # Save results
        save_results(results)
        
        # Plot results
        plot_results(results, save_path="logical_error_rate_plot.png")
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Code size: {results['num_qubits']} qubits")
        print(f"Trials per error rate: {results['num_trials']}")
        print(f"BP iterations: {results['bp_iterations']}")
        
        # Find threshold (where logical error rate ≈ physical error rate)
        logical_ratesBP = np.array(results['logical_error_ratesBP'])
        logical_ratesEnum = np.array(results['logical_error_ratesEnum'])
        p_errors_arr = np.array(results['p_errors'])
        
        # Find crossover point
        for i in range(len(logical_ratesBP)-1):
            if logical_ratesBP[i] < p_errors_arr[i] and logical_ratesBP[i+1] > p_errors_arr[i+1]:
                # Linear interpolation to estimate threshold
                x1, y1 = np.log(p_errors_arr[i]), np.log(logical_ratesBP[i]/p_errors_arr[i])
                x2, y2 = np.log(p_errors_arr[i+1]), np.log(logical_ratesBP[i+1]/p_errors_arr[i+1])
                threshold = np.exp(x1 - y1 * (x2-x1)/(y2-y1))
                print(f"\nEstimated threshold: p_th ≈ {threshold:.4f}")
                break
        for i in range(len(logical_ratesEnum)-1):
            if logical_ratesEnum[i] < p_errors_arr[i] and logical_ratesEnum[i+1] > p_errors_arr[i+1]:
                # Linear interpolation to estimate threshold
                x1, y1 = np.log(p_errors_arr[i]), np.log(logical_ratesEnum[i]/p_errors_arr[i])
                x2, y2 = np.log(p_errors_arr[i+1]), np.log(logical_ratesEnum[i+1]/p_errors_arr[i+1])
                threshold = np.exp(x1 - y1 * (x2-x1)/(y2-y1))
                print(f"\nEstimated threshold: p_th ≈ {threshold:.4f}")
                break

if __name__ == "__main__":
    main() 