#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// CUDA kernel for generating all error patterns up to max_weight
__global__ void generate_all_errors(int* errors, int* error_weights, int n, int max_weight, int* total_errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate total number of error patterns
    int total = 0;
    for (int w = 0; w <= max_weight; w++) {
        int combinations = 1;
        for (int i = 0; i < w; i++) {
            combinations *= (n - i);
            combinations /= (i + 1);
        }
        total += combinations;
    }
    
    if (idx == 0) {
        *total_errors = total;
    }
    
    if (idx >= total) return;
    
    // Find which weight this pattern should have
    int current_idx = idx;
    int target_weight = 0;
    
    for (int w = 0; w <= max_weight; w++) {
        int combinations = 1;
        for (int i = 0; i < w; i++) {
            combinations *= (n - i);
            combinations /= (i + 1);
        }
        
        if (current_idx < combinations) {
            target_weight = w;
            break;
        }
        current_idx -= combinations;
    }
    
    // Generate error pattern with target_weight
    for (int i = 0; i < n; i++) {
        errors[idx * n + i] = 0;
    }
    
    // Generate combination using bit manipulation
    if (target_weight > 0) {
        int pattern = 0;
        int count = 0;
        
        // Generate the current_idx-th combination of target_weight bits
        for (int i = 0; i < (1 << n) && count < current_idx + 1; i++) {
            int bits = __popc(i);
            if (bits == target_weight) {
                if (count == current_idx) {
                    pattern = i;
                    break;
                }
                count++;
            }
        }
        
        // Set error pattern
        for (int i = 0; i < n; i++) {
            if (pattern & (1 << i)) {
                errors[idx * n + i] = 1;
            }
        }
    }
    
    // Set error weight
    error_weights[idx] = target_weight;
}

// CUDA kernel for computing syndrome from error pattern
__global__ void compute_syndrome_enum(int* errors, int* D_matrix, int* DL_matrix, 
                                     int* syndromes, int* logical_syndromes,
                                     int n, int d_rows, int dl_rows, int total_errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_errors) return;
    
    // Compute syndrome s = D * e (mod 2)
    for (int i = 0; i < d_rows; i++) {
        int val = 0;
        for (int j = 0; j < n; j++) {
            val ^= (D_matrix[i * n + j] & errors[idx * n + j]);
        }
        syndromes[idx * d_rows + i] = val;
    }
    
    // Compute logical syndrome sl = DL * e (mod 2)
    for (int i = 0; i < dl_rows; i++) {
        int val = 0;
        for (int j = 0; j < n; j++) {
            val ^= (DL_matrix[i * n + j] & errors[idx * n + j]);
        }
        logical_syndromes[idx * dl_rows + i] = val;
    }
}

// CUDA kernel for computing likelihood weights
__global__ void compute_likelihood_weights_enum(int* errors, float* weights, float* error_weights,
                                               int n, int total_errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_errors) return;
    
    float weight = 0.0f;
    for (int j = 0; j < n; j++) {
        if (errors[idx * n + j] == 1) {
            weight += weights[j];
        }
    }
    error_weights[idx] = weight;
}

// Host function to launch CUDA kernels for enumeration
extern "C" {
    void cuda_enumerate_errors(int* h_errors, float* h_weights, int* h_D_matrix, int* h_DL_matrix,
                              int* h_syndromes, int* h_logical_syndromes, float* h_error_weights,
                              int n, int d_rows, int dl_rows, int max_weight, int* total_errors) {
        
        // Calculate total number of error patterns
        int total = 0;
        for (int w = 0; w <= max_weight; w++) {
            int combinations = 1;
            for (int i = 0; i < w; i++) {
                combinations *= (n - i);
                combinations /= (i + 1);
            }
            total += combinations;
        }
        
        // Limit total to prevent memory issues
        const int MAX_PATTERNS = 1000000; // 1 million patterns max
        if (total > MAX_PATTERNS) {
            total = MAX_PATTERNS;
            printf("Warning: Limiting to %d patterns due to memory constraints\n", MAX_PATTERNS);
        }
        
        *total_errors = total;
        
        // Device memory allocation
        int* d_errors;
        float* d_weights;
        int* d_D_matrix;
        int* d_DL_matrix;
        int* d_syndromes;
        int* d_logical_syndromes;
        float* d_error_weights;
        int* d_total_errors;
        
        cudaMalloc(&d_errors, total * n * sizeof(int));
        cudaMalloc(&d_weights, n * sizeof(float));
        cudaMalloc(&d_D_matrix, d_rows * n * sizeof(int));
        cudaMalloc(&d_DL_matrix, dl_rows * n * sizeof(int));
        cudaMalloc(&d_syndromes, total * d_rows * sizeof(int));
        cudaMalloc(&d_logical_syndromes, total * dl_rows * sizeof(int));
        cudaMalloc(&d_error_weights, total * sizeof(float));
        cudaMalloc(&d_total_errors, sizeof(int));
        
        // Copy data to device
        cudaMemcpy(d_weights, h_weights, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_D_matrix, h_D_matrix, d_rows * n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_DL_matrix, h_DL_matrix, dl_rows * n * sizeof(int), cudaMemcpyHostToDevice);
        
        // Setup kernel configuration
        int block_size = 256;
        int grid_size = (total + block_size - 1) / block_size;
        
        // Generate all error patterns
        generate_all_errors<<<grid_size, block_size>>>(d_errors, d_error_weights, n, max_weight, d_total_errors);
        cudaDeviceSynchronize();
        
        // Compute syndromes
        compute_syndrome_enum<<<grid_size, block_size>>>(d_errors, d_D_matrix, d_DL_matrix,
                                                        d_syndromes, d_logical_syndromes,
                                                        n, d_rows, dl_rows, total);
        cudaDeviceSynchronize();
        
        // Compute likelihood weights
        compute_likelihood_weights_enum<<<grid_size, block_size>>>(d_errors, d_weights, d_error_weights,
                                                                  n, total);
        cudaDeviceSynchronize();
        
        // Copy results back to host
        cudaMemcpy(h_errors, d_errors, total * n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_syndromes, d_syndromes, total * d_rows * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_logical_syndromes, d_logical_syndromes, total * dl_rows * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_error_weights, d_error_weights, total * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Free device memory
        cudaFree(d_errors);
        cudaFree(d_weights);
        cudaFree(d_D_matrix);
        cudaFree(d_DL_matrix);
        cudaFree(d_syndromes);
        cudaFree(d_logical_syndromes);
        cudaFree(d_error_weights);
        cudaFree(d_total_errors);
    }
} 