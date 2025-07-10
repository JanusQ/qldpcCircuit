#include <cuda_runtime.h>
#include <curand_kernel.h>

// CUDA kernel for CSC matrix-vector multiplication
__global__ void csc_multiply_kernel(int* values, int* row_indices, int* col_ptr,
                                   int* vec, int* result, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        result[row] = 0;
        for (int col = 0; col < cols; col++) {
            for (int k = col_ptr[col]; k < col_ptr[col + 1]; k++) {
                if (row_indices[k] == row) {
                    result[row] ^= (values[k] & vec[col]);
                }
            }
        }
    }
}

// CUDA kernel for batch error pattern generation
__global__ void generate_error_patterns_kernel(int* patterns, int n, int max_weight, 
                                              int total_patterns, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_patterns) {
        curandState localState = states[idx];
        
        // Generate random error pattern with weight <= max_weight
        int weight = 0;
        for (int i = 0; i < n; i++) {
            float rand_val = curand_uniform(&localState);
            int bit = (rand_val < 0.1) ? 1 : 0;  // 10% probability of error
            patterns[idx * n + i] = bit;
            weight += bit;
        }
        
        // If weight exceeds max_weight, randomly flip bits to reduce weight
        while (weight > max_weight) {
            int flip_idx = curand(&localState) % n;
            if (patterns[idx * n + flip_idx] == 1) {
                patterns[idx * n + flip_idx] = 0;
                weight--;
            }
        }
        
        states[idx] = localState;
    }
}

// CUDA kernel for likelihood computation
__global__ void compute_likelihoods_kernel(int* syndromes, int* logical_syndromes,
                                          float* weights, float* likelihoods,
                                          int syndrome_size, int logical_size,
                                          int total_patterns) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_patterns) {
        // Compute syndrome hash for lookup
        int hash = 0;
        for (int i = 0; i < syndrome_size; i++) {
            hash = hash * 31 + syndromes[idx * syndrome_size + i];
        }
        
        // Compute logical syndrome hash
        int logical_hash = 0;
        for (int i = 0; i < logical_size; i++) {
            logical_hash = logical_hash * 31 + logical_syndromes[idx * logical_size + i];
        }
        
        // Store likelihood (simplified - in practice would use atomic operations)
        likelihoods[idx] = hash + logical_hash * 0.001f;  // Placeholder
    }
}

// Kernel to initialize random states
__global__ void init_random_states_kernel(curandState* states, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// External C functions for calling from C++
extern "C" {
    void cuda_csc_multiply(int* values, int* row_indices, int* col_ptr,
                          int* vec, int* result, int rows, int cols) {
        int block_size = 256;
        int grid_size = (rows + block_size - 1) / block_size;
        csc_multiply_kernel<<<grid_size, block_size>>>(values, row_indices, col_ptr, vec, result, rows, cols);
    }
    
    void cuda_generate_patterns(int* patterns, int n, int max_weight, 
                               int total_patterns, curandState* states) {
        int block_size = 256;
        int grid_size = (total_patterns + block_size - 1) / block_size;
        generate_error_patterns_kernel<<<grid_size, block_size>>>(patterns, n, max_weight, total_patterns, states);
    }
    
    void cuda_compute_likelihoods(int* syndromes, int* logical_syndromes,
                                 float* weights, float* likelihoods,
                                 int syndrome_size, int logical_size, int total_patterns) {
        int block_size = 256;
        int grid_size = (total_patterns + block_size - 1) / block_size;
        compute_likelihoods_kernel<<<grid_size, block_size>>>(syndromes, logical_syndromes, weights, likelihoods, syndrome_size, logical_size, total_patterns);
    }
    
    void cuda_init_random_states(curandState* states, int num_states, unsigned long long seed) {
        int block_size = 256;
        int grid_size = (num_states + block_size - 1) / block_size;
        init_random_states_kernel<<<grid_size, block_size>>>(states, seed);
    }
} 