#include <cuda_runtime.h>

// CUDA kernel for CSC matrix-vector multiplication (mod 2)
__global__ void csc_multiply_mod2(int* values, int* row_indices, int* col_ptr,
                                 int* vec, int* result, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    // Initialize result for this row
    result[row] = 0;
    
    // Process each column
    for (int col = 0; col < cols; col++) {
        if (vec[col]) {  // Only process if input vector has 1
            // Check if this row has a non-zero entry in this column
            for (int k = col_ptr[col]; k < col_ptr[col + 1]; k++) {
                if (row_indices[k] == row) {
                    result[row] = result[row] ^ 1;  // XOR with 1
                    break;
                }
            }
        }
    }
}

// Optimized CSC matrix-vector multiplication using shared memory
__global__ void csc_multiply_mod2_optimized(int* values, int* row_indices, int* col_ptr,
                                           int* vec, int* result, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    // Initialize result for this row
    result[row] = 0;
    
    // Process each column
    for (int col = 0; col < cols; col++) {
        if (vec[col]) {  // Only process if input vector has 1
            // Binary search for row in this column's non-zero entries
            int start = col_ptr[col];
            int end = col_ptr[col + 1];
            
            while (start < end) {
                int mid = (start + end) / 2;
                if (row_indices[mid] == row) {
                    result[row] = result[row] ^ 1;  // XOR with 1
                    break;
                } else if (row_indices[mid] < row) {
                    start = mid + 1;
                } else {
                    end = mid;
                }
            }
        }
    }
}

// Batch CSC matrix-vector multiplication for multiple error patterns
__global__ void batch_csc_multiply_mod2(int* values, int* row_indices, int* col_ptr,
                                       int* errors, int* syndromes, int* logical_syndromes,
                                       int rows, int cols, int dl_rows, int total_errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int error_idx = idx / rows;
    int row = idx % rows;
    
    if (error_idx >= total_errors || row >= rows) return;
    
    // Initialize result for this row and error pattern
    if (row < rows - dl_rows) {
        syndromes[error_idx * (rows - dl_rows) + row] = 0;
    } else {
        logical_syndromes[error_idx * dl_rows + (row - (rows - dl_rows))] = 0;
    }
    
    // Process each column
    for (int col = 0; col < cols; col++) {
        if (errors[error_idx * cols + col]) {  // Only process if error has 1
            // Check if this row has a non-zero entry in this column
            for (int k = col_ptr[col]; k < col_ptr[col + 1]; k++) {
                if (row_indices[k] == row) {
                    if (row < rows - dl_rows) {
                        syndromes[error_idx * (rows - dl_rows) + row] ^= 1;
                    } else {
                        logical_syndromes[error_idx * dl_rows + (row - (rows - dl_rows))] ^= 1;
                    }
                    break;
                }
            }
        }
    }
}

// Host function to launch CSC matrix-vector multiplication
extern "C" {
    void cuda_csc_multiply_mod2(int* h_values, int* h_row_indices, int* h_col_ptr,
                               int* h_vec, int* h_result, int rows, int cols) {
        
        // Device memory allocation
        int* d_values;
        int* d_row_indices;
        int* d_col_ptr;
        int* d_vec;
        int* d_result;
        
        cudaMalloc(&d_values, h_col_ptr[cols] * sizeof(int));
        cudaMalloc(&d_row_indices, h_col_ptr[cols] * sizeof(int));
        cudaMalloc(&d_col_ptr, (cols + 1) * sizeof(int));
        cudaMalloc(&d_vec, cols * sizeof(int));
        cudaMalloc(&d_result, rows * sizeof(int));
        
        // Copy data to device
        cudaMemcpy(d_values, h_values, h_col_ptr[cols] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_indices, h_row_indices, h_col_ptr[cols] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_ptr, h_col_ptr, (cols + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vec, h_vec, cols * sizeof(int), cudaMemcpyHostToDevice);
        
        // Setup kernel configuration
        int block_size = 256;
        int grid_size = (rows + block_size - 1) / block_size;
        
        // Launch kernel
        csc_multiply_mod2<<<grid_size, block_size>>>(d_values, d_row_indices, d_col_ptr,
                                                    d_vec, d_result, rows, cols);
        cudaDeviceSynchronize();
        
        // Copy result back to host
        cudaMemcpy(h_result, d_result, rows * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Free device memory
        cudaFree(d_values);
        cudaFree(d_row_indices);
        cudaFree(d_col_ptr);
        cudaFree(d_vec);
        cudaFree(d_result);
    }
    
    void cuda_batch_csc_multiply_mod2(int* h_values, int* h_row_indices, int* h_col_ptr,
                                     int* h_errors, int* h_syndromes, int* h_logical_syndromes,
                                     int rows, int cols, int dl_rows, int total_errors) {
        
        // Device memory allocation
        int* d_values;
        int* d_row_indices;
        int* d_col_ptr;
        int* d_errors;
        int* d_syndromes;
        int* d_logical_syndromes;
        
        cudaMalloc(&d_values, h_col_ptr[cols] * sizeof(int));
        cudaMalloc(&d_row_indices, h_col_ptr[cols] * sizeof(int));
        cudaMalloc(&d_col_ptr, (cols + 1) * sizeof(int));
        cudaMalloc(&d_errors, total_errors * cols * sizeof(int));
        cudaMalloc(&d_syndromes, total_errors * (rows - dl_rows) * sizeof(int));
        cudaMalloc(&d_logical_syndromes, total_errors * dl_rows * sizeof(int));
        
        // Copy data to device
        cudaMemcpy(d_values, h_values, h_col_ptr[cols] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_indices, h_row_indices, h_col_ptr[cols] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_ptr, h_col_ptr, (cols + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_errors, h_errors, total_errors * cols * sizeof(int), cudaMemcpyHostToDevice);
        
        // Setup kernel configuration
        int block_size = 256;
        int grid_size = (total_errors * rows + block_size - 1) / block_size;
        
        // Launch kernel
        batch_csc_multiply_mod2<<<grid_size, block_size>>>(d_values, d_row_indices, d_col_ptr,
                                                          d_errors, d_syndromes, d_logical_syndromes,
                                                          rows, cols, dl_rows, total_errors);
        cudaDeviceSynchronize();
        
        // Copy results back to host
        cudaMemcpy(h_syndromes, d_syndromes, total_errors * (rows - dl_rows) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_logical_syndromes, d_logical_syndromes, total_errors * dl_rows * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Free device memory
        cudaFree(d_values);
        cudaFree(d_row_indices);
        cudaFree(d_col_ptr);
        cudaFree(d_errors);
        cudaFree(d_syndromes);
        cudaFree(d_logical_syndromes);
    }
} 