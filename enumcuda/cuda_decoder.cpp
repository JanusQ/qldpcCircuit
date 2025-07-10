#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <map>
#include <bitset>
#include <cassert>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "csc_matrix.h"

// External CUDA function declarations
extern "C" {
    void cuda_csc_multiply(int* values, int* row_indices, int* col_ptr,
                          int* vec, int* result, int rows, int cols);
    void cuda_generate_patterns(int* patterns, int n, int max_weight, 
                               int total_patterns, curandState* states);
    void cuda_compute_likelihoods(int* syndromes, int* logical_syndromes,
                                 float* weights, float* likelihoods,
                                 int syndrome_size, int logical_size, int total_patterns);
    void cuda_init_random_states(curandState* states, int num_states, unsigned long long seed);
}

using namespace std;

using Vec = vector<bool>;
using Matrix = vector<Vec>;
using Real = double;



class CUDADecoder {
private:
    CSCMatrix D_csc;
    CSCMatrix DL_csc;
    CSCMatrix Dfull_csc;
    vector<Real> weights;
    int max_weight;
    int batch_size;
    
    // Device memory
    int *d_values, *d_row_indices, *d_col_ptr;
    int *d_patterns, *d_syndromes, *d_logical_syndromes;
    float *d_weights, *d_likelihoods;
    curandState *d_states;
    
    // Host memory for results
    vector<int> h_syndromes, h_logical_syndromes;
    vector<float> h_likelihoods;
    
    // Hash table for syndrome lookup (simplified)
    map<vector<int>, vector<pair<vector<int>, Real>>> likelihood_table;

    void initialize_cuda() {
        cudaSetDevice(0);
        
        // Allocate device memory for CSC matrices
        cudaMalloc(&d_values, Dfull_csc.values.size() * sizeof(int));
        cudaMalloc(&d_row_indices, Dfull_csc.row_indices.size() * sizeof(int));
        cudaMalloc(&d_col_ptr, Dfull_csc.col_ptr.size() * sizeof(int));
        
        // Copy CSC matrix data to device
        cudaMemcpy(d_values, Dfull_csc.values.data(), 
                   Dfull_csc.values.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_indices, Dfull_csc.row_indices.data(), 
                   Dfull_csc.row_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_ptr, Dfull_csc.col_ptr.data(), 
                   Dfull_csc.col_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
        
        // Allocate device memory for batch processing
        cudaMalloc(&d_patterns, batch_size * Dfull_csc.cols * sizeof(int));
        cudaMalloc(&d_syndromes, batch_size * Dfull_csc.rows * sizeof(int));
        cudaMalloc(&d_logical_syndromes, batch_size * DL_csc.rows * sizeof(int));
        cudaMalloc(&d_weights, Dfull_csc.cols * sizeof(float));
        cudaMalloc(&d_likelihoods, batch_size * sizeof(float));
        cudaMalloc(&d_states, batch_size * sizeof(curandState));
        
        // Initialize random states
        cuda_init_random_states(d_states, batch_size, time(NULL));
        
        // Allocate host memory
        h_syndromes.resize(batch_size * Dfull_csc.rows);
        h_logical_syndromes.resize(batch_size * DL_csc.rows);
        h_likelihoods.resize(batch_size);
    }

    void cleanup_cuda() {
        cudaFree(d_values);
        cudaFree(d_row_indices);
        cudaFree(d_col_ptr);
        cudaFree(d_patterns);
        cudaFree(d_syndromes);
        cudaFree(d_logical_syndromes);
        cudaFree(d_weights);
        cudaFree(d_likelihoods);
        cudaFree(d_states);
    }

    void generate_error_patterns() {
        int n = Dfull_csc.cols;
        int total_patterns = batch_size;
        
        // Generate error patterns on GPU
        cuda_generate_patterns(d_patterns, n, max_weight, total_patterns, d_states);
        
        cudaDeviceSynchronize();
    }

    void compute_syndromes() {
        int rows = Dfull_csc.rows;
        int cols = Dfull_csc.cols;
        
        // Compute syndromes for all patterns
        cuda_csc_multiply(d_values, d_row_indices, d_col_ptr, d_patterns, d_syndromes, rows, cols);
        
        cudaDeviceSynchronize();
        
        // Copy results back to host
        cudaMemcpy(h_syndromes.data(), d_syndromes, 
                   batch_size * rows * sizeof(int), cudaMemcpyDeviceToHost);
    }

    void compute_logical_syndromes() {
        int logical_rows = DL_csc.rows;
        int cols = DL_csc.cols;
        
        // Extract logical part of syndromes
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < logical_rows; j++) {
                h_logical_syndromes[i * logical_rows + j] = 
                    h_syndromes[i * Dfull_csc.rows + D_csc.rows + j];
            }
        }
        
        // Copy to device for further processing
        cudaMemcpy(d_logical_syndromes, h_logical_syndromes.data(),
                   batch_size * logical_rows * sizeof(int), cudaMemcpyHostToDevice);
    }

    void compute_likelihoods() {
        int syndrome_size = D_csc.rows;
        int logical_size = DL_csc.rows;
        
        // Compute likelihoods on GPU
        cuda_compute_likelihoods(d_syndromes, d_logical_syndromes, d_weights, d_likelihoods,
                                syndrome_size, logical_size, batch_size);
        
        cudaDeviceSynchronize();
        
        // Copy results back to host
        cudaMemcpy(h_likelihoods.data(), d_likelihoods,
                   batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

public:
    CUDADecoder(const CSCMatrix& D_matrix, const CSCMatrix& DL_matrix,
                const vector<Real>& priors, int maximum_weight, int batch_size_ = 10000) {
        D_csc = D_matrix;
        DL_csc = DL_matrix;
        
        // Create Dfull matrix
        Dfull_csc.rows = D_matrix.rows + DL_matrix.rows;
        Dfull_csc.cols = D_matrix.cols;
        Dfull_csc.values = D_matrix.values;
        Dfull_csc.row_indices = D_matrix.row_indices;
        Dfull_csc.col_ptr = D_matrix.col_ptr;
        
        // Add DL matrix entries
        for (int j = 0; j < DL_matrix.cols; j++) {
            for (int k = DL_matrix.col_ptr[j]; k < DL_matrix.col_ptr[j + 1]; k++) {
                Dfull_csc.values.push_back(DL_matrix.values[k]);
                Dfull_csc.row_indices.push_back(DL_matrix.row_indices[k] + D_matrix.rows);
            }
        }
        
        // Update column pointers
        for (int j = 1; j <= Dfull_csc.cols; j++) {
            Dfull_csc.col_ptr[j] = D_matrix.col_ptr[j] + 
                                  (DL_matrix.col_ptr[j] - DL_matrix.col_ptr[j - 1]);
        }
        
        max_weight = maximum_weight;
        batch_size = batch_size_;
        
        // Convert weights to float for GPU
        weights.clear();
        for (size_t i = 0; i < Dfull_csc.cols; i++) {
            weights.push_back(log((1 - priors[i]) / priors[i]));
        }
        
        initialize_cuda();
        
        // Copy weights to device
        vector<float> fweights(weights.begin(), weights.end());
        cudaMemcpy(d_weights, fweights.data(), 
                   weights.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        cout << "CUDA Decoder initialized with batch size: " << batch_size << endl;
        cout << "D matrix: " << D_csc.rows << "x" << D_csc.cols << endl;
        cout << "DL matrix: " << DL_csc.rows << "x" << DL_csc.cols << endl;
    }
    
    CUDADecoder(const Matrix& D_matrix, const Matrix& DL_matrix,
                const vector<Real>& priors, int maximum_weight, int batch_size_ = 10000) {
        D_csc.from_dense(D_matrix);
        DL_csc.from_dense(DL_matrix);
        
        // Create Dfull matrix
        Matrix Dfull_dense = D_matrix;
        Dfull_dense.insert(Dfull_dense.end(), DL_matrix.begin(), DL_matrix.end());
        Dfull_csc.from_dense(Dfull_dense);
        
        max_weight = maximum_weight;
        batch_size = batch_size_;
        
        // Convert weights to float for GPU
        weights.clear();
        for (size_t i = 0; i < Dfull_csc.cols; i++) {
            weights.push_back(log((1 - priors[i]) / priors[i]));
        }
        
        initialize_cuda();
        
        // Copy weights to device
        vector<float> fweights(weights.begin(), weights.end());
        cudaMemcpy(d_weights, fweights.data(), 
                   weights.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        cout << "CUDA Decoder initialized with batch size: " << batch_size << endl;
    }

    ~CUDADecoder() {
        cleanup_cuda();
    }

    // Process one batch of error patterns
    void process_batch() {
        generate_error_patterns();
        compute_syndromes();
        compute_logical_syndromes();
        compute_likelihoods();
        
        // Update likelihood table (simplified)
        update_likelihood_table();
    }

    void update_likelihood_table() {
        int syndrome_size = D_csc.rows;
        int logical_size = DL_csc.rows;
        
        for (int i = 0; i < batch_size; i++) {
            // Extract syndrome
            vector<int> syndrome(h_syndromes.begin() + i * Dfull_csc.rows,
                               h_syndromes.begin() + i * Dfull_csc.rows + syndrome_size);
            
            // Extract logical syndrome
            vector<int> logical_syndrome(h_logical_syndromes.begin() + i * logical_size,
                                       h_logical_syndromes.begin() + (i + 1) * logical_size);
            
            // Update table (simplified - would need proper atomic operations in practice)
            if (likelihood_table.find(syndrome) == likelihood_table.end()) {
                likelihood_table[syndrome] = vector<pair<vector<int>, Real>>();
            }
            
            // Add to table
            likelihood_table[syndrome].push_back({logical_syndrome, h_likelihoods[i]});
        }
    }

    // Decode a syndrome using the computed likelihood table
    Vec decode(const Vec& syndrome) {
        // Convert syndrome to int vector
        vector<int> syndrome_int;
        for (bool val : syndrome) {
            syndrome_int.push_back(val ? 1 : 0);
        }
        
        // Look up in table
        auto it = likelihood_table.find(syndrome_int);
        if (it != likelihood_table.end() && !it->second.empty()) {
            // Find best logical syndrome
            Real best_score = -1.0;
            vector<int> best_logical;
            
            for (const auto& pair : it->second) {
                if (pair.second > best_score) {
                    best_score = pair.second;
                    best_logical = pair.first;
                }
            }
            
            // Convert back to bool vector
            Vec result;
            for (int val : best_logical) {
                result.push_back(val == 1);
            }
            return result;
        }
        
        // Return zero vector if not found
        return Vec(DL_csc.rows, false);
    }

    // Test logical error rate with GPU acceleration
    Real test_logical_error_rate(Real p, int trials) {
        int n = Dfull_csc.cols;
        int batches = (trials + batch_size - 1) / batch_size;
        
        vector<Real> prior(n, p);
        update_weights(prior);
        
        int success = 0;
        
        for (int batch = 0; batch < batches; batch++) {
            // Process batch
            process_batch();
            
            // Test decoding for this batch
            int batch_trials = min(batch_size, trials - batch * batch_size);
            for (int i = 0; i < batch_trials; i++) {
                // Extract syndrome from batch results
                Vec s;
                for (int j = 0; j < D_csc.rows; j++) {
                    s.push_back(h_syndromes[i * Dfull_csc.rows + j] == 1);
                }
                
                // Extract true logical syndrome
                Vec sl_true;
                for (int j = 0; j < DL_csc.rows; j++) {
                    sl_true.push_back(h_logical_syndromes[i * DL_csc.rows + j] == 1);
                }
                
                // Decode
                Vec sl_decoded = decode(s);
                if (sl_decoded == sl_true) {
                    success++;
                }
            }
        }
        
        return 1.0 - static_cast<Real>(success) / trials;
    }

    void update_weights(const vector<Real>& priors) {
        weights.clear();
        for (size_t i = 0; i < Dfull_csc.cols; i++) {
            weights.push_back(log((1 - priors[i]) / priors[i]));
        }
        
        // Copy updated weights to device
        vector<float> fweights(weights.begin(), weights.end());
        cudaMemcpy(d_weights, fweights.data(), 
                   weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    void print_stats() {
        cout << "CUDA Decoder Statistics:" << endl;
        cout << "Batch size: " << batch_size << endl;
        cout << "Max weight: " << max_weight << endl;
        cout << "Code length: " << Dfull_csc.cols << endl;
        cout << "Likelihood table entries: " << likelihood_table.size() << endl;
        cout << "D matrix: " << D_csc.rows << "x" << D_csc.cols 
             << " (sparsity: " << D_csc.get_sparsity() * 100 << "%)" << endl;
        cout << "DL matrix: " << DL_csc.rows << "x" << DL_csc.cols 
             << " (sparsity: " << DL_csc.get_sparsity() * 100 << "%)" << endl;
    }
};

int main() {
    // Example usage
    // Matrix D = np.load("chk.npy")
    // Matrix DL = np.load("obs.npy")
    // vector<Real> priors = np.load("priors.npy")
    Matrix D = {
        {true, false, true, false, true, false},
        {false, true, true, false, false, true}
    };
    Matrix DL = {
        {true, true, false, true, false, false}
    };

    vector<Real> priors(6, 0.01);
    int max_weight = 3;
    int batch_size = 10000;

    // Create CUDA decoder
    CUDADecoder decoder(D, DL, priors, max_weight, batch_size);
    decoder.print_stats();

    // Test performance
    cout << "\nTesting CUDA decoder performance..." << endl;
    for (Real p : {0.001, 0.003, 0.005, 0.007, 0.01}) {
        Real err_rate = decoder.test_logical_error_rate(p, 10000);
        cout << "Prior error probability p=" << p << ", Logical Error Rate = " << err_rate << endl;
    }

    return 0;
} 