#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <map>
#include <bitset>
#include <cassert>
#include <algorithm>
#include <cuda_runtime.h>
#include <unordered_map>
#include "csc_matrix.h"
// #include "enum_cuda.cu"
using namespace std;

using Vec = vector<bool>;
using Matrix = vector<Vec>;
using Real = double;

// Hash function for vector<bool>
struct VecHasher {
    std::size_t operator()(const Vec& vec) const {
        std::size_t seed = vec.size();
        for (size_t i = 0; i < vec.size(); ++i) {
            seed ^= vec[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// External CUDA function declarations
extern "C" {
    void cuda_csc_multiply_mod2(int* h_values, int* h_row_indices, int* h_col_ptr,
                               int* h_vec, int* h_result, int rows, int cols);
    
    void cuda_batch_csc_multiply_mod2(int* h_values, int* h_row_indices, int* h_col_ptr,
                                     int* h_errors, int* h_syndromes, int* h_logical_syndromes,
                                     int rows, int cols, int dl_rows, int total_errors);
}

class EnumDecoderCSC {
private:
    CSCMatrix D_csc;
    CSCMatrix DL_csc;
    CSCMatrix Dfull_csc;
    vector<Real> weights;
    int max_weight;
    unordered_map<Vec, unordered_map<Vec, Real, VecHasher>, VecHasher> likelihoods;
    unordered_map<Vec, Vec, VecHasher> best_logical_syndromes;

    // Convert bool vector to int vector
    vector<int> vec_to_int(const Vec& vec) {
        vector<int> result;
        for (bool val : vec) {
            result.push_back(val ? 1 : 0);
        }
        return result;
    }

    // Convert int vector to bool vector
    Vec int_to_vec(const vector<int>& int_vec) {
        Vec result;
        for (int val : int_vec) {
            result.push_back(val == 1);
        }
        return result;
    }

    // Concatenate two CSC matrices vertically
    CSCMatrix concat_csc_matrices(const CSCMatrix& A, const CSCMatrix& B) {
        CSCMatrix result;
        result.rows = A.rows + B.rows;
        result.cols = A.cols;
        
        // Combine values and row indices
        result.values = A.values;
        result.row_indices = A.row_indices;
        result.col_ptr = A.col_ptr;
        
        // Add B matrix entries with adjusted row indices
        for (int j = 0; j < B.cols; j++) {
            for (int k = B.col_ptr[j]; k < B.col_ptr[j + 1]; k++) {
                result.values.push_back(B.values[k]);
                result.row_indices.push_back(B.row_indices[k] + A.rows);
            }
        }
        
        // Update column pointers
        for (int j = 1; j <= result.cols; j++) {
            result.col_ptr[j] = A.col_ptr[j] + (B.col_ptr[j] - B.col_ptr[j - 1]);
        }
        
        return result;
    }

    // Hamming weight of a vector
    int hamming_weight(const Vec& vec) {
        int sum = 0;
        for (bool v : vec) sum += v;
        return sum;
    }

    // Enumerate all low-weight error patterns up to a certain weight
    void enumerate_errors(int n, int max_weight, vector<Vec>& errors) {
        for (int w = 0; w <= max_weight; ++w) {
            vector<bool> mask(n, false);
            fill(mask.begin(), mask.begin() + w, true);
            do {
                Vec e(n, false);
                for (int i = 0; i < n; ++i) {
                    if (mask[i]) e[i] = true;
                }
                errors.push_back(e);
            } while (prev_permutation(mask.begin(), mask.end()));
        }
    }

    // Precompute likelihood table using CSC format and CUDA
    void precompute_likelihoods_csc() {
        likelihoods.clear();
        best_logical_syndromes.clear();
        
        vector<Vec> errors;
        enumerate_errors(weights.size(), max_weight, errors);
        
        int n = weights.size();
        int total_errors = errors.size();
        
        cout << "Enumerated " << total_errors << " error patterns using CSC format" << endl;
        cout << "D matrix sparsity: " << D_csc.get_sparsity() * 100 << "%" << endl;
        cout << "DL matrix sparsity: " << DL_csc.get_sparsity() * 100 << "%" << endl;
        
        // Convert errors to int format
        vector<int> h_errors;
        for (const Vec& e : errors) {
            vector<int> e_int = vec_to_int(e);
            h_errors.insert(h_errors.end(), e_int.begin(), e_int.end());
        }
        
        // Allocate host memory for results
        vector<int> h_syndromes(total_errors * D_csc.rows);
        vector<int> h_logical_syndromes(total_errors * DL_csc.rows);
        vector<float> h_error_weights(total_errors);
        
        // Call CUDA function for batch CSC matrix-vector multiplication
        cuda_batch_csc_multiply_mod2(Dfull_csc.values.data(), Dfull_csc.row_indices.data(), 
                                    Dfull_csc.col_ptr.data(), h_errors.data(),
                                    h_syndromes.data(), h_logical_syndromes.data(),
                                    Dfull_csc.rows, Dfull_csc.cols, DL_csc.rows, total_errors);
        
        // Process results
        for (int i = 0; i < total_errors; ++i) {
            // Extract syndrome and logical syndrome
            Vec s, sl;
            for (int j = 0; j < D_csc.rows; ++j) {
                s.push_back(h_syndromes[i * D_csc.rows + j] == 1);
            }
            for (int j = 0; j < DL_csc.rows; ++j) {
                sl.push_back(h_logical_syndromes[i * DL_csc.rows + j] == 1);
            }
            
            // Compute likelihood weight
            Real prob_weight = 0.0;
            for (int j = 0; j < n; ++j) {
                if (h_errors[i * n + j] == 1) {
                    prob_weight += weights[j];
                }
            }
            
            // Update likelihood table
            if (likelihoods.find(s) == likelihoods.end()) {
                likelihoods[s] = unordered_map<Vec, Real, VecHasher>();
            }
            if (likelihoods[s].find(sl) == likelihoods[s].end()) {
                likelihoods[s][sl] = 0.0;
            }
            likelihoods[s][sl] += exp(-prob_weight);
        }
        
        // Find best logical syndrome for each syndrome
        for (auto it = likelihoods.begin(); it != likelihoods.end(); ++it) {
            const Vec& syndrome = it->first;
            const auto& logical_map = it->second;
            
            Vec best_sl;
            Real best_score = -1.0;
            for (auto it2 = logical_map.begin(); it2 != logical_map.end(); ++it2) {
                const Vec& sl = it2->first;
                Real score = it2->second;
                if (score > best_score) {
                    best_score = score;
                    best_sl = sl;
                }
            }
            best_logical_syndromes[syndrome] = best_sl;
        }
        
        cout << "Found " << likelihoods.size() << " unique syndromes" << endl;
    }

public:
    // Constructor with CSC matrices
    EnumDecoderCSC(const CSCMatrix& D_matrix, const CSCMatrix& DL_matrix, 
                   const vector<Real>& priors, int maximum_weight) {
        D_csc = D_matrix;
        DL_csc = DL_matrix;
        Dfull_csc = concat_csc_matrices(D_matrix, DL_matrix);
        max_weight = maximum_weight;
        
        // Initialize CUDA
        cudaSetDevice(0);
        
        // Update weights and precompute likelihoods
        update_weights(priors);
        cout << "Finished CSC-optimized likelihood computation" << endl;
    }
    
    // Constructor with dense matrices (converts to CSC)
    EnumDecoderCSC(const Matrix& D_matrix, const Matrix& DL_matrix, 
                   const vector<Real>& priors, int maximum_weight) {
        D_csc.from_dense(D_matrix);
        DL_csc.from_dense(DL_matrix);
        
        // Create Dfull matrix
        Matrix Dfull_dense;
        Dfull_dense = D_matrix;
        Dfull_dense.insert(Dfull_dense.end(), DL_matrix.begin(), DL_matrix.end());
        Dfull_csc.from_dense(Dfull_dense);
        
        max_weight = maximum_weight;
        
        // Initialize CUDA
        cudaSetDevice(0);
        
        // Update weights and precompute likelihoods
        update_weights(priors);
        cout << "Finished CSC-optimized likelihood computation" << endl;
    }

    // Update weights and recompute likelihoods
    void update_weights(const vector<Real>& priors) {
        weights.clear();
        for (size_t i = 0; i < Dfull_csc.cols; i++) {
            weights.push_back(log((1 - priors[i]) / priors[i]));
        }
        precompute_likelihoods_csc();
    }

    // Decode function - O(1) lookup
    Vec decode(const Vec& syndrome) {
        auto it = best_logical_syndromes.find(syndrome);
        if (it != best_logical_syndromes.end()) {
            return it->second;
        }
        return Vec(DL_csc.rows, false);
    }

    // Simulate decoding and estimate logical error rate
    Real test_logical_error_rate(Real p, int trials) {
        int n = Dfull_csc.cols;
        
        vector<Real> prior(n, p);
        update_weights(prior);

        random_device rd;
        mt19937 gen(rd());
        bernoulli_distribution d(p);

        int success = 0;

        for (int t = 0; t < trials; ++t) {
            Vec e(n);
            for (int j = 0; j < n; ++j) e[j] = d(gen);

            // Use CSC matrix-vector multiplication
            Vec s_full = Dfull_csc.multiply_mod2(e);
            Vec s = Vec(s_full.begin(), s_full.begin() + D_csc.rows);
            Vec sl_true = Vec(s_full.begin() + D_csc.rows, s_full.end());

            Vec sl_decoded = decode(s);
            if (sl_decoded == sl_true) ++success;
        }

        return 1.0 - static_cast<Real>(success) / trials;
    }

    // Get statistics
    void print_stats() {
        cout << "CSC EnumDecoder Statistics:" << endl;
        cout << "Total unique syndromes: " << likelihoods.size() << endl;
        cout << "Max weight: " << max_weight << endl;
        cout << "Code length: " << Dfull_csc.cols << endl;
        cout << "D matrix: " << D_csc.rows << "x" << D_csc.cols 
             << " (sparsity: " << D_csc.get_sparsity() * 100 << "%)" << endl;
        cout << "DL matrix: " << DL_csc.rows << "x" << DL_csc.cols 
             << " (sparsity: " << DL_csc.get_sparsity() * 100 << "%)" << endl;
    }
};

int main() {
    // Example: (toy) D and DL for 6-bit code
    Matrix D = {
        {true, false, true, false, true, false},
        {false, true, true, false, false, true},
        {true, false, true, false, true, false},
        {false, true, true, false, false, true}
    };
    Matrix DL = {
        {true, true, false, true, false, false}
    };

    vector<Real> priors(6, 0.01);

    // Create CSC-optimized decoder
    EnumDecoderCSC decoder(D, DL, priors, 3);
    decoder.print_stats();

    for (Real p : {0.001, 0.003, 0.005, 0.007, 0.01, 0.05, 0.1, 0.15}) {
        Real err_rate = decoder.test_logical_error_rate(p, 1000);
        cout << "Prior error probability p=" << p << ", Logical Error Rate = " << err_rate << endl;
    }
    return 0;
} 