#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <map>
#include <bitset>
#include <cassert>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

using Vec = vector<bool>;
using Matrix = vector<Vec>;
using Real = double;

// External CUDA function declaration
extern "C" {
    void cuda_compute_syndromes(int* h_errors, float* h_weights, int* h_D_matrix, int* h_DL_matrix,
                               int* h_syndromes, int* h_logical_syndromes, float* h_error_weights,
                               int n, int d_rows, int dl_rows, int total_errors);
}

class EnumDecoderCUDA {
private:
    Matrix D;
    Matrix DL;
    vector<Real> weights;
    Matrix Dfull;
    int max_weight;
    map<Vec, map<Vec, Real>> likelihoods; // syndrome -> logical syndrome -> likelihood
    map<Vec, Vec> best_logical_syndromes; // syndrome -> best logical syndrome

    // Convert bool matrix to int matrix for CUDA
    vector<int> matrix_to_int(const Matrix& mat) {
        vector<int> result;
        for (const auto& row : mat) {
            for (bool val : row) {
                result.push_back(val ? 1 : 0);
            }
        }
        return result;
    }

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

    // Syndrome = D * e (mod 2)
    Vec mod2_matvec(const Matrix& mat, const Vec& vec) {
        Vec result(mat.size(), false);
        for (size_t i = 0; i < mat.size(); ++i) {
            bool val = false;
            for (size_t j = 0; j < vec.size(); ++j) {
                val ^= (mat[i][j] & vec[j]);
            }
            result[i] = val;
        }
        return result;
    }

    // Concatenate two matrices vertically
    Matrix concat_matrix(const Matrix& A, const Matrix& B) {
        Matrix result = A;
        result.insert(result.end(), B.begin(), B.end());
        return result;
    }

    // Concatenate two vectors
    Vec concat_vec(const Vec& a, const Vec& b) {
        Vec result = a;
        result.insert(result.end(), b.begin(), b.end());
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

    // Precompute likelihood table using CUDA for syndrome computation
    void precompute_likelihoods_cuda() {
        likelihoods.clear();
        best_logical_syndromes.clear();
        
        vector<Vec> errors;
        enumerate_errors(weights.size(), max_weight, errors);
        
        int n = weights.size();
        int d_rows = D.size();
        int dl_rows = DL.size();
        int total_errors = errors.size();
        
        cout << "Enumerated " << total_errors << " error patterns, using CUDA for syndrome computation" << endl;
        
        // Convert data to int format for CUDA
        vector<int> D_int = matrix_to_int(D);
        vector<int> DL_int = matrix_to_int(DL);
        vector<float> weights_float(weights.begin(), weights.end());
        
        // Convert errors to int format
        vector<int> h_errors;
        for (const Vec& e : errors) {
            vector<int> e_int = vec_to_int(e);
            h_errors.insert(h_errors.end(), e_int.begin(), e_int.end());
        }
        
        // Allocate host memory for results
        vector<int> h_syndromes(total_errors * d_rows);
        vector<int> h_logical_syndromes(total_errors * dl_rows);
        vector<float> h_error_weights(total_errors);
        
        // Call CUDA function for syndrome computation
        cuda_compute_syndromes(h_errors.data(), weights_float.data(), D_int.data(), DL_int.data(),
                              h_syndromes.data(), h_logical_syndromes.data(), h_error_weights.data(),
                              n, d_rows, dl_rows, total_errors);
        
        // Process results
        for (int i = 0; i < total_errors; ++i) {
            // Extract syndrome and logical syndrome
            Vec s, sl;
            for (int j = 0; j < d_rows; ++j) {
                s.push_back(h_syndromes[i * d_rows + j] == 1);
            }
            for (int j = 0; j < dl_rows; ++j) {
                sl.push_back(h_logical_syndromes[i * dl_rows + j] == 1);
            }
            
            Real prob_weight = h_error_weights[i];
            
            // Update likelihood table
            if (likelihoods.find(s) == likelihoods.end()) {
                likelihoods[s] = map<Vec, Real>();
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
    // Constructor to initialize the EnumDecoderCUDA object
    EnumDecoderCUDA(const Matrix& D_matrix, const Matrix& DL_matrix, const vector<Real>& priors, int maximum_weight) {
        Dfull = concat_matrix(D_matrix, DL_matrix);
        D = D_matrix;
        DL = DL_matrix;
        max_weight = maximum_weight;
        
        // Initialize CUDA
        cudaSetDevice(0);
        
        // Update the weights by priors probabilities and precompute likelihoods
        update_weights(priors);
        cout << "Finished CUDA-accelerated likelihood computation" << endl;
    }

    // Update the weights by priors probabilities
    void update_weights(const std::vector<double>& priors) {
        weights.clear();
        for (size_t i = 0; i < Dfull[0].size(); i++) {
            weights.push_back(std::log((1 - priors[i]) / priors[i]));
        }
        // Recompute likelihood table when weights change
        precompute_likelihoods_cuda();
    }

    // Decoder logic - now just performs O(1) lookup
    Vec decode(const Vec& syndrome) {
        auto it = best_logical_syndromes.find(syndrome);
        if (it != best_logical_syndromes.end()) {
            return it->second;
        }
        // Return empty vector if syndrome not found (shouldn't happen with proper enumeration)
        return Vec(DL.size(), false);
    }

    // Simulate decoding and estimate logical error rate
    Real test_logical_error_rate(Real p, int trials) {
        int n = D[0].size();
        
        vector<Real> prior(n, p);
        update_weights(prior);

        random_device rd;
        mt19937 gen(rd());
        bernoulli_distribution d(p);

        int success = 0;

        for (int t = 0; t < trials; ++t) {
            Vec e(n);
            for (int j = 0; j < n; ++j) e[j] = d(gen);

            Vec s_full = mod2_matvec(Dfull, e);
            Vec s = Vec(s_full.begin(), s_full.begin() + D.size());
            Vec sl_true = Vec(s_full.begin() + D.size(), s_full.end());

            Vec sl_decoded = decode(s);
            if (sl_decoded == sl_true) ++success;
        }

        return 1.0 - static_cast<Real>(success) / trials;
    }

    // Get statistics
    void print_stats() {
        cout << "CUDA EnumDecoder Statistics:" << endl;
        cout << "Total unique syndromes: " << likelihoods.size() << endl;
        cout << "Max weight: " << max_weight << endl;
        cout << "Code length: " << D[0].size() << endl;
    }
};

int main() {
    // Example: (toy) D and DL for 6-bit code
    Matrix D = {
        {true, false, true, false, true, false},
        {false, true, true, false, false, true}
    };
    Matrix DL = {
        {true, true, false, true, false, false}
    };

    vector<Real> priors(6, 0.01);

    EnumDecoderCUDA decoder(D, DL, priors, 3);
    decoder.print_stats();

    for (Real p : {0.001, 0.003, 0.005, 0.007, 0.01, 0.05, 0.1, 0.15}) {
        Real err_rate = decoder.test_logical_error_rate(p, 1000);
        cout << "Prior error probability p=" << p << ", Logical Error Rate = " << err_rate << endl;
    }
    return 0;
} 