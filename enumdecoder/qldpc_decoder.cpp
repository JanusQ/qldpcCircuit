#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <map>
#include <bitset>
#include <cassert>
#include <algorithm>

using namespace std;

using Vec = vector<bool>;
using Matrix = vector<Vec>;
using Real = double;

class EnumDecoder {
private:
    Matrix D;
    Matrix DL;
    vector<Real> weights;
    Matrix Dfull;
    int max_weight;
    map<Vec, map<Vec, Real>> likelihoods; // syndrome -> logical syndrome -> likelihood
    map<Vec, Vec> best_logical_syndromes; // syndrome -> best logical syndrome

    // Compute w_j = ln((1 - p) / p)
    Real compute_weight(Real p) {
        return log((1 - p) / p);
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

    // Precompute likelihood table for all error patterns
    void precompute_likelihoods() {
        likelihoods.clear();

        best_logical_syndromes.clear();
        vector<Vec> errors;
        enumerate_errors(weights.size(), max_weight, errors);
        for (const Vec& e : errors) {
            Vec s_full = mod2_matvec(Dfull, e);
 
            Vec s = Vec(s_full.begin(), s_full.begin() + D.size());
            Vec sl = Vec(s_full.begin() + D.size(), s_full.end());
            
            Real prob_weight = 0.0;
            for (size_t j = 0; j < e.size(); ++j) {
                if (e[j]) prob_weight += weights[j];
            }
            // 如果 s sl 不存在，则初始化
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
        // print some example for best_logical_syndromes
        // cout  << "size of likelihoods: " << likelihoods.size() << endl;
        // cout << "best_logical_syndromes: " << endl; 
        // int count = 0;
        // for (auto it = best_logical_syndromes.begin(); it != best_logical_syndromes.end(); ++it) {
        //     const Vec& syndrome = it->first;
        //     const Vec& sl = it->second;
        //     for(size_t i=0; i<syndrome.size(); i++){
        //         cout << syndrome[i] << " ";
        //     }
        //     cout << "| ";
        //     for(size_t i=0; i<sl.size(); i++){
        //         cout << sl[i] << " ";
        //     }
        //     cout << endl;
        //     count++;
        //     if (count > 10) break;
        // }
    }

public:
    // Constructor to initialize the EnumDecoder object
    EnumDecoder(const Matrix& D_matrix, const Matrix& DL_matrix, const vector<Real>& priors,  int maximum_weight){
        Dfull = concat_matrix(D_matrix, DL_matrix);
        D = D_matrix;
        DL = DL_matrix;
        // cout << "size of D: " << D.size() << endl;
        // cout << "size of DL: " << DL.size() << endl;
        // cout << "size of Dfull: " << Dfull.size() << endl;
        max_weight = maximum_weight;
        // update the weights by priors probabilities and precompute likelihoods
        update_weights(priors);
        // cout << "Finished precomputing likelihoods" << endl;
    }

    // update the weights by priors probabilities
    void update_weights(const std::vector<double>& priors) {
        weights.clear();
        for(size_t i=0; i<Dfull[0].size(); i++){
            weights.push_back(std::log((1-priors[i])/priors[i]));
        }
        // Recompute likelihood table when weights change
        precompute_likelihoods();
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
        
        vector<Real> prior(n,p);
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
        cout << "EnumDecoder Statistics:" << endl;
        cout << "Total unique syndromes: " << likelihoods.size() << endl;
        cout << "Max weight: " << max_weight << endl;
        cout << "Code length: " << D[0].size() << endl;
    }

    // // Helper function for matrix-vector multiplication (made public for Python binding)
    // Vec mod2_matvec(const Matrix& mat, const Vec& vec) {
    //     Vec result(mat.size(), false);
    //     for (size_t i = 0; i < mat.size(); ++i) {
    //         bool val = false;
    //         for (size_t j = 0; j < vec.size(); ++j) {
    //             val ^= (mat[i][j] & vec[j]);
    //         }
    //         result[i] = val;
    //     }
    //     return result;
    // }
};

// int main() {
//     // Example: (toy) D and DL for 6-bit code
//     Matrix D = {
//         {true, false, true, false, true, false},
//         {false, true, true, false, false, true}
//     };
//     Matrix DL = {
//         {true, true, false, true, false, false}
//     };

//     vector<Real> priors(6, 0.01); // 假设先验概率都为0.01

//     EnumDecoder decoder(D, DL, priors, 3);

//     for (Real p : {0.001, 0.003, 0.005, 0.007, 0.01, 0.05, 0.1, 0.15}) {
//         Real err_rate = decoder.test_logical_error_rate(p, 1000);
//         cout << "Prior error probability p=" << p << ", Logical Error Rate = " << err_rate << endl;
//     }
//     return 0;
// }
