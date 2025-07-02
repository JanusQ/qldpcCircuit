#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <map>
#include <bitset>
#include <cassert>
#include <algorithm> // 添加这一行

using namespace std;

using Vec = vector<bool>;
using Matrix = vector<Vec>;
using Real = double;

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

// Decoder logic
Vec decode(const Matrix& D, const Matrix& DL, const Vec& syndrome, const vector<Real>& weights, int max_weight) {
    Matrix Dfull = concat_matrix(D, DL);
    map<Vec, map<Vec, Real>> likelihoods; // syndrome' -> logical syndrome -> likelihood
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
        likelihoods[s][sl] += exp(-prob_weight);
    }

    const auto& L = likelihoods[syndrome];
    Vec best_sl;
    Real best_score = -1.0;
    for (const auto& [sl, score] : L) {
        if (score > best_score) {
            best_score = score;
            best_sl = sl;
        }
    }
    return best_sl;
}

// Simulate decoding and estimate logical error rate
Real test_logical_error_rate(const Matrix& D, const Matrix& DL, Real p, int trials, int max_weight) {
    int n = D[0].size();
    int k = DL.size();
    
    vector<Real> weights(n);
    for (int j = 0; j < n; ++j) weights[j] = compute_weight(p);

    Matrix Dfull = concat_matrix(D, DL);

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

        Vec sl_decoded = decode(D, DL, s, weights, max_weight);
        if (sl_decoded == sl_true) ++success;
    }

    return 1.0 - static_cast<Real>(success) / trials;
}

int main() {
    // Example: (toy) D and DL for 6-bit code
    Matrix D = {
        {true, false, true, false, true, false},
        {false, true, true, false, false, true}
    };
    Matrix DL = {
        {true, true, false, true, false, false}
    };

    for (Real p : {0.001, 0.003,0.005,0.007,0.01, 0.05, 0.1, 0.15}) {
        Real err_rate = test_logical_error_rate(D, DL, p, 1000, 3);
        cout << "Prior error probability p=" << p << ", Logical Error Rate = " << err_rate << endl;
    }
    return 0;
}
