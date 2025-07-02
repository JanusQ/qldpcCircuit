#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <map>
#include <bitset>
#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include <numeric>
#include <chrono>
using namespace std;

using Vec = vector<bool>;
using Matrix = vector<Vec>;
using Real = double;

Real compute_weight(Real p) {
    return log((1 - p) / p);
}

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

Matrix concat_matrix(const Matrix& A, const Matrix& B) {
    Matrix result = A;
    result.insert(result.end(), B.begin(), B.end());
    return result;
}

Vec concat_vec(const Vec& a, const Vec& b) {
    Vec result = a;
    result.insert(result.end(), b.begin(), b.end());
    return result;
}

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

__device__ int dot_mod2(const bool* row, const bool* e, int n) {
    bool sum = 0;
    for (int i = 0; i < n; ++i)
        sum ^= (row[i] & e[i]);
    return sum;
}

__global__ void compute_likelihoods_kernel(
    const bool* Dfull,
    int m, int n,
    const bool* errors,
    int num_errors,
    const double* weights,
    double* likelihood_table,
    int num_syndromes,
    int num_sl,
    int syndrome_bits,
    int sl_bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_errors) return;

    const bool* e = errors + idx * n;

    int s_idx = 0, sl_idx = 0;
    for (int i = 0; i < m; ++i) {
        bool bit = dot_mod2(Dfull + i * n, e, n);
        if (i < syndrome_bits)
            s_idx |= (bit << i);
        else
            sl_idx |= (bit << (i - syndrome_bits));
    }

    double weight = 0.0;
    for (int j = 0; j < n; ++j)
        if (e[j]) weight += weights[j];

    atomicAdd(&likelihood_table[s_idx * num_sl + sl_idx], exp(-weight));
}

Vec decode(const Matrix& D, const Matrix& DL, const Vec& syndrome, const vector<Real>& weights, int max_weight) {
    int m = D.size() + DL.size();
    int n = D[0].size();
    int syndrome_bits = D.size();
    int sl_bits = DL.size();
    int num_syndromes = 1 << syndrome_bits;
    int num_sl = 1 << sl_bits;

    vector<Vec> errors;
    enumerate_errors(n, max_weight, errors);
    int num_errors = errors.size();

    vector<bool> flat_Dfull(m * n);
    for (int i = 0; i < D.size(); ++i)
        for (int j = 0; j < n; ++j)
            flat_Dfull[i * n + j] = D[i][j];
    for (int i = 0; i < DL.size(); ++i)
        for (int j = 0; j < n; ++j)
            flat_Dfull[(D.size() + i) * n + j] = DL[i][j];

    vector<bool> flat_errors(num_errors * n);
    for (int i = 0; i < num_errors; ++i)
        for (int j = 0; j < n; ++j)
            flat_errors[i * n + j] = errors[i][j];

    vector<double> likelihood_table(num_syndromes * num_sl, 0.0);

    bool *d_Dfull, *d_errors;
    double *d_weights, *d_likelihood;
    cudaMalloc(&d_Dfull, flat_Dfull.size() * sizeof(bool));
    cudaMalloc(&d_errors, flat_errors.size() * sizeof(bool));
    cudaMalloc(&d_weights, weights.size() * sizeof(double));
    cudaMalloc(&d_likelihood, likelihood_table.size() * sizeof(double));

    cudaMemcpy(d_Dfull, flat_Dfull.data(), flat_Dfull.size() * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_errors, flat_errors.data(), flat_errors.size() * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_likelihood, 0, likelihood_table.size() * sizeof(double));

    int blockSize = 256;
    int gridSize = (num_errors + blockSize - 1) / blockSize;
    compute_likelihoods_kernel<<<gridSize, blockSize>>>(
        d_Dfull, m, n,
        d_errors, num_errors,
        d_weights, d_likelihood,
        num_syndromes, num_sl, syndrome_bits, sl_bits
    );

    cudaMemcpy(likelihood_table.data(), d_likelihood, likelihood_table.size() * sizeof(double), cudaMemcpyDeviceToHost);

    int s_idx = 0;
    for (int i = 0; i < syndrome_bits; ++i)
        s_idx |= (syndrome[i] << i);

    Vec best_sl(sl_bits);
    double best_score = -1.0;
    for (int i = 0; i < num_sl; ++i) {
        double score = likelihood_table[s_idx * num_sl + i];
        if (score > best_score) {
            best_score = score;
            for (int b = 0; b < sl_bits; ++b)
                best_sl[b] = (i >> b) & 1;
        }
    }

    cudaFree(d_Dfull);
    cudaFree(d_errors);
    cudaFree(d_weights);
    cudaFree(d_likelihood);

    return best_sl;
}

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
    Matrix D = {
        {true, false, true, false, true, false},
        {false, true, true, false, false, true}
    };
    Matrix DL = {
        {true, true, false, true, false, false}
    };

    for (Real p : {0.01, 0.05, 0.1, 0.15}) {
        Real err_rate = test_logical_error_rate(D, DL, p, 1000, 3);
        cout << "Prior error probability p=" << p << ", Logical Error Rate = " << err_rate << endl;
    }
    return 0;
}
