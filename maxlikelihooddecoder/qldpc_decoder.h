#ifndef QLDPC_DECODER_H
#define QLDPC_DECODER_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <bitset>
#include <limits>
#include <random>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

std::vector<std::vector<int>> readCSVToIntMatrix(const std::string& filename);

class QLDPCDecoder {
private:
    std::vector<std::vector<int>> D_prime;  // Decoding matrix [I | B]
    std::vector<std::vector<int>> B;        // B matrix from D' = [I | B]
    std::vector<double> w1, w2;             // Weight vectors
    int n, m, n2, nL;                       // Dimensions
    int max_hamming_weight;                 // Maximum Hamming weight to consider

    std::vector<int> xor_vectors(const std::vector<int>& a, const std::vector<int>& b);
    std::vector<int> multiply_gf2(const std::vector<std::vector<int>>& matrix, const std::vector<int>& vec);
    double dot_product(const std::vector<double>& a, const std::vector<int>& b);
    std::vector<int> hadamard_product(const std::vector<int>& a, const std::vector<int>& b);
    int hamming_weight(const std::vector<int>& vec);
    void generate_error_vectors(int length, int max_weight, int current_pos, std::vector<int>& current_vec, int current_weight, std::vector<std::vector<int>>& result);
    double calculate_energy_contribution(const std::vector<int>& syndrome, const std::vector<int>& e2);

public:
    QLDPCDecoder(const std::vector<std::vector<int>>& decoding_matrix, const std::vector<double>& prior_probs, int max_weight = 10, int num_logical_bits = 0);
    void update_weights(const std::vector<double>& priors);
    std::vector<int> decode(const std::vector<int>& syndrome_prime);
    bool verify_decoding(const std::vector<int>& error_logical_vector, const std::vector<int>& logical_syndrome);
    void print_stats();
    double test_logical_error_rate(int num_trials, const std::vector<double>& error_probs, bool verbose = false);
    std::vector<int> generate_random_error(const std::vector<double>& error_probs, std::mt19937& gen);
    void test_different_error_rates();
};

#endif // QLDPC_DECODER_H 