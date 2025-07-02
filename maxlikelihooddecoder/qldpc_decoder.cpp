#include "qldpc_decoder.h"

std::vector<std::vector<int>> readCSVToIntMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }
    std::vector<std::vector<int>> matrix;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<int> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stoi(cell));
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("无效的整数: " + cell);
            } catch (const std::out_of_range& e) {
                throw std::runtime_error("整数超出范围: " + cell);
            }
        }
        matrix.push_back(row);
    }
    return matrix;
}

// QLDPCDecoder implementation

std::vector<int> QLDPCDecoder::xor_vectors(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] ^ b[i];
    }
    return result;
}

std::vector<int> QLDPCDecoder::multiply_gf2(const std::vector<std::vector<int>>& matrix, const std::vector<int>& vec) {
    std::vector<int> result(matrix.size(), 0);
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < vec.size(); j++) {
            result[i] ^= (matrix[i][j] & vec[j]);
        }
    }
    return result;
}

double QLDPCDecoder::dot_product(const std::vector<double>& a, const std::vector<int>& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

std::vector<int> QLDPCDecoder::hadamard_product(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] * b[i];
    }
    return result;
}

int QLDPCDecoder::hamming_weight(const std::vector<int>& vec) {
    int weight = 0;
    for (int bit : vec) {
        weight += bit;
    }
    return weight;
}

void QLDPCDecoder::generate_error_vectors(int length, int max_weight, int current_pos, std::vector<int>& current_vec, int current_weight, std::vector<std::vector<int>>& result) {
    if (current_pos == length) {
        if (current_weight <= max_weight) {
            result.push_back(current_vec);
        }
        return;
    }
    if (current_weight <= max_weight) {
        current_vec[current_pos] = 0;
        generate_error_vectors(length, max_weight, current_pos + 1, current_vec, current_weight, result);
        if (current_weight < max_weight) {
            current_vec[current_pos] = 1;
            generate_error_vectors(length, max_weight, current_pos + 1, current_vec, current_weight + 1, result);
        }
    }
}

double QLDPCDecoder::calculate_energy_contribution(const std::vector<int>& syndrome, const std::vector<int>& e2) {
    std::vector<int> Be2 = multiply_gf2(B, e2);
    std::vector<int> s_hadamard_Be2 = hadamard_product(syndrome, Be2);
    double energy = -dot_product(w1, Be2) - dot_product(w2, e2) + 2.0 * dot_product(w1, s_hadamard_Be2);
    return energy;
}

QLDPCDecoder::QLDPCDecoder(const std::vector<std::vector<int>>& decoding_matrix, const std::vector<double>& prior_probs, int max_weight, int num_logical_bits) {
    m = decoding_matrix.size();
    n = decoding_matrix[0].size();
    n2 = n - m;
    nL = num_logical_bits;
    B.resize(m, std::vector<int>(n2));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n2; j++) {
            B[i][j] = decoding_matrix[i][m + j];
        }
    }
    update_weights(prior_probs);
    max_hamming_weight = max_weight;
    D_prime = decoding_matrix;
}

void QLDPCDecoder::update_weights(const std::vector<double>& priors) {
    std::vector<double> weights = {};
    for(int i=0; i<priors.size(); i++){
        weights.push_back(std::log((1-priors[i])/priors[i]));
    }
    w1.clear();
    for(int i=0; i<D_prime.size(); i++){
        w1.push_back(weights[i]);
    }
    w2.clear();
    for(int i=D_prime.size(); i<priors.size(); i++){
        w2.push_back(weights[i]);
    }
}

std::vector<int> QLDPCDecoder::decode(const std::vector<int>& syndrome_prime) {
    std::vector<std::vector<int>> e2_candidates;
    std::vector<int> temp_vec(n2, 0);
    generate_error_vectors(n2, max_hamming_weight, 0, temp_vec, 0, e2_candidates);
    double best_log_likelihood = std::numeric_limits<double>::lowest();
    std::vector<int> best_logical_syndrome;
    int num_logical_bits = nL;
    for (int logical_config = 0; logical_config < (1 << num_logical_bits); logical_config++) {
        std::vector<int> current_logical(num_logical_bits);
        for (int i = 0; i < num_logical_bits; i++) {
            current_logical[i] = (logical_config >> i) & 1;
        }
        std::vector<int> full_syndrome = syndrome_prime;
        full_syndrome.insert(full_syndrome.end(), current_logical.begin(), current_logical.end());
        double total_energy = 0.0;
        for (const auto& e2 : e2_candidates) {
            double energy = calculate_energy_contribution(full_syndrome, e2);
            total_energy += std::exp(-energy);
        }
        double total_log_likelihood = -dot_product(w1, full_syndrome) + std::log(total_energy);
        if (total_log_likelihood > best_log_likelihood) {
            best_log_likelihood = total_log_likelihood;
            best_logical_syndrome = current_logical;
        }
    }
    return best_logical_syndrome;
}

bool QLDPCDecoder::verify_decoding(const std::vector<int>& error_logical_vector, const std::vector<int>& logical_syndrome) {
    return error_logical_vector == logical_syndrome;
}

void QLDPCDecoder::print_stats() {
    std::cout << "QLDPC Decoder Configuration:" << std::endl;
    std::cout << "m (syndrome length): " << m << std::endl;
    std::cout << "n2 (free errors): " << n2 << std::endl;
    std::cout << "nL (#logical operators): " << nL << std::endl;
    std::cout << "Max Hamming weight: " << max_hamming_weight << std::endl;
    std::cout << "B matrix size: " << B.size() << "x" << (B.empty() ? 0 : B[0].size()) << std::endl;
}

double QLDPCDecoder::test_logical_error_rate(int num_trials, const std::vector<double>& error_probs, bool verbose) {
    std::random_device rd;
    std::mt19937 gen(rd());
    int total_trials = 0;
    int successful_corrections = 0;
    int logical_errors = 0;
    std::cout << "\n=== Testing Logical Error Rate ===" << std::endl;
    std::cout << "Number of trials: " << num_trials << std::endl;
    std::cout << "Error probabilities: ";
    for (double p : error_probs) {
        std::cout << p << " ";
    }
    std::cout << std::endl;
    update_weights(error_probs);
    for (int trial = 0; trial < num_trials; trial++) {
        std::vector<int> true_error(error_probs.size());
        for (size_t i = 0; i < error_probs.size(); i++) {
            std::uniform_real_distribution<> dis(0.0, 1.0);
            true_error[i] = (dis(gen) < error_probs[i]) ? 1 : 0;
        }
        std::vector<int> true_syndrome_prime = multiply_gf2(D_prime, true_error);
        std::vector<int> true_logical_syndrome(nL, 0);
        for (int i = 0; i < nL; i++) {
            true_logical_syndrome[i] = true_syndrome_prime[true_syndrome_prime.size() - nL + i];
        }
        std::vector<int> true_syndrome(true_syndrome_prime.begin(), true_syndrome_prime.end() - nL);
        std::vector<int> decoded_logical_syndrome = decode(true_syndrome);
        bool decoding_success = (decoded_logical_syndrome == true_logical_syndrome);
        if (decoding_success) {
            successful_corrections++;
        } else {
            logical_errors++;
        }
        total_trials++;
        if (verbose && (trial < 10 || trial % (num_trials / 10) == 0)) {
            std::cout << "Trial " << trial + 1 << ":" << std::endl;
            std::cout << "  True error (weight=" << hamming_weight(true_error) << "): ";
            for (int bit : true_error) std::cout << bit;
            std::cout << std::endl;
            std::cout << "  True logical syndrome: ";
            for (int bit : true_logical_syndrome) std::cout << bit;
            std::cout << std::endl;
            std::cout << "  Decoded logical syndrome: ";
            for (int bit : decoded_logical_syndrome) std::cout << bit;
            std::cout << std::endl;
            std::cout << "  Success: " << (decoding_success ? "YES" : "NO") << std::endl;
            std::cout << std::endl;
        }
    }
    double success_rate = (double)successful_corrections / total_trials;
    double logical_error_rate = (double)logical_errors / total_trials;
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Total trials: " << total_trials << std::endl;
    std::cout << "Successful corrections: " << successful_corrections << std::endl;
    std::cout << "Logical errors: " << logical_errors << std::endl;
    std::cout << "Success rate: " << success_rate << " (" << (success_rate * 100) << "%)" << std::endl;
    std::cout << "Logical error rate: " << logical_error_rate << " (" << (logical_error_rate * 100) << "%)" << std::endl;
    return logical_error_rate;
}

std::vector<int> QLDPCDecoder::generate_random_error(const std::vector<double>& error_probs, std::mt19937& gen) {
    std::vector<int> error(error_probs.size());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < error_probs.size(); i++) {
        error[i] = (dis(gen) < error_probs[i]) ? 1 : 0;
    }
    return error;
}

void QLDPCDecoder::test_different_error_rates() {
    std::cout << "\n=== Testing Different Physical Error Rates ===" << std::endl;
    std::vector<double> physical_error_rates = {0.001,0.003,0.007, 0.01, 0.03, 0.05};
    for (double p_error : physical_error_rates) {
        std::vector<double> uniform_error_probs(n, p_error);
        std::cout << "\nPhysical error rate: " << p_error << std::endl;
        double logical_error_rate = test_logical_error_rate(1000, uniform_error_probs, false);
        std::cout << "Logical error rate: " << logical_error_rate << std::endl;
        std::cout << "Error suppression factor: " << p_error / logical_error_rate << std::endl;
    }
} 