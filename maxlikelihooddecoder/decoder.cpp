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
#include <stdexcept> // 用于异常处理

std::vector<std::vector<int>> readCSVToIntMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }
    
    std::vector<std::vector<int>> matrix;
    std::string line;
    
    while (std::getline(file, line)) {
        // 跳过空行
        if (line.empty()) continue;
        
        std::vector<int> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            try {
                // 将字符串转换为整数
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



class QLDPCDecoder {
private:
    std::vector<std::vector<int>> D_prime;  // Decoding matrix [I | B]
    std::vector<std::vector<int>> B;        // B matrix from D' = [I | B]
    std::vector<double> w1, w2;             // Weight vectors
    int n, m, n2, nL;                             // Dimensions
    int max_hamming_weight;                 // Maximum Hamming weight to consider
    
    // XOR operation for binary vectors
    std::vector<int> xor_vectors(const std::vector<int>& a, const std::vector<int>& b) {
        std::vector<int> result(a.size());
        for (size_t i = 0; i < a.size(); i++) {
            result[i] = a[i] ^ b[i];
        }
        return result;
    }
    
    // Matrix-vector multiplication in GF(2)
    std::vector<int> multiply_gf2(const std::vector<std::vector<int>>& matrix, 
                                  const std::vector<int>& vec) {
        std::vector<int> result(matrix.size(), 0);
        for (size_t i = 0; i < matrix.size(); i++) {
            for (size_t j = 0; j < vec.size(); j++) {
                result[i] ^= (matrix[i][j] & vec[j]);
            }
        }
        return result;
    }
    
    // Dot product
    double dot_product(const std::vector<double>& a, const std::vector<int>& b) {
        double result = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    // Element-wise multiplication (Hadamard product)
    std::vector<int> hadamard_product(const std::vector<int>& a, const std::vector<int>& b) {
        std::vector<int> result(a.size());
        for (size_t i = 0; i < a.size(); i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }
    
    // Calculate Hamming weight
    int hamming_weight(const std::vector<int>& vec) {
        int weight = 0;
        for (int bit : vec) {
            weight += bit;
        }
        return weight;
    }
    
    // Generate all binary vectors of length n with Hamming weight <= max_weight
    void generate_error_vectors(int length, int max_weight, int current_pos, 
                               std::vector<int>& current_vec, int current_weight,
                               std::vector<std::vector<int>>& result) {
        if (current_pos == length) {
            if (current_weight <= max_weight) {
                result.push_back(current_vec);
            }
            return;
        }
        
        if (current_weight <= max_weight) {
            // Try bit = 0
            current_vec[current_pos] = 0;
            generate_error_vectors(length, max_weight, current_pos + 1, 
                                 current_vec, current_weight, result);
            
            // Try bit = 1 (if we haven't exceeded max weight)
            if (current_weight < max_weight) {
                current_vec[current_pos] = 1;
                generate_error_vectors(length, max_weight, current_pos + 1, 
                                     current_vec, current_weight + 1, result);
            }
        }
    }
    
    // Calculate the energy contribution for a given syndrome and e2
    double calculate_energy_contribution(const std::vector<int>& syndrome, 
                                       const std::vector<int>& e2) {
        // Calculate B * e2
        std::vector<int> Be2 = multiply_gf2(B, e2);
        
        // Calculate s ⊙ (B * e2) - element-wise multiplication
        std::vector<int> s_hadamard_Be2 = hadamard_product(syndrome, Be2);
        
        // Calculate energy terms
        double energy = -dot_product(w1, Be2) - dot_product(w2, e2) + 
                       2.0 * dot_product(w1, s_hadamard_Be2);
        
        return energy;
    }

public:
    QLDPCDecoder(const std::vector<std::vector<int>>& decoding_matrix,
                 const std::vector<double>& prior_probs,
                 int max_weight = 10, int num_logical_bits = 0) {
        // Split D' = [I | B] into identity and B parts
        m = decoding_matrix.size();
        n = decoding_matrix[0].size();
        n2 = n - m;
        nL = num_logical_bits;
        
        // Extract B matrix (right part of D')
        B.resize(m, std::vector<int>(n2));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n2; j++) {
                B[i][j] = decoding_matrix[i][m + j];
            }
        }
        // update the weights by priors probabilities
        update_weights(prior_probs);
        
        max_hamming_weight = max_weight;
        
        D_prime = decoding_matrix;
    }
    // update the weights by priors probabilities
    void update_weights(const std::vector<double>& priors) {
            std::vector<double> weights = {};  // Weights for s'
        for(int i=0; i<priors.size(); i++){
        weights.push_back(std::log((1-priors[i])/priors[i]));
        }
        std::vector<double> w1 = {};  // Weights for e1
        for(int i=0; i<D_prime.size(); i++){
            w1.push_back(weights[i]);
        }
        std::vector<double> w2 = {};  // Weights for e2
        for(int i=D_prime.size(); i<priors.size(); i++){
            w2.push_back(weights[i]);
        }
    }

    // Main decoding function
    std::vector<int> decode(const std::vector<int>& syndrome_prime) {
        
        // std::cout << "Starting QLDPC decoding with syndrome size: " << syndrome_prime.size() << std::endl;
        
        // Generate all possible e2 vectors with low Hamming weight
        std::vector<std::vector<int>> e2_candidates;
        std::vector<int> temp_vec(n2, 0);
        generate_error_vectors(n2, max_hamming_weight, 0, temp_vec, 0, e2_candidates);
        
        // std::cout << "Generated " << e2_candidates.size() << " candidate e2 vectors" << std::endl;
        
        // For each possible logical syndrome, calculate the likelihood
        double best_log_likelihood = std::numeric_limits<double>::lowest();
        std::vector<int> best_logical_syndrome;
        
        // Try different logical syndromes (this is a simplified approach)
        // In practice, you'd iterate over all possible logical syndromes
        int num_logical_bits = nL;
        
        for (int logical_config = 0; logical_config < (1 << num_logical_bits); logical_config++) {
            std::vector<int> current_logical(num_logical_bits);
            for (int i = 0; i < num_logical_bits; i++) {
                current_logical[i] = (logical_config >> i) & 1;
            }
            
            // Construct full syndrome [s' | s_L]
            std::vector<int> full_syndrome = syndrome_prime;
            full_syndrome.insert(full_syndrome.end(), current_logical.begin(), current_logical.end());
            
            // Calculate the partition function for this syndrome
            double log_partition = std::numeric_limits<double>::lowest();
            double total_energy = 0.0;
            for (const auto& e2 : e2_candidates) {
                double energy = calculate_energy_contribution(full_syndrome, e2);
                total_energy += std::exp(-energy);
            }
            
            // Calculate total log-likelihood for this logical syndrome
            double total_log_likelihood = -dot_product(w1, full_syndrome) + std::log(total_energy);
            
            if (total_log_likelihood > best_log_likelihood) {
                best_log_likelihood = total_log_likelihood;
                best_logical_syndrome = current_logical;
            }
        }
        
        
        // std::cout << "Best log-likelihood: " << best_log_likelihood << std::endl;
        return best_logical_syndrome;
    }
    
    // Verify the decoding result
    bool verify_decoding(const std::vector<int>& error_logical_vector, 
                        const std::vector<int>& logical_syndrome) {
        return error_logical_vector == logical_syndrome;
    }
    
    // Print decoder statistics
    void print_stats() {
        std::cout << "QLDPC Decoder Configuration:" << std::endl;
        std::cout << "m (syndrome length): " << m << std::endl;
        std::cout << "n2 (free errors): " << n2 << std::endl;
        std::cout << "nL (#logical operators): " << nL << std::endl;
        std::cout << "Max Hamming weight: " << max_hamming_weight << std::endl;
        std::cout << "B matrix size: " << B.size() << "x" << (B.empty() ? 0 : B[0].size()) << std::endl;
    }

    // Test function to evaluate logical error rate
    double test_logical_error_rate(int num_trials, const std::vector<double>& error_probs,
                                  bool verbose = false) {
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
            // Generate random error vector according to prior probabilities
            std::vector<int> true_error(error_probs.size());
            for (size_t i = 0; i < error_probs.size(); i++) {
                std::uniform_real_distribution<> dis(0.0, 1.0);
                true_error[i] = (dis(gen) < error_probs[i]) ? 1 : 0;
            }

            // Calculate true syndromes using the full decoding matrix
            std::vector<int> true_syndrome_prime = multiply_gf2(
                D_prime, true_error
            );
            std::vector<int> true_logical_syndrome(nL, 0);  // Assume all logical qubits are 0
            // the last nL bits of true_syndrome_prime are the logical syndrome
            for (int i = 0; i < nL; i++) {
                true_logical_syndrome[i] = true_syndrome_prime[true_syndrome_prime.size() - nL + i];
            }
            std::vector<int> true_syndrome(true_syndrome_prime.begin(), true_syndrome_prime.end() - nL);
            // Perform decoding
            std::vector<int> decoded_logical_syndrome = decode(true_syndrome);

            // Check if decoding was successful
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

    // Generate error vectors for statistical testing
    std::vector<int> generate_random_error(const std::vector<double>& error_probs,
                                         std::mt19937& gen) {
        std::vector<int> error(error_probs.size());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (size_t i = 0; i < error_probs.size(); i++) {
            error[i] = (dis(gen) < error_probs[i]) ? 1 : 0;
        }
        return error;
    }

    // Advanced testing with different error models
    void test_different_error_rates() {
        std::cout << "\n=== Testing Different Physical Error Rates ===" << std::endl;
        std::vector<double> physical_error_rates = {0.001,0.003,0.007, 0.01, 0.03, 0.05};

        for (double p_error : physical_error_rates) {
            std::vector<double> uniform_error_probs(n, p_error);  // All qubits have same error rate

            std::cout << "\nPhysical error rate: " << p_error << std::endl;
            double logical_error_rate = test_logical_error_rate(1000, uniform_error_probs, false);

            std::cout << "Logical error rate: " << logical_error_rate << std::endl;
            std::cout << "Error suppression factor: " << p_error / logical_error_rate << std::endl;
        }
    }
};

// // Example usage and testing
// int main() {
//     // Example: Small QLDPC code for demonstration
//     // D' = [I | B] where I is 3x3 identity and B is 3x2
//     std::vector<std::vector<int>> D_prime = {
//         {1, 0, 0, 1, 1, 0, 1},
//         {0, 1, 0, 1, 0, 1, 0},
//         {0, 0, 1, 0, 1, 0, 1},
//         {0, 0, 0, 1, 1, 1, 0} // logical check equaiton
//     };
    
//     std::vector<double> priors(D_prime[0].size(), 0.01);  // Prior probabilities for each qubit

//     QLDPCDecoder decoder(D_prime, priors, 2, 1);  // Max Hamming weight = 2
//     decoder.print_stats();
    
//     // Single decoding example
//     std::cout << "\n=== Single Decoding Example ===" << std::endl;
//     std::vector<int> syndrome_prime = {0, 0, 1};
//     std::vector<int> logical_syndrome(1, 0);  // Assume 1 logical qubit
    
//     std::cout << "Decoding syndrome: ";
//     for (int bit : syndrome_prime) {
//         std::cout << bit << " ";
//     }
//     std::cout << std::endl;
    
//     // Perform decoding
//     std::vector<int> decoded_logical_error = decoder.decode(syndrome_prime);
    
//     std::cout << "Decoded error vector: ";
//     for (int bit : decoded_logical_error) {
//         std::cout << bit << " ";
//     }
//     std::cout << std::endl;
    
//     std::cout << "Logical syndrome: ";
//     for (int bit : logical_syndrome) {
//         std::cout << bit << " ";
//     }
//     std::cout << std::endl;
    
//     // Verify the result
//     if (decoder.verify_decoding(decoded_logical_error, logical_syndrome)) {
//         std::cout << "✓ Decoding verification successful!" << std::endl;
//     } else {
//         std::cout << "✗ Decoding verification failed!" << std::endl;
//     }
    
//     // Test different error rates
//     decoder.test_different_error_rates();
    
//     return 0;
// }
