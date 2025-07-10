#include "bp_decoder.hpp"
#include <random>
#include <iostream>
#include <limits>

LikelihoodDecoder::LikelihoodDecoder(const std::vector<std::vector<int>>& D_matrix,
                                     const std::vector<std::vector<int>>& DL_matrix,
                                     const std::vector<double>& error_probs,
                                     int iterations,
                                     const std::string& decoder_name) 
    : D(D_matrix.size(), D_matrix[0].size()), 
      DL(DL_matrix.size(), D_matrix[0].size()),
      num_iterations(iterations),
      name(decoder_name) {
    
    // Build sparse matrix D
    for (int i = 0; i < D_matrix.size(); i++) {
        for (int j = 0; j < D_matrix[i].size(); j++) {
            if (D_matrix[i][j] == 1) {
                D.add_entry(i, j);
            }
        }
    }
    
    // Build sparse matrix DL
    for (int i = 0; i < DL_matrix.size(); i++) {
        for (int j = 0; j < DL_matrix[i].size(); j++) {
            if (DL_matrix[i][j] == 1) {
                DL.add_entry(i, j);
            }
        }
    }
    
    // Calculate weights: ln((1-p)/p)
    weights.resize(error_probs.size());
    for (int i = 0; i < error_probs.size(); i++) {
        weights[i] = std::log((1.0 - error_probs[i]) / error_probs[i]);
    }
    
    // Initialize message storage
    check_to_bit_msgs.resize(D.rows, std::vector<double>(D.cols, 0.0));
    bit_to_check_msgs.resize(D.rows, std::vector<double>(D.cols, 0.0));
    bit_beliefs.resize(D.cols, 0.0);
}

void LikelihoodDecoder::initialize_messages() {
    // Initialize all messages to zero (log-likelihood ratio)
    for (auto& row : check_to_bit_msgs) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    for (auto& row : bit_to_check_msgs) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    
    // Initialize bit beliefs with prior probabilities
    for (int i = 0; i < D.cols; i++) {
        bit_beliefs[i] = weights[i];
    }
}

void LikelihoodDecoder::run_bp_iteration(const std::vector<int>& syndrome) {
    // Update check-to-bit messages
    for (int check = 0; check < D.rows; check++) {
        for (int idx = 0; idx < D.row_indices[check].size(); idx++) {
            int bit = D.row_indices[check][idx];
            
            // Product of tanh of all incoming messages except from current bit
            double product = (syndrome[check] == 1) ? -1.0 : 1.0;
            for (int other_idx = 0; other_idx < D.row_indices[check].size(); other_idx++) {
                if (other_idx != idx) {
                    int other_bit = D.row_indices[check][other_idx];
                    product *= std::tanh(bit_to_check_msgs[check][other_bit] / 2.0);
                }
            }
            
            // Avoid numerical issues
            product = std::max(std::min(product, 0.999999), -0.999999);
            check_to_bit_msgs[check][bit] = 2.0 * std::atanh(product);
        }
    }
    
    // Update bit-to-check messages and beliefs
    for (int bit = 0; bit < D.cols; bit++) {
        // Calculate total incoming message
        double total_llr = weights[bit];
        for (int check : D.col_indices[bit]) {
            total_llr += check_to_bit_msgs[check][bit];
        }
        bit_beliefs[bit] = total_llr;
        
        // Update bit-to-check messages
        for (int check : D.col_indices[bit]) {
            bit_to_check_msgs[check][bit] = total_llr - check_to_bit_msgs[check][bit];
        }
    }
}

std::vector<int> LikelihoodDecoder::sample_error_from_beliefs() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    std::vector<int> error(D.cols, 0);
    for (int i = 0; i < D.cols; i++) {
        // Convert log-likelihood ratio to probability
        double prob_error = 1.0 / (1.0 + std::exp(bit_beliefs[i]));
        if (dis(gen) < prob_error) {
            error[i] = 1;
        }
    }
    return error;
}

std::string LikelihoodDecoder::vector_to_string(const std::vector<int>& vec) {
    std::string result;
    for (int val : vec) {
        result += std::to_string(val);
    }
    return result;
}

std::vector<int> LikelihoodDecoder::string_to_vector(const std::string& str) {
    std::vector<int> result;
    for (char c : str) {
        result.push_back(c - '0');
    }
    return result;
}

std::vector<int> LikelihoodDecoder::compute_logical_syndrome(const std::vector<int>& error) {
    std::vector<int> logical_syndrome(DL.rows, 0);
    for (int row = 0; row < DL.rows; row++) {
        for (int col : DL.row_indices[row]) {
            logical_syndrome[row] ^= error[col];
        }
    }
    return logical_syndrome;
}

std::vector<int> LikelihoodDecoder::hard_decision_from_beliefs() {
    std::vector<int> error(D.cols, 0);
    for (int i = 0; i < D.cols; i++) {
        // Hard decision: if LLR < 0, more likely to be error
        if (bit_beliefs[i] < 0) {
            error[i] = 1;
        }
    }
    return error;
}

bool LikelihoodDecoder::check_syndrome_validity(const std::vector<int>& error, const std::vector<int>& syndrome) {
    for (int check = 0; check < D.rows; check++) {
        int parity = 0;
        for (int bit : D.row_indices[check]) {
            parity ^= error[bit];
        }
        if (parity != syndrome[check]) {
            return false;
        }
    }
    return true;
}

double LikelihoodDecoder::compute_error_likelihood(const std::vector<int>& error) {
    double log_likelihood = 0.0;
    for (int i = 0; i < error.size(); i++) {
        if (error[i] == 1) {
            log_likelihood -= weights[i];
        }
    }
    return log_likelihood;
}

std::vector<int> LikelihoodDecoder::decode(const std::vector<int>& syndrome) {
    logical_syndrome_likelihood.clear();
    
    // Use a set to track unique errors we've already processed
    std::unordered_map<std::string, bool> processed_errors;
    
    // Initialize messages
    initialize_messages();
    
    // Run BP iterations and collect unique valid errors
    for (int iter = 0; iter < num_iterations; iter++) {
        run_bp_iteration(syndrome);
        if (iter < 10){
            continue;
        }
        // Extract hard decision from current beliefs
        std::vector<int> candidate_error = hard_decision_from_beliefs();
        
        // Check if this error satisfies the syndrome
        if (check_syndrome_validity(candidate_error, syndrome)) {
            std::string error_key = vector_to_string(candidate_error);
            
            // Only process each unique error once
            if (processed_errors.find(error_key) == processed_errors.end()) {
                processed_errors[error_key] = true;
                
                // Compute logical syndrome
                std::vector<int> logical_syn = compute_logical_syndrome(candidate_error);
                std::string logical_key = vector_to_string(logical_syn);
                
                // Compute likelihood for this error
                double log_likelihood = compute_error_likelihood(candidate_error);
                
                // Accumulate likelihood for this logical syndrome
                if (logical_syndrome_likelihood.find(logical_key) == logical_syndrome_likelihood.end()) {
                    logical_syndrome_likelihood[logical_key] = log_likelihood;
                } else {
                    // Log-sum-exp trick for numerical stability
                    double max_ll = std::max(logical_syndrome_likelihood[logical_key], log_likelihood);
                    logical_syndrome_likelihood[logical_key] = max_ll + 
                        std::log(std::exp(logical_syndrome_likelihood[logical_key] - max_ll) + 
                                 std::exp(log_likelihood - max_ll));
                }
            }
        }
    }
    
    // Find most likely logical syndrome
    std::string best_syndrome;
    double best_likelihood = -std::numeric_limits<double>::infinity();
    
    for (const auto& pair : logical_syndrome_likelihood) {
        if (pair.second > best_likelihood) {
            best_likelihood = pair.second;
            best_syndrome = pair.first;
        }
    }
    
    // Return the most likely logical syndrome
    if (best_syndrome.empty()) {
        // No valid errors found, return zero syndrome
        return std::vector<int>(DL.rows, 0);
    }
    
    return string_to_vector(best_syndrome);
} 