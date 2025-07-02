#include "belief_propagation.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace qldpc_bp;

// Function to print a matrix
void print_matrix(const std::vector<std::vector<int>>& matrix, const std::string& name) {
    std::cout << name << ":" << std::endl;
    for (const auto& row : matrix) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Function to print a vector
void print_vector(const std::vector<int>& vec, const std::string& name) {
    std::cout << name << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Function to print marginal probabilities
void print_marginals(const std::vector<Message>& marginals) {
    std::cout << "Logical syndrome marginal probabilities:" << std::endl;
    for (size_t i = 0; i < marginals.size(); ++i) {
        std::cout << "s_L[" << i << "]: P(0) = " << std::fixed << std::setprecision(6) 
                  << marginals[i].prob_0 << ", P(1) = " << marginals[i].prob_1 << std::endl;
    }
    std::cout << std::endl;
}

// Create a simple repetition code example
std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> create_repetition_code_example() {
    // Simple 5-bit repetition code
    // D matrix (parity check matrix): enforces that all bits are equal
    std::vector<std::vector<int>> D = {
        {1, 1, 0, 0, 0},  // e_0 + e_1 = s'_0
        {0, 1, 1, 0, 0},  // e_1 + e_2 = s'_1
        {0, 0, 1, 1, 0},  // e_2 + e_3 = s'_2
        {0, 0, 0, 1, 1}   // e_3 + e_4 = s'_3
    };
    
    // D_L matrix (logical check matrix): logical syndrome is the parity of all bits
    std::vector<std::vector<int>> D_L = {
        {1, 1, 1, 1, 1}   // s_L = e_0 + e_1 + e_2 + e_3 + e_4
    };
    
    return {D, D_L};
}

// Create a more complex surface code-like example
std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> create_surface_code_example() {
    // 3x3 grid surface code-like structure (simplified)
    // 9 physical qubits arranged in a 3x3 grid
    std::vector<std::vector<int>> D = {
        {1, 1, 0, 1, 0, 0, 0, 0, 0},  // Top-left stabilizer
        {0, 1, 1, 0, 1, 0, 0, 0, 0},  // Top-right stabilizer
        {0, 0, 0, 1, 1, 0, 1, 0, 0},  // Bottom-left stabilizer
        {0, 0, 0, 0, 1, 1, 0, 1, 0},  // Bottom-right stabilizer
        {1, 0, 0, 0, 0, 0, 1, 1, 0},  // Left column stabilizer
        {0, 0, 1, 0, 0, 0, 0, 1, 1}   // Right column stabilizer
    };
    
    // Two logical operators (simplified)
    std::vector<std::vector<int>> D_L = {
        {1, 0, 1, 0, 0, 0, 1, 0, 1},  // First logical operator
        {1, 1, 1, 0, 0, 0, 0, 0, 0}   // Second logical operator
    };
    
    return {D, D_L};
}

// Function to simulate errors and syndrome
std::vector<int> simulate_syndrome(const std::vector<std::vector<int>>& D, 
                                  const std::vector<int>& error_pattern) {
    std::vector<int> syndrome(D.size(), 0);
    
    for (size_t i = 0; i < D.size(); ++i) {
        int parity = 0;
        for (size_t j = 0; j < D[i].size(); ++j) {
            parity ^= (D[i][j] * error_pattern[j]);
        }
        syndrome[i] = parity;
    }
    
    return syndrome;
}

// Function to compute true logical syndrome
std::vector<int> compute_true_logical_syndrome(const std::vector<std::vector<int>>& D_L,
                                              const std::vector<int>& error_pattern) {
    std::vector<int> logical_syndrome(D_L.size(), 0);
    
    for (size_t i = 0; i < D_L.size(); ++i) {
        int parity = 0;
        for (size_t j = 0; j < D_L[i].size(); ++j) {
            parity ^= (D_L[i][j] * error_pattern[j]);
        }
        logical_syndrome[i] = parity;
    }
    
    return logical_syndrome;
}

int main() {
    std::cout << "=== Belief Propagation QLDPC Decoder Example ===" << std::endl << std::endl;
    
    // Example 1: Simple repetition code
    std::cout << "--- Example 1: Repetition Code ---" << std::endl;
    auto [D1, D_L1] = create_repetition_code_example();
    
    print_matrix(D1, "D matrix (syndrome constraints)");
    print_matrix(D_L1, "D_L matrix (logical constraints)");
    
    // Set up error probabilities (low error rate)
    std::vector<double> error_probs1(5, 0.1);  // 10% error probability for each qubit
    
    // Create decoder
    BeliefPropagationDecoder decoder1(D1, D_L1, error_probs1);
    
    // Simulate an error pattern
    std::vector<int> error_pattern1 = {1, 0, 0, 0, 1};  // Errors on qubits 0 and 4
    std::vector<int> observed_syndrome1 = simulate_syndrome(D1, error_pattern1);
    std::vector<int> true_logical_syndrome1 = compute_true_logical_syndrome(D_L1, error_pattern1);
    
    print_vector(error_pattern1, "True error pattern");
    print_vector(observed_syndrome1, "Observed syndrome");
    print_vector(true_logical_syndrome1, "True logical syndrome");
    
    // Decode
    decoder1.set_observed_syndrome(observed_syndrome1);
    std::vector<int> decoded_logical_syndrome1 = decoder1.decode(50, 1e-6);
    
    print_vector(decoded_logical_syndrome1, "Decoded logical syndrome");
    
    // Print marginals
    auto marginals1 = decoder1.compute_logical_marginals();
    print_marginals(marginals1);
    
    // Check if decoding was successful
    bool success1 = (decoded_logical_syndrome1 == true_logical_syndrome1);
    std::cout << "Decoding " << (success1 ? "SUCCESSFUL" : "FAILED") << std::endl << std::endl;
    
    // Example 2: Surface code-like structure
    std::cout << "--- Example 2: Surface Code-like Structure ---" << std::endl;
    auto [D2, D_L2] = create_surface_code_example();
    
    print_matrix(D2, "D matrix (syndrome constraints)");
    print_matrix(D_L2, "D_L matrix (logical constraints)");
    
    // Set up error probabilities
    std::vector<double> error_probs2(9, 0.05);  // 5% error probability for each qubit
    
    // Create decoder
    BeliefPropagationDecoder decoder2(D2, D_L2, error_probs2);
    
    // Simulate an error pattern
    std::vector<int> error_pattern2 = {0, 1, 0, 1, 0, 0, 0, 1, 0};  // Errors on qubits 1, 3, 7
    std::vector<int> observed_syndrome2 = simulate_syndrome(D2, error_pattern2);
    std::vector<int> true_logical_syndrome2 = compute_true_logical_syndrome(D_L2, error_pattern2);
    
    print_vector(error_pattern2, "True error pattern");
    print_vector(observed_syndrome2, "Observed syndrome");
    print_vector(true_logical_syndrome2, "True logical syndrome");
    
    // Decode
    decoder2.set_observed_syndrome(observed_syndrome2);
    std::vector<int> decoded_logical_syndrome2 = decoder2.decode(50, 1e-6);
    
    print_vector(decoded_logical_syndrome2, "Decoded logical syndrome");
    
    // Print marginals
    auto marginals2 = decoder2.compute_logical_marginals();
    print_marginals(marginals2);
    
    // Check if decoding was successful
    bool success2 = (decoded_logical_syndrome2 == true_logical_syndrome2);
    std::cout << "Decoding " << (success2 ? "SUCCESSFUL" : "FAILED") << std::endl << std::endl;
    
    // Example 3: Random error simulation
    std::cout << "--- Example 3: Random Error Simulation (Repetition Code) ---" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution error_dist(0.15);  // 15% error probability
    
    int num_trials = 10;
    int successful_decodings = 0;
    
    for (int trial = 0; trial < num_trials; ++trial) {
        // Generate random error pattern
        std::vector<int> random_error(5);
        for (int i = 0; i < 5; ++i) {
            random_error[i] = error_dist(gen) ? 1 : 0;
        }
        
        auto random_syndrome = simulate_syndrome(D1, random_error);
        auto true_logical = compute_true_logical_syndrome(D_L1, random_error);
        
        decoder1.set_observed_syndrome(random_syndrome);
        auto decoded_logical = decoder1.decode(30, 1e-5);
        
        bool trial_success = (decoded_logical == true_logical);
        successful_decodings += trial_success ? 1 : 0;
        
        std::cout << "Trial " << std::setw(2) << trial + 1 << ": ";
        print_vector(random_error, "Error");
        std::cout << "         ";
        print_vector(true_logical, "True logical");
        std::cout << "         ";
        print_vector(decoded_logical, "Decoded");
        std::cout << "         " << (trial_success ? "SUCCESS" : "FAIL") << std::endl;
    }
    
    std::cout << std::endl << "Success rate: " << successful_decodings << "/" << num_trials 
              << " (" << std::fixed << std::setprecision(1) 
              << (100.0 * successful_decodings / num_trials) << "%)" << std::endl;
    
    return 0;
} 