#ifndef MARGINAL_BP_DECODER_HPP
#define MARGINAL_BP_DECODER_HPP

#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <memory>
#include <cstddef>  // for size_t

// Sparse matrix representation for QLDPC codes
struct SparseMatrix {
    int rows;
    int cols;
    std::vector<std::vector<int>> row_indices;  // For each row, indices of non-zero columns
    std::vector<std::vector<int>> col_indices;  // For each column, indices of non-zero rows
    
    SparseMatrix(int r, int c) : rows(r), cols(c), row_indices(r), col_indices(c) {}
    
    void add_entry(int row, int col) {
        row_indices[row].push_back(col);
        col_indices[col].push_back(row);
    }
};

// Message structure for BP
struct Message {
    std::vector<double> prob;  // [P(0), P(1)]
    
    Message() : prob(2, 0.5) {}
    Message(double p0, double p1) : prob{p0, p1} {}
    
    void normalize() {
        double sum = prob[0] + prob[1];
        if (sum > 0) {
            prob[0] /= sum;
            prob[1] /= sum;
        }
    }
};

class QLDPC_BP_Marginals {
private:
    SparseMatrix D_prime;  // Syndrome constraint matrix
    SparseMatrix D_L;      // Logical syndrome constraint matrix
    std::vector<int> s_prime;  // Observed syndrome vector
    std::vector<double> weights;  // Log-likelihood weights
    
    int n_vars;      // number of error bits
    int n_syndromes; // number of syndrome constraints
    int n_logical;   // number of logical syndrome bits
    
    // Factor graph connectivity
    std::vector<std::vector<int>> var_to_syn_checks;
    std::vector<std::vector<int>> syn_check_to_vars;
    std::vector<std::vector<int>> var_to_log_checks;
    std::vector<std::vector<int>> log_check_to_vars;
    
    // BP messages
    std::vector<std::vector<Message>> msg_var_to_syn;
    std::vector<std::vector<Message>> msg_syn_to_var;
    std::vector<std::vector<Message>> msg_sL_to_log;
    std::vector<std::vector<Message>> msg_log_to_sL;
    std::vector<std::vector<Message>> msg_var_to_log;
    std::vector<std::vector<Message>> msg_log_to_var;
    
public:
    QLDPC_BP_Marginals(const std::vector<std::vector<int>>& D_prime_matrix,
                       const std::vector<std::vector<int>>& D_L_matrix,
                       const std::vector<int>& s_prime_vec,
                       const std::vector<double>& weights_vec);
    
    // Main BP functions
    std::pair<bool, int> run_belief_propagation(int max_iterations = 50, double tolerance = 1e-6);
    std::vector<std::vector<double>> compute_logical_syndrome_marginals();
    std::pair<std::vector<int>, std::vector<std::vector<double>>> find_most_likely_logical_syndrome();
    
private:
    // Helper functions
    void build_factor_graph();
    void initialize_messages();
    void update_syndrome_check_messages();
    void update_logical_check_messages();
    void update_variable_messages();
    void update_logical_syndrome_messages();
    
    // Utility functions
    Message compute_parity_distribution(const std::vector<int>& var_indices, 
                                       const std::vector<std::vector<Message>>& messages);
    double compute_convolution_probability(const Message& msg1, const Message& msg2, bool xor_op);
};

#endif // MARGINAL_BP_DECODER_HPP 