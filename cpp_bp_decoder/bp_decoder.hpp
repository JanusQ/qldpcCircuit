#ifndef BP_DECODER_HPP
#define BP_DECODER_HPP

#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>

// Sparse matrix representation for LDPC codes
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

class LikelihoodDecoder {
private:
    SparseMatrix D;   // Decoding matrix
    SparseMatrix DL;  // Logical check matrix
    std::vector<double> weights;  // ln((1-p)/p) for each bit
    int num_iterations;
    std::string name;  // Decoder name
    
    // BP message passing variables
    std::vector<std::vector<double>> check_to_bit_msgs;
    std::vector<std::vector<double>> bit_to_check_msgs;
    std::vector<double> bit_beliefs;
    
    // Accumulated likelihood map
    std::unordered_map<std::string, double> logical_syndrome_likelihood;
    
public:
    LikelihoodDecoder(const std::vector<std::vector<int>>& D_matrix,
                      const std::vector<std::vector<int>>& DL_matrix,
                      const std::vector<double>& error_probs,
                      int iterations,
                      const std::string& decoder_name = "BP_AccumulatedLikelihood");
    
    std::vector<int> decode(const std::vector<int>& syndrome);
    
    // Getter for name
    const std::string& get_name() const { return name; }
    
private:
    void initialize_messages();
    void run_bp_iteration(const std::vector<int>& syndrome);
    std::vector<int> sample_error_from_beliefs();
    std::vector<int> hard_decision_from_beliefs();
    std::string vector_to_string(const std::vector<int>& vec);
    std::vector<int> string_to_vector(const std::string& str);
    std::vector<int> compute_logical_syndrome(const std::vector<int>& error);
    double compute_error_likelihood(const std::vector<int>& error);
    bool check_syndrome_validity(const std::vector<int>& error, const std::vector<int>& syndrome);
};

#endif // BP_DECODER_HPP 