#ifndef CSC_MATRIX_H
#define CSC_MATRIX_H

#include <vector>
#include <iostream>

// CSC (Compressed Sparse Column) Matrix structure
struct CSCMatrix {
    std::vector<int> values;     // Non-zero values
    std::vector<int> row_indices; // Row indices of non-zero values
    std::vector<int> col_ptr;    // Column pointers (start index of each column)
    int rows, cols;              // Matrix dimensions
    
    CSCMatrix() : rows(0), cols(0) {}
    
    CSCMatrix(int r, int c) : rows(r), cols(c) {
        col_ptr.resize(cols + 1, 0);
    }
    
    // Convert from dense boolean matrix
    void from_dense(const std::vector<std::vector<bool>>& dense) {
        rows = dense.size();
        cols = dense[0].size();
        col_ptr.resize(cols + 1, 0);
        
        values.clear();
        row_indices.clear();
        
        // Count non-zeros per column
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                if (dense[i][j]) {
                    values.push_back(1);
                    row_indices.push_back(i);
                    col_ptr[j + 1]++;
                }
            }
        }
        
        // Convert to cumulative sum
        for (int j = 1; j <= cols; j++) {
            col_ptr[j] += col_ptr[j - 1];
        }
    }
    
    // Convert to dense boolean matrix
    std::vector<std::vector<bool>> to_dense() const {
        std::vector<std::vector<bool>> dense(rows, std::vector<bool>(cols, false));
        
        for (int j = 0; j < cols; j++) {
            for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) {
                dense[row_indices[k]][j] = true;
            }
        }
        
        return dense;
    }
    
    // Matrix-vector multiplication (mod 2)
    std::vector<bool> multiply_mod2(const std::vector<bool>& vec) const {
        std::vector<bool> result(rows, false);
        
        for (int j = 0; j < cols; j++) {
            if (vec[j]) {  // Only process if input vector has 1 at position j
                for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) {
                    result[row_indices[k]] = !result[row_indices[k]];  // XOR operation
                }
            }
        }
        
        return result;
    }
    
    // Print matrix for debugging
    void print() const {
        std::cout << "CSC Matrix (" << rows << "x" << cols << "):" << std::endl;
        std::cout << "Values: ";
        for (int v : values) std::cout << v << " ";
        std::cout << std::endl;
        
        std::cout << "Row indices: ";
        for (int r : row_indices) std::cout << r << " ";
        std::cout << std::endl;
        
        std::cout << "Col ptr: ";
        for (int c : col_ptr) std::cout << c << " ";
        std::cout << std::endl;
    }
    
    // Get sparsity ratio
    double get_sparsity() const {
        int nnz = values.size();
        return 1.0 - (double)nnz / (rows * cols);
    }
};

#endif // CSC_MATRIX_H 