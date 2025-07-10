#include "csc_matrix.h"
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void CSCMatrix::from_dense(const vector<vector<bool>>& dense_matrix) {
    rows = dense_matrix.size();
    if (rows == 0) {
        cols = 0;
        return;
    }
    cols = dense_matrix[0].size();
    
    values.clear();
    row_indices.clear();
    col_ptr.clear();
    col_ptr.push_back(0);  // Start of first column
    
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            if (dense_matrix[i][j]) {
                values.push_back(1);
                row_indices.push_back(i);
            }
        }
        col_ptr.push_back(values.size());
    }
}

vector<bool> CSCMatrix::multiply_mod2(const vector<bool>& vec) const {
    vector<bool> result(rows, false);
    
    for (int j = 0; j < cols; j++) {
        if (vec[j]) {  // Only process if vector element is true
            for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) {
                result[row_indices[k]] = !result[row_indices[k]];  // XOR operation
            }
        }
    }
    
    return result;
}

vector<int> CSCMatrix::multiply_mod2_int(const vector<int>& vec) const {
    vector<int> result(rows, 0);
    
    for (int j = 0; j < cols; j++) {
        if (vec[j]) {  // Only process if vector element is non-zero
            for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) {
                result[row_indices[k]] ^= 1;  // XOR operation
            }
        }
    }
    
    return result;
}

double CSCMatrix::get_sparsity() const {
    if (rows == 0 || cols == 0) return 0.0;
    return 1.0 - (double)values.size() / (rows * cols);
}

void CSCMatrix::print() const {
    cout << "CSC Matrix (" << rows << "x" << cols << ")" << endl;
    cout << "Values: ";
    for (int val : values) cout << val << " ";
    cout << endl;
    cout << "Row indices: ";
    for (int idx : row_indices) cout << idx << " ";
    cout << endl;
    cout << "Column pointers: ";
    for (int ptr : col_ptr) cout << ptr << " ";
    cout << endl;
    cout << "Sparsity: " << get_sparsity() * 100 << "%" << endl;
}

vector<vector<bool>> CSCMatrix::to_dense() const {
    vector<vector<bool>> dense(rows, vector<bool>(cols, false));
    
    for (int j = 0; j < cols; j++) {
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) {
            dense[row_indices[k]][j] = true;
        }
    }
    
    return dense;
}

bool CSCMatrix::operator==(const CSCMatrix& other) const {
    if (rows != other.rows || cols != other.cols) return false;
    if (values != other.values) return false;
    if (row_indices != other.row_indices) return false;
    if (col_ptr != other.col_ptr) return false;
    return true;
}

CSCMatrix CSCMatrix::transpose() const {
    CSCMatrix transposed;
    transposed.rows = cols;
    transposed.cols = rows;
    
    // Count non-zeros per row (original columns)
    vector<int> row_counts(rows, 0);
    for (int val : values) {
        row_counts[val]++;
    }
    
    // Build column pointers for transposed matrix
    transposed.col_ptr.push_back(0);
    for (int i = 0; i < rows; i++) {
        transposed.col_ptr.push_back(transposed.col_ptr.back() + row_counts[i]);
    }
    
    // Build values and row indices
    vector<int> row_positions(rows, 0);
    for (int j = 0; j < cols; j++) {
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) {
            int row = row_indices[k];
            int pos = transposed.col_ptr[row] + row_positions[row]++;
            transposed.values.push_back(values[k]);
            transposed.row_indices.push_back(j);
        }
    }
    
    return transposed;
}

int CSCMatrix::get_nnz() const {
    return values.size();
}

vector<int> CSCMatrix::get_row_counts() const {
    vector<int> counts(rows, 0);
    for (int idx : row_indices) {
        counts[idx]++;
    }
    return counts;
}

vector<int> CSCMatrix::get_col_counts() const {
    vector<int> counts(cols, 0);
    for (int j = 0; j < cols; j++) {
        counts[j] = col_ptr[j + 1] - col_ptr[j];
    }
    return counts;
} 