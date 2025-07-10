#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <chrono>

/**
 * Mean-Field Decoder for QLDPC Codes
 * 
 * This implementation solves the mean-field equations for CSS quantum LDPC codes:
 * 
 * n_i = tanh(sum_j (w_j/2) * (-1)^e_{0j} * (L_x)_{ji} * prod_k m_k^{(D_x)_{jk}} * prod_{i'!=i} n_{i'}^{(L_x)_{ji'}})
 * m_k = tanh(sum_j (w_j/2) * (-1)^e_{0j} * (D_x)_{jk} * prod_{k'} m_{k'}^{(D_x)_{jk'}} * prod_i n_i^{(L_x)_{ji}})
 * 
 * where:
 * - w_j = ln((1-p_j)/p_j) are the error weights
 * - e_{0j} are the initial error syndrome bits
 * - D_x and L_x are sparse matrices in CSC format
 * - n_i are the logical spin magnetizations
 * - m_k are the check spin magnetizations
 */

// CSC (Compressed Sparse Column) matrix structure
struct CSCMatrix {
    std::vector<int> row_indices;    // Row indices of non-zero elements
    std::vector<int> col_ptr;        // Column pointers (start of each column)
    std::vector<double> values;      // Non-zero values
    int num_rows;
    int num_cols;
    
    CSCMatrix() : num_rows(0), num_cols(0) {}
    
    CSCMatrix(int rows, int cols) : num_rows(rows), num_cols(cols) {
        col_ptr.resize(cols + 1, 0);
    }
};

/**
 * Mean-field decoder class
 */
class MeanFieldDecoder {
private:
    CSCMatrix D_x;           // Check matrix (syndrome to qubit)
    CSCMatrix L_x;           // Logical operator matrix (logical to qubit)
    std::vector<double> w;   // Error weights w_j = ln((1-p_j)/p_j)
    std::vector<int> e0;     // Initial error syndrome e_0
    int max_iterations;      // Maximum iterations for convergence
    double tolerance;        // Convergence tolerance
    std::mt19937 rng;        // Random number generator
    
public:
    MeanFieldDecoder(int max_iter = 100, double tol = 1e-6) 
        : max_iterations(max_iter), tolerance(tol) {
        // Initialize random number generator
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        rng.seed(seed);
    }
    
    /**
     * Set the check matrix D_x (syndrome to qubit mapping)
     */
    void setCheckMatrix(const CSCMatrix& matrix) {
        D_x = matrix;
    }
    
    /**
     * Set the logical operator matrix L_x (logical operators to qubit mapping)
     */
    void setLogicalMatrix(const CSCMatrix& matrix) {
        L_x = matrix;
    }
    
    /**
     * Set error weights w_j = ln((1-p_j)/p_j)
     */
    void setErrorWeights(const std::vector<double>& weights) {
        w = weights;
    }
    
    /**
     * Set initial error syndrome e_0
     */
    void setInitialSyndrome(const std::vector<int>& syndrome) {
        e0 = syndrome;
    }
    
    /**
     * Compute eta_j = (-1)^e_{0j} = 1 - 2*e_{0j}
     */
    std::vector<double> computeEta() const {
        std::vector<double> eta(e0.size());
        for (size_t j = 0; j < e0.size(); ++j) {
            eta[j] = 1.0 - 2.0 * e0[j];
        }
        return eta;
    }
    
    /**
     * Compute product of magnetizations for a given qubit j
     * prod_k m_k^{(D_x)_{jk}} * prod_i n_i^{(L_x)_{ji}}
     */
    double computeMagnetizationProduct(int j, 
                                     const std::vector<double>& m, 
                                     const std::vector<double>& n) const {
        double product = 1.0;
        
        // Product over check spins: prod_k m_k^{(D_x)_{jk}}
        for (int col = 0; col < D_x.num_cols; ++col) {
            for (int idx = D_x.col_ptr[col]; idx < D_x.col_ptr[col + 1]; ++idx) {
                if (D_x.row_indices[idx] == j) {
                    product *= m[col];
                    break;
                }
            }
        }
        
        // Product over logical spins: prod_i n_i^{(L_x)_{ji}}
        for (int col = 0; col < L_x.num_cols; ++col) {
            for (int idx = L_x.col_ptr[col]; idx < L_x.col_ptr[col + 1]; ++idx) {
                if (L_x.row_indices[idx] == j) {
                    product *= n[col];
                    break;
                }
            }
        }
        
        return product;
    }
    
    /**
     * Update check spin magnetization m_k
     */
    double updateCheckMagnetization(int k, 
                                   const std::vector<double>& m, 
                                   const std::vector<double>& n,
                                   const std::vector<double>& eta) const {
        double field = 0.0;
        
        // Sum over all qubits j that are connected to check k
        for (int idx = D_x.col_ptr[k]; idx < D_x.col_ptr[k + 1]; ++idx) {
            int j = D_x.row_indices[idx];
            
            // Compute product excluding m_k itself
            double product = 1.0;
            
            // Product over other check spins
            for (int k2 = 0; k2 < D_x.num_cols; ++k2) {
                if (k2 == k) continue;
                for (int idx2 = D_x.col_ptr[k2]; idx2 < D_x.col_ptr[k2 + 1]; ++idx2) {
                    if (D_x.row_indices[idx2] == j) {
                        product *= m[k2];
                        break;
                    }
                }
            }
            
            // Product over logical spins
            for (int i = 0; i < L_x.num_cols; ++i) {
                for (int idx2 = L_x.col_ptr[i]; idx2 < L_x.col_ptr[i + 1]; ++idx2) {
                    if (L_x.row_indices[idx2] == j) {
                        product *= n[i];
                        break;
                    }
                }
            }
            
            field += (w[j] / 2.0) * eta[j] * product;
        }
        
        return std::tanh(field);
    }
    
    /**
     * Update logical spin magnetization n_i
     */
    double updateLogicalMagnetization(int i, 
                                     const std::vector<double>& m, 
                                     const std::vector<double>& n,
                                     const std::vector<double>& eta) const {
        double field = 0.0;
        
        // Sum over all qubits j that are connected to logical operator i
        for (int idx = L_x.col_ptr[i]; idx < L_x.col_ptr[i + 1]; ++idx) {
            int j = L_x.row_indices[idx];
            
            // Compute product excluding n_i itself
            double product = 1.0;
            
            // Product over check spins
            for (int k = 0; k < D_x.num_cols; ++k) {
                for (int idx2 = D_x.col_ptr[k]; idx2 < D_x.col_ptr[k + 1]; ++idx2) {
                    if (D_x.row_indices[idx2] == j) {
                        product *= m[k];
                        break;
                    }
                }
            }
            
            // Product over other logical spins
            for (int i2 = 0; i2 < L_x.num_cols; ++i2) {
                if (i2 == i) continue;
                for (int idx2 = L_x.col_ptr[i2]; idx2 < L_x.col_ptr[i2 + 1]; ++idx2) {
                    if (L_x.row_indices[idx2] == j) {
                        product *= n[i2];
                        break;
                    }
                }
            }
            
            field += (w[j] / 2.0) * eta[j] * product;
        }
        
        return std::tanh(field);
    }
    
    /**
     * Check convergence by computing the maximum change in magnetizations
     */
    double checkConvergence(const std::vector<double>& m_old, 
                           const std::vector<double>& m_new,
                           const std::vector<double>& n_old, 
                           const std::vector<double>& n_new) const {
        double max_change = 0.0;
        
        // Check check spin changes
        for (size_t k = 0; k < m_old.size(); ++k) {
            max_change = std::max(max_change, std::abs(m_new[k] - m_old[k]));
        }
        
        // Check logical spin changes
        for (size_t i = 0; i < n_old.size(); ++i) {
            max_change = std::max(max_change, std::abs(n_new[i] - n_old[i]));
        }
        
        return max_change;
    }
    
    /**
     * Main decoding function
     * Returns the optimal logical correction l_i = (1 - sign(n_i))/2
     */
    std::vector<int> decode() {
        if (D_x.num_cols == 0 || L_x.num_cols == 0 || w.empty() || e0.empty()) {
            std::cerr << "Error: Matrices and error data not properly initialized" << std::endl;
            return {};
        }
        
        // Initialize magnetizations with small random values
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        
        std::vector<double> m(D_x.num_cols);  // Check spin magnetizations
        std::vector<double> n(L_x.num_cols);  // Logical spin magnetizations
        
        for (int k = 0; k < D_x.num_cols; ++k) {
            m[k] = dist(rng);
        }
        for (int i = 0; i < L_x.num_cols; ++i) {
            n[i] = dist(rng);
        }
        
        // Compute eta_j = (-1)^e_{0j}
        std::vector<double> eta = computeEta();
        
        // Iterative mean-field updates
        for (int iter = 0; iter < max_iterations; ++iter) {
            std::vector<double> m_new = m;
            std::vector<double> n_new = n;
            
            // Update check spin magnetizations
            for (int k = 0; k < D_x.num_cols; ++k) {
                m_new[k] = updateCheckMagnetization(k, m, n, eta);
            }
            
            // Update logical spin magnetizations
            for (int i = 0; i < L_x.num_cols; ++i) {
                n_new[i] = updateLogicalMagnetization(i, m, n, eta);
            }
            
            // Check convergence
            double max_change = checkConvergence(m, m_new, n, n_new);
            
            // Update magnetizations
            m = m_new;
            n = n_new;
            
            if (max_change < tolerance) {
                std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
                break;
            }
        }
        
        // Convert magnetizations to logical corrections
        std::vector<int> logical_correction(L_x.num_cols);
        for (int i = 0; i < L_x.num_cols; ++i) {
            // s_i = sign(n_i), l_i = (1 - s_i)/2
            logical_correction[i] = (1 - (n[i] >= 0 ? 1 : -1)) / 2;
        }
        
        return logical_correction;
    }
    
    /**
     * Get final magnetizations for analysis
     */
    std::pair<std::vector<double>, std::vector<double>> getMagnetizations() {
        // Re-run decoding to get final magnetizations
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        
        std::vector<double> m(D_x.num_cols);
        std::vector<double> n(L_x.num_cols);
        
        for (int k = 0; k < D_x.num_cols; ++k) {
            m[k] = dist(rng);
        }
        for (int i = 0; i < L_x.num_cols; ++i) {
            n[i] = dist(rng);
        }
        
        std::vector<double> eta = computeEta();
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            std::vector<double> m_new = m;
            std::vector<double> n_new = n;
            
            for (int k = 0; k < D_x.num_cols; ++k) {
                m_new[k] = updateCheckMagnetization(k, m, n, eta);
            }
            
            for (int i = 0; i < L_x.num_cols; ++i) {
                n_new[i] = updateLogicalMagnetization(i, m, n, eta);
            }
            
            double max_change = checkConvergence(m, m_new, n, n_new);
            m = m_new;
            n = n_new;
            
            if (max_change < tolerance) break;
        }
        
        return {m, n};
    }
}; 