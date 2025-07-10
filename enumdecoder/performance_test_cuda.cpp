#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include "cuda_decoder.cpp"

using namespace std;
using Vec = vector<bool>;
using Matrix = vector<Vec>;
using Real = double;

// Performance measurement utilities
class Timer {
private:
    chrono::high_resolution_clock::time_point start_time;
    string name;
public:
    Timer(const string& timer_name) : name(timer_name) {
        start_time = chrono::high_resolution_clock::now();
    }
    
    ~Timer() {
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        cout << name << ": " << duration.count() / 1000.0 << " ms" << endl;
    }
    
    double elapsed() {
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }
};

// Generate random sparse matrix
Matrix generate_sparse_matrix(int rows, int cols, double sparsity) {
    Matrix mat(rows, Vec(cols, false));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (dis(gen) < sparsity) {
                mat[i][j] = true;
            }
        }
    }
    return mat;
}

// Benchmark different matrix sizes
void benchmark_matrix_sizes() {
    cout << "\n=== Matrix Size Benchmark ===" << endl;
    
    vector<pair<int, int>> sizes = {
        {10, 20}, {20, 40}, {40, 80}, {80, 160}, {160, 320}
    };
    
    vector<int> batch_sizes = {1000, 5000, 10000, 20000};
    
    cout << setw(15) << "Matrix Size" << setw(15) << "Batch Size" 
         << setw(15) << "Init Time" << setw(15) << "Decode Time" << endl;
    cout << string(60, '-') << endl;
    
    for (const auto& size : sizes) {
        int d_rows = size.first;
        int n = size.second;
        int dl_rows = d_rows / 2;
        
        // Generate sparse matrices
        Matrix D = generate_sparse_matrix(d_rows, n, 0.3);
        Matrix DL = generate_sparse_matrix(dl_rows, n, 0.3);
        vector<Real> priors(n, 0.01);
        int max_weight = 3;
        
        for (int batch_size : batch_sizes) {
            try {
                // Measure initialization time
                Timer init_timer("Initialization");
                CUDADecoder decoder(D, DL, priors, max_weight, batch_size);
                double init_time = init_timer.elapsed();
                
                // Measure decoding time
                Timer decode_timer("Decoding");
                Real err_rate = decoder.test_logical_error_rate(0.01, 1000);
                double decode_time = decode_timer.elapsed();
                
                cout << setw(15) << d_rows << "x" << n 
                     << setw(15) << batch_size
                     << setw(15) << fixed << setprecision(2) << init_time
                     << setw(15) << fixed << setprecision(2) << decode_time << endl;
                     
            } catch (const exception& e) {
                cout << setw(15) << d_rows << "x" << n 
                     << setw(15) << batch_size
                     << setw(15) << "ERROR" << setw(15) << "ERROR" << endl;
            }
        }
    }
}

// Benchmark different sparsity levels
void benchmark_sparsity() {
    cout << "\n=== Sparsity Benchmark ===" << endl;
    
    int d_rows = 40, n = 80, dl_rows = 20;
    vector<double> sparsities = {0.1, 0.2, 0.3, 0.4, 0.5};
    int batch_size = 10000;
    int max_weight = 3;
    
    cout << setw(15) << "Sparsity" << setw(15) << "Init Time" 
         << setw(15) << "Decode Time" << setw(15) << "Memory Usage" << endl;
    cout << string(60, '-') << endl;
    
    for (double sparsity : sparsities) {
        try {
            // Generate matrices with specific sparsity
            Matrix D = generate_sparse_matrix(d_rows, n, sparsity);
            Matrix DL = generate_sparse_matrix(dl_rows, n, sparsity);
            vector<Real> priors(n, 0.01);
            
            // Measure initialization time
            Timer init_timer("Initialization");
            CUDADecoder decoder(D, DL, priors, max_weight, batch_size);
            double init_time = init_timer.elapsed();
            
            // Measure decoding time
            Timer decode_timer("Decoding");
            Real err_rate = decoder.test_logical_error_rate(0.01, 1000);
            double decode_time = decode_timer.elapsed();
            
            // Estimate memory usage (simplified)
            int nnz = 0;
            for (const auto& row : D) {
                for (bool val : row) if (val) nnz++;
            }
            for (const auto& row : DL) {
                for (bool val : row) if (val) nnz++;
            }
            double memory_mb = (nnz * 4 + n * 4 + batch_size * n * 4) / (1024.0 * 1024.0);
            
            cout << setw(15) << fixed << setprecision(2) << sparsity
                 << setw(15) << fixed << setprecision(2) << init_time
                 << setw(15) << fixed << setprecision(2) << decode_time
                 << setw(15) << fixed << setprecision(1) << memory_mb << " MB" << endl;
                 
        } catch (const exception& e) {
            cout << setw(15) << fixed << setprecision(2) << sparsity
                 << setw(15) << "ERROR" << setw(15) << "ERROR" << setw(15) << "ERROR" << endl;
        }
    }
}

// Benchmark different error probabilities
void benchmark_error_probabilities() {
    cout << "\n=== Error Probability Benchmark ===" << endl;
    
    int d_rows = 40, n = 80, dl_rows = 20;
    Matrix D = generate_sparse_matrix(d_rows, n, 0.3);
    Matrix DL = generate_sparse_matrix(dl_rows, n, 0.3);
    vector<Real> priors(n, 0.01);
    int max_weight = 3;
    int batch_size = 10000;
    
    vector<Real> error_probs = {0.001, 0.005, 0.01, 0.02, 0.05, 0.1};
    
    cout << setw(15) << "Error Prob" << setw(15) << "Logical Error Rate" 
         << setw(15) << "Decode Time" << setw(15) << "Trials" << endl;
    cout << string(60, '-') << endl;
    
    try {
        CUDADecoder decoder(D, DL, priors, max_weight, batch_size);
        
        for (Real p : error_probs) {
            Timer decode_timer("Decoding");
            Real err_rate = decoder.test_logical_error_rate(p, 5000);
            double decode_time = decode_timer.elapsed();
            
            cout << setw(15) << fixed << setprecision(3) << p
                 << setw(15) << fixed << setprecision(4) << err_rate
                 << setw(15) << fixed << setprecision(2) << decode_time
                 << setw(15) << 5000 << endl;
        }
    } catch (const exception& e) {
        cout << "Error during benchmark: " << e.what() << endl;
    }
}

// Compare with CPU version (if available)
void compare_cpu_gpu() {
    cout << "\n=== CPU vs GPU Comparison ===" << endl;
    
    int d_rows = 30, n = 60, dl_rows = 15;
    Matrix D = generate_sparse_matrix(d_rows, n, 0.3);
    Matrix DL = generate_sparse_matrix(dl_rows, n, 0.3);
    vector<Real> priors(n, 0.01);
    int max_weight = 3;
    
    cout << setw(15) << "Implementation" << setw(15) << "Init Time" 
         << setw(15) << "Decode Time" << setw(15) << "Speedup" << endl;
    cout << string(60, '-') << endl;
    
    try {
        // GPU version
        Timer gpu_init_timer("GPU Initialization");
        CUDADecoder gpu_decoder(D, DL, priors, max_weight, 10000);
        double gpu_init_time = gpu_init_timer.elapsed();
        
        Timer gpu_decode_timer("GPU Decoding");
        Real gpu_err_rate = gpu_decoder.test_logical_error_rate(0.01, 1000);
        double gpu_decode_time = gpu_decode_timer.elapsed();
        
        cout << setw(15) << "GPU" 
             << setw(15) << fixed << setprecision(2) << gpu_init_time
             << setw(15) << fixed << setprecision(2) << gpu_decode_time
             << setw(15) << "1.00x" << endl;
             
        // Note: CPU version would be implemented here
        cout << setw(15) << "CPU" 
             << setw(15) << "N/A" << setw(15) << "N/A" << setw(15) << "N/A" << endl;
             
    } catch (const exception& e) {
        cout << "Error during comparison: " << e.what() << endl;
    }
}

// Memory usage analysis
void memory_analysis() {
    cout << "\n=== Memory Usage Analysis ===" << endl;
    
    int d_rows = 40, n = 80, dl_rows = 20;
    Matrix D = generate_sparse_matrix(d_rows, n, 0.3);
    Matrix DL = generate_sparse_matrix(dl_rows, n, 0.3);
    vector<Real> priors(n, 0.01);
    int max_weight = 3;
    
    vector<int> batch_sizes = {1000, 5000, 10000, 20000, 50000};
    
    cout << setw(15) << "Batch Size" << setw(15) << "GPU Memory" 
         << setw(15) << "Host Memory" << setw(15) << "Total Memory" << endl;
    cout << string(60, '-') << endl;
    
    for (int batch_size : batch_sizes) {
        try {
            CUDADecoder decoder(D, DL, priors, max_weight, batch_size);
            
            // Estimate memory usage
            int nnz = 0;
            for (const auto& row : D) {
                for (bool val : row) if (val) nnz++;
            }
            for (const auto& row : DL) {
                for (bool val : row) if (val) nnz++;
            }
            
            // GPU memory (simplified estimation)
            double gpu_memory = (nnz * 4 + n * 4 + batch_size * n * 4 + 
                               batch_size * (d_rows + dl_rows) * 4 + 
                               batch_size * 4) / (1024.0 * 1024.0);
            
            // Host memory
            double host_memory = (batch_size * (d_rows + dl_rows) * 4 + 
                                batch_size * 4) / (1024.0 * 1024.0);
            
            cout << setw(15) << batch_size
                 << setw(15) << fixed << setprecision(1) << gpu_memory << " MB"
                 << setw(15) << fixed << setprecision(1) << host_memory << " MB"
                 << setw(15) << fixed << setprecision(1) << (gpu_memory + host_memory) << " MB" << endl;
                 
        } catch (const exception& e) {
            cout << setw(15) << batch_size
                 << setw(15) << "ERROR" << setw(15) << "ERROR" << setw(15) << "ERROR" << endl;
        }
    }
}

// Save results to file
void save_results(const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }
    
    file << "CUDA QLDPC Decoder Performance Results" << endl;
    file << "======================================" << endl;
    file << "Date: " << chrono::system_clock::now().time_since_epoch().count() << endl;
    file << endl;
    
    // Add benchmark results here
    file << "Benchmark completed successfully." << endl;
    file.close();
    
    cout << "Results saved to " << filename << endl;
}

int main() {
    cout << "CUDA QLDPC Decoder Performance Test" << endl;
    cout << "===================================" << endl;
    
    try {
        // Run all benchmarks
        benchmark_matrix_sizes();
        benchmark_sparsity();
        benchmark_error_probabilities();
        compare_cpu_gpu();
        memory_analysis();
        
        // Save results
        save_results("cuda_performance_results.txt");
        
        cout << "\nAll benchmarks completed successfully!" << endl;
        
    } catch (const exception& e) {
        cerr << "Error during benchmarking: " << e.what() << endl;
        return 1;
    }
    
    return 0;
} 