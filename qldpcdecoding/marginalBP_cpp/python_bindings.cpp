#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "marginal_bp_decoder.hpp"

namespace py = pybind11;

// Helper function to convert numpy array to vector<vector<int>>
std::vector<std::vector<int>> numpy_to_vector2d(py::array_t<int> arr) {
    if (arr.ndim() != 2) {
        throw std::runtime_error("Input must be a 2D array");
    }
    
    auto shape = arr.shape();
    std::vector<std::vector<int>> result(shape[0], std::vector<int>(shape[1]));
    
    auto unchecked = arr.unchecked<2>();
    for (py::ssize_t i = 0; i < shape[0]; i++) {
        for (py::ssize_t j = 0; j < shape[1]; j++) {
            result[i][j] = unchecked(i, j);
        }
    }
    
    return result;
}

// Helper function to convert numpy array to vector<int>
std::vector<int> numpy_to_vector_int(py::array_t<int> arr) {
    if (arr.ndim() != 1) {
        throw std::runtime_error("Input must be a 1D array");
    }
    
    auto shape = arr.shape();
    std::vector<int> result(shape[0]);
    
    auto unchecked = arr.unchecked<1>();
    for (py::ssize_t i = 0; i < shape[0]; i++) {
        result[i] = unchecked(i);
    }
    
    return result;
}

// Helper function to convert numpy array to vector<double>
std::vector<double> numpy_to_vector_double(py::array_t<double> arr) {
    if (arr.ndim() != 1) {
        throw std::runtime_error("Input must be a 1D array");
    }
    
    auto shape = arr.shape();
    std::vector<double> result(shape[0]);
    
    auto unchecked = arr.unchecked<1>();
    for (py::ssize_t i = 0; i < shape[0]; i++) {
        result[i] = unchecked(i);
    }
    
    return result;
}

// Helper function to convert vector<vector<double>> to numpy array
py::array_t<double> vector2d_to_numpy(const std::vector<std::vector<double>>& vec) {
    if (vec.empty()) {
        return py::array_t<double>();
    }
    
    size_t rows = vec.size();
    size_t cols = vec[0].size();
    
    py::array_t<double> result({rows, cols});
    auto result_ptr = static_cast<double*>(result.mutable_unchecked<2>().mutable_data(0, 0));
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result_ptr[i * cols + j] = vec[i][j];
        }
    }
    
    return result;
}

// Helper function to convert vector<int> to numpy array
py::array_t<int> vector_to_numpy(const std::vector<int>& vec) {
    py::array_t<int> result(vec.size());
    auto result_ptr = static_cast<int*>(result.mutable_unchecked<1>().mutable_data(0));
    
    for (size_t i = 0; i < vec.size(); i++) {
        result_ptr[i] = vec[i];
    }
    
    return result;
}

PYBIND11_MODULE(MarginalBPDecoder, m) {
    m.doc() = "QLDPC Belief Propagation with Marginal Estimation Decoder";
    
    py::class_<QLDPC_BP_Marginals>(m, "QLDPC_BP_Marginals")
        .def(py::init([](py::array_t<int> D_prime, py::array_t<int> D_L, 
                         py::array_t<int> s_prime, py::array_t<double> weights) {
            auto D_prime_vec = numpy_to_vector2d(D_prime);
            auto D_L_vec = numpy_to_vector2d(D_L);
            auto s_prime_vec = numpy_to_vector_int(s_prime);
            auto weights_vec = numpy_to_vector_double(weights);
            return new QLDPC_BP_Marginals(D_prime_vec, D_L_vec, s_prime_vec, weights_vec);
        }), py::arg("D_prime"), py::arg("D_L"), py::arg("s_prime"), py::arg("weights"),
        "Initialize the marginal BP decoder with syndrome constraint matrix D_prime, logical check matrix D_L, observed syndrome s_prime, and log-likelihood weights")
        
        .def("run_belief_propagation", [](QLDPC_BP_Marginals& self, int max_iterations, double tolerance) {
            auto result = self.run_belief_propagation(max_iterations, tolerance);
            return py::make_tuple(result.first, result.second);
        }, py::arg("max_iterations") = 50, py::arg("tolerance") = 1e-6,
        "Run belief propagation to convergence. Returns (converged, iterations)")
        
        .def("compute_logical_syndrome_marginals", [](QLDPC_BP_Marginals& self) {
            auto marginals = self.compute_logical_syndrome_marginals();
            return vector2d_to_numpy(marginals);
        }, "Compute marginal probabilities P(s_L_i | s') for each logical syndrome bit")
        
        .def("find_most_likely_logical_syndrome", [](QLDPC_BP_Marginals& self) {
            auto result = self.find_most_likely_logical_syndrome();
            auto most_likely_s_L = vector_to_numpy(result.first);
            auto marginals = vector2d_to_numpy(result.second);
            return py::make_tuple(most_likely_s_L, marginals);
        }, "Find the most likely logical syndrome based on marginals. Returns (most_likely_s_L, marginals)");
} 