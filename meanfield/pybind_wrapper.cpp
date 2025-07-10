#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "meanfield_decoder.cpp"

namespace py = pybind11;

/**
 * Python wrapper for the Mean-Field Decoder
 * 
 * This module provides a Python interface to the C++ mean-field decoder
 * for QLDPC codes, with support for sparse matrices and numpy arrays.
 */

// Helper function to convert scipy sparse matrix to CSC format
CSCMatrix scipy_sparse_to_csc(py::object sparse_matrix) {
    // Get the CSC format data from scipy sparse matrix
    py::object csc_matrix = sparse_matrix.attr("tocsc")();
    
    // Extract the data
    py::array_t<int> indices = csc_matrix.attr("indices").cast<py::array_t<int>>();
    py::array_t<int> indptr = csc_matrix.attr("indptr").cast<py::array_t<int>>();
    py::array_t<double> data = csc_matrix.attr("data").cast<py::array_t<double>>();
    
    // Get matrix dimensions
    int num_rows = csc_matrix.attr("shape").attr("__getitem__")(0).cast<int>();
    int num_cols = csc_matrix.attr("shape").attr("__getitem__")(1).cast<int>();
    
    // Create CSC matrix
    CSCMatrix csc(num_rows, num_cols);
    
    // Copy data
    auto indices_buf = indices.template unchecked<1>();
    auto indptr_buf = indptr.template unchecked<1>();
    auto data_buf = data.template unchecked<1>();
    
    csc.row_indices.resize(indices_buf.shape(0));
    csc.col_ptr.resize(indptr_buf.shape(0));
    csc.values.resize(data_buf.shape(0));
    
    for (py::ssize_t i = 0; i < indices_buf.shape(0); ++i) {
        csc.row_indices[i] = indices_buf(i);
    }
    
    for (py::ssize_t i = 0; i < indptr_buf.shape(0); ++i) {
        csc.col_ptr[i] = indptr_buf(i);
    }
    
    for (py::ssize_t i = 0; i < data_buf.shape(0); ++i) {
        csc.values[i] = data_buf(i);
    }
    
    return csc;
}

// Helper function to convert numpy array to vector
template<typename T>
std::vector<T> numpy_to_vector(py::array_t<T> array) {
    auto buf = array.template unchecked<1>();
    std::vector<T> vec(buf.shape(0));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        vec[i] = buf(i);
    }
    return vec;
}

// Helper function to convert vector to numpy array
template<typename T>
py::array_t<T> vector_to_numpy(const std::vector<T>& vec) {
    py::array_t<T> array(vec.size());
    auto buf = array.template mutable_unchecked<1>();
    for (size_t i = 0; i < vec.size(); ++i) {
        buf(i) = vec[i];
    }
    return array;
}

// Python wrapper class
class PyMeanFieldDecoder {
private:
    MeanFieldDecoder decoder;
    
public:
    PyMeanFieldDecoder(int max_iterations = 100, double tolerance = 1e-6)
        : decoder(max_iterations, tolerance) {}
    
    /**
     * Set the check matrix D_x from a scipy sparse matrix
     */
    void set_check_matrix(py::object sparse_matrix) {
        CSCMatrix csc = scipy_sparse_to_csc(sparse_matrix);
        decoder.setCheckMatrix(csc);
    }
    
    /**
     * Set the logical operator matrix L_x from a scipy sparse matrix
     */
    void set_logical_matrix(py::object sparse_matrix) {
        CSCMatrix csc = scipy_sparse_to_csc(sparse_matrix);
        decoder.setLogicalMatrix(csc);
    }
    
    /**
     * Set error weights from numpy array
     */
    void set_error_weights(py::array_t<double> weights) {
        std::vector<double> w = numpy_to_vector(weights);
        decoder.setErrorWeights(w);
    }
    
    /**
     * Set initial error syndrome from numpy array
     */
    void set_initial_syndrome(py::array_t<int> syndrome) {
        std::vector<int> e0 = numpy_to_vector(syndrome);
        decoder.setInitialSyndrome(e0);
    }
    
    /**
     * Compute eta_j = (-1)^e_{0j} and return as numpy array
     */
    py::array_t<double> compute_eta() {
        std::vector<double> eta = decoder.computeEta();
        return vector_to_numpy(eta);
    }
    
    /**
     * Main decoding function
     * Returns the optimal logical correction as numpy array
     */
    py::array_t<int> decode() {
        std::vector<int> result = decoder.decode();
        return vector_to_numpy(result);
    }
    
    /**
     * Get final magnetizations for analysis
     * Returns tuple of (check_magnetizations, logical_magnetizations)
     */
    py::tuple get_magnetizations() {
        auto [m, n] = decoder.getMagnetizations();
        return py::make_tuple(vector_to_numpy(m), vector_to_numpy(n));
    }
    
    /**
     * Set convergence parameters
     */
    void set_parameters(int max_iterations, double tolerance) {
        // Create new decoder with updated parameters
        decoder = MeanFieldDecoder(max_iterations, tolerance);
    }
};

PYBIND11_MODULE(meanfield_decoder, m) {
    m.doc() = R"pbdoc(
        Mean-Field Decoder for QLDPC Codes
        
        This module provides a Python interface to the C++ mean-field decoder
        for CSS quantum LDPC codes. The decoder solves the mean-field equations:
        
        n_i = tanh(sum_j (w_j/2) * (-1)^e_{0j} * (L_x)_{ji} * prod_k m_k^{(D_x)_{jk}} * prod_{i'!=i} n_{i'}^{(L_x)_{ji'}})
        m_k = tanh(sum_j (w_j/2) * (-1)^e_{0j} * (D_x)_{jk} * prod_{k'} m_{k'}^{(D_x)_{jk'}} * prod_i n_i^{(L_x)_{ji}})
        
        where:
        - w_j = ln((1-p_j)/p_j) are the error weights
        - e_{0j} are the initial error syndrome bits
        - D_x and L_x are sparse matrices in CSC format
        - n_i are the logical spin magnetizations
        - m_k are the check spin magnetizations
    )pbdoc";
    
    py::class_<PyMeanFieldDecoder>(m, "MeanFieldDecoder")
        .def(py::init<int, double>(), 
             py::arg("max_iterations") = 100, 
             py::arg("tolerance") = 1e-6,
             "Initialize the mean-field decoder")
        .def("set_check_matrix", &PyMeanFieldDecoder::set_check_matrix,
             py::arg("sparse_matrix"),
             "Set the check matrix D_x from a scipy sparse matrix")
        .def("set_logical_matrix", &PyMeanFieldDecoder::set_logical_matrix,
             py::arg("sparse_matrix"),
             "Set the logical operator matrix L_x from a scipy sparse matrix")
        .def("set_error_weights", &PyMeanFieldDecoder::set_error_weights,
             py::arg("weights"),
             "Set error weights w_j = ln((1-p_j)/p_j) from numpy array")
        .def("set_initial_syndrome", &PyMeanFieldDecoder::set_initial_syndrome,
             py::arg("syndrome"),
             "Set initial error syndrome e_0 from numpy array")
        .def("compute_eta", &PyMeanFieldDecoder::compute_eta,
             "Compute eta_j = (-1)^e_{0j} and return as numpy array")
        .def("decode", &PyMeanFieldDecoder::decode,
             "Perform mean-field decoding and return optimal logical correction")
        .def("get_magnetizations", &PyMeanFieldDecoder::get_magnetizations,
             "Get final magnetizations (check_mags, logical_mags) for analysis")
        .def("set_parameters", &PyMeanFieldDecoder::set_parameters,
             py::arg("max_iterations"), py::arg("tolerance"),
             "Set convergence parameters");
} 