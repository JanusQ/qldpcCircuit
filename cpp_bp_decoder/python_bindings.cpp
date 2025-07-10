#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "bp_decoder.hpp"

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

PYBIND11_MODULE(BPdecoder, m) {
    m.doc() = "Belief Propagation with Accumulated Likelihood Decoder for QLDPC Codes";
    
    py::class_<LikelihoodDecoder>(m, "likelihoodDecoder")
        .def(py::init([](py::array_t<int> D, py::array_t<int> DL, 
                         py::array_t<double> error_probs, int iterations,
                         const std::string& name) {
            auto D_vec = numpy_to_vector2d(D);
            auto DL_vec = numpy_to_vector2d(DL);
            auto probs_vec = numpy_to_vector_double(error_probs);
            return new LikelihoodDecoder(D_vec, DL_vec, probs_vec, iterations, name);
        }), py::arg("D"), py::arg("DL"), py::arg("error_probs"), py::arg("iterations"),
        py::arg("name") = "BP_AccumulatedLikelihood",
        "Initialize the decoder with decoding matrix D, logical check matrix DL, error probabilities, number of BP iterations, and optional name")
        
        .def("decode", [](LikelihoodDecoder& self, py::array_t<int> syndrome) {
            auto syndrome_vec = numpy_to_vector_int(syndrome);
            auto result = self.decode(syndrome_vec);
            
            // Convert result back to numpy array
            py::array_t<int> output(result.size());
            auto output_ptr = static_cast<int*>(output.mutable_unchecked<1>().mutable_data(0));
            for (size_t i = 0; i < result.size(); i++) {
                output_ptr[i] = result[i];
            }
            return output;
        }, py::arg("syndrome"),
        "Decode the syndrome and return the most likely logical syndrome")
        
        .def_property_readonly("name", &LikelihoodDecoder::get_name,
        "Get the decoder name");
} 