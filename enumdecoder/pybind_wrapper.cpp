#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "qldpc_decoder.cpp"
#include "qldpc_decoder_csc_simple.cpp"
namespace py = pybind11;

PYBIND11_MODULE(enumlikelydecoder, m) {
    py::class_<EnumDecoder>(m, "EnumDecoder")
        .def(py::init<const std::vector<std::vector<bool>>&, const std::vector<std::vector<bool>>&, const std::vector<double>&, int>(),
             py::arg("D_matrix"), py::arg("DL_matrix"), py::arg("priors"), py::arg("maximum_weight"))
        .def("decode", &EnumDecoder::decode, py::arg("syndrome"))
        .def("test_logical_error_rate", &EnumDecoder::test_logical_error_rate, py::arg("p"), py::arg("trials"))
        .def("update_weights", &EnumDecoder::update_weights, py::arg("priors"))
        .def("print_stats", &EnumDecoder::print_stats);
    
    py::class_<EnumDecoderCSC>(m, "EnumDecoderCSC")
        .def(py::init<const std::vector<std::vector<bool>>&, const std::vector<std::vector<bool>>&, const std::vector<double>&, int>(),
             py::arg("D_matrix"), py::arg("DL_matrix"), py::arg("priors"), py::arg("maximum_weight"))
        .def("decode", &EnumDecoderCSC::decode, py::arg("syndrome"))
        .def("test_logical_error_rate", &EnumDecoderCSC::test_logical_error_rate, py::arg("p"), py::arg("trials"))
        .def("update_weights", &EnumDecoderCSC::update_weights, py::arg("priors"))
        .def("print_stats", &EnumDecoderCSC::print_stats);
    
    m.doc() = "QLDPC Enumeration Decoder with CSC optimization support";
    
    m.attr("__version__") = "1.0.0";
} 