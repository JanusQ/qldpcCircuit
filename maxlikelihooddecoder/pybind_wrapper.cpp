#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "qldpc_decoder.h"

namespace py = pybind11;

PYBIND11_MODULE(maxlikelihooddecoder, m) {
    py::class_<QLDPCDecoder>(m, "QLDPCDecoder")
        .def(py::init<const std::vector<std::vector<int>>&, const std::vector<double>&, int, int>())
        .def("decode", &QLDPCDecoder::decode)
        .def("update_weights", &QLDPCDecoder::update_weights)
        .def("print_stats", &QLDPCDecoder::print_stats)
        .def("verify_decoding", &QLDPCDecoder::verify_decoding)
        .def("test_logical_error_rate", &QLDPCDecoder::test_logical_error_rate)
        .def("test_different_error_rates", &QLDPCDecoder::test_different_error_rates);
} 