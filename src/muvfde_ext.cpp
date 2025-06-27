#include <nanobind/nanobind.h>
#include <cstdint>
#include "fixed_dimensional_encoding_config.h"
#include "fixed_dimensional_encoding.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(muvfde_ext, m) {
    m.doc() = "Convert Multi-dimensional vector's to fixed-dimensional encodings";

    // 1) Bind the C++ class
    nb::class_<multidimensional_encoding::FixedDimensionalEncodingConfig>(m, "fixed_dimensional_encoding_config", "Configuration for fixed-dimensional encoding of multi-dimensional vectors")
        // default ctor
        .def(nb::init<>())
        // fluent setters
        .def("set_num_repetitions", &multidimensional_encoding::FixedDimensionalEncodingConfig::set_num_repetitions, "r"_a)
        .def("set_num_simhash_projections", &multidimensional_encoding::FixedDimensionalEncodingConfig::set_num_simhash_projections, "k"_a)
        .def("set_seed", &multidimensional_encoding::FixedDimensionalEncodingConfig::set_seed, "s"_a)
        .def("set_encoding_type", &multidimensional_encoding::FixedDimensionalEncodingConfig::set_encoding_type, "type"_a)
        .def("set_projection_dimension", &multidimensional_encoding::FixedDimensionalEncodingConfig::set_projection_dimension, "dp"_a)
        .def("set_projection_type", &multidimensional_encoding::FixedDimensionalEncodingConfig::set_projection_type, "pt"_a)
        .def("enable_fill_empty", &multidimensional_encoding::FixedDimensionalEncodingConfig::enable_fill_empty, "v"_a = true)
        .def("set_final_projection_dimension", &multidimensional_encoding::FixedDimensionalEncodingConfig::set_final_projection_dimension, "d"_a)
        // read-only accessors as Python properties
        .def_prop_ro("num_repetitions", &multidimensional_encoding::FixedDimensionalEncodingConfig::num_repetitions)
        .def_prop_ro("num_simhash_projections", &multidimensional_encoding::FixedDimensionalEncodingConfig::num_simhash_projections)
        .def_prop_ro("seed", &multidimensional_encoding::FixedDimensionalEncodingConfig::seed)
        .def_prop_ro("projection_dimension", &multidimensional_encoding::FixedDimensionalEncodingConfig::projection_dimension)
        .def_prop_ro("fill_empty_partitions", &multidimensional_encoding::FixedDimensionalEncodingConfig::fill_empty_partitions)
        .def_prop_ro("final_projection_dimension", &multidimensional_encoding::FixedDimensionalEncodingConfig::final_projection_dimension)
        .def_prop_ro("encoding_type", &multidimensional_encoding::FixedDimensionalEncodingConfig::encoding_type)
        .def_prop_ro("projection_type", &multidimensional_encoding::FixedDimensionalEncodingConfig::projection_type);

    nb::enum_<multidimensional_encoding::EncodingType>(m, "encoding_type", "Encoding types for fixed-dimensional encoding of multi-dimensional vectors")
        .value("DEFAULT_SUM", multidimensional_encoding::EncodingType::DEFAULT_SUM)
        .value("AVERAGE",    multidimensional_encoding::EncodingType::AVERAGE)
        .export_values();

    nb::enum_<multidimensional_encoding::ProjectionType>(m, "projection_type", "Projection types for fixed-dimensional encoding of multi-dimensional vectors")
        .value("DEFAULT_IDENTITY", multidimensional_encoding::ProjectionType::DEFAULT_IDENTITY)
        .value("AMS_SKETCH",       multidimensional_encoding::ProjectionType::AMS_SKETCH)
        .export_values();

    m.def("generate_fixed_dimensional_encoding",
          &multidimensional_encoding::GenerateFixedDimensionalEncoding,
          "input_embedding_matrix"_a, "config"_a,
          "Generates a fixed-dimensional encoding for the input embedding matrix using the provided configuration.");
}
