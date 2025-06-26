#include <nanobind/nanobind.h>
#include <cstdint>
#include "fixed_dimensional_encoding_config.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(muvfde_ext, m) {
    m.doc() = "Convert Multi-dimensional vector's to fixed-dimensional encodings";

    // 1) Bind the C++ class
    auto cfg = nb::class_<multidimensional_encoding::FixedDimensionalEncodingConfig>(m, "FixedDimensionalEncodingConfig")
        // default ctor
        .def(nb::init<>())
        // fluent setters
        .def("set_dimension", &multidimensional_encoding::FixedDimensionalEncodingConfig::set_dimension, "d"_a)
        .def("set_num_repetitions", &multidimensional_encoding::FixedDimensionalEncodingConfig::set_num_repetitions, "r"_a)
        .def("set_num_simhash_projections", &multidimensional_encoding::FixedDimensionalEncodingConfig::set_num_simhash_projections, "k"_a)
        .def("set_seed", &multidimensional_encoding::FixedDimensionalEncodingConfig::set_seed, "s"_a)
        .def("set_encoding_type", &multidimensional_encoding::FixedDimensionalEncodingConfig::set_encoding_type, "type"_a)
        .def("set_projection_dimension", &multidimensional_encoding::FixedDimensionalEncodingConfig::set_projection_dimension, "dp"_a)
        .def("set_projection_type", &multidimensional_encoding::FixedDimensionalEncodingConfig::set_projection_type, "pt"_a)
        .def("enable_fill_empty", &multidimensional_encoding::FixedDimensionalEncodingConfig::enable_fill_empty, "v"_a = true)
        .def("set_final_projection_dimension", &multidimensional_encoding::FixedDimensionalEncodingConfig::set_final_projection_dimension, "d"_a)
        // read-only accessors as Python properties
        .def_prop_ro("dimension", &multidimensional_encoding::FixedDimensionalEncodingConfig::dimension)
        .def_prop_ro("num_repetitions", &multidimensional_encoding::FixedDimensionalEncodingConfig::num_repetitions)
        .def_prop_ro("num_simhash_projections", &multidimensional_encoding::FixedDimensionalEncodingConfig::num_simhash_projections)
        .def_prop_ro("seed", &multidimensional_encoding::FixedDimensionalEncodingConfig::seed)
        .def_prop_ro("projection_dimension", &multidimensional_encoding::FixedDimensionalEncodingConfig::projection_dimension)
        .def_prop_ro("fill_empty_partitions", &multidimensional_encoding::FixedDimensionalEncodingConfig::fill_empty_partitions)
        .def_prop_ro("final_projection_dimension", &multidimensional_encoding::FixedDimensionalEncodingConfig::final_projection_dimension)
        ;

    // 2) Bind EncodingType enum inside the class scope
    nb::enum_<multidimensional_encoding::FixedDimensionalEncodingConfig::EncodingType>(cfg, "EncodingType")
        .value("DEFAULT_SUM", multidimensional_encoding::FixedDimensionalEncodingConfig::EncodingType::DEFAULT_SUM)
        .value("AVERAGE",    multidimensional_encoding::FixedDimensionalEncodingConfig::EncodingType::AVERAGE)
        .export_values();

    // 3) Bind ProjectionType enum inside the class scope
    nb::enum_<multidimensional_encoding::FixedDimensionalEncodingConfig::ProjectionType>(cfg, "ProjectionType")
        .value("DEFAULT_IDENTITY", multidimensional_encoding::FixedDimensionalEncodingConfig::ProjectionType::DEFAULT_IDENTITY)
        .value("AMS_SKETCH",       multidimensional_encoding::FixedDimensionalEncodingConfig::ProjectionType::AMS_SKETCH)
        .export_values();
}
