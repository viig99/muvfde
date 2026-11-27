#pragma once

#include <nanobind/eigen/dense.h>

#include "fixed_dimensional_encoding_config.h"

namespace nb = nanobind;

namespace multidimensional_encoding
{
    Eigen::VectorXf GenerateFixedDimensionalEncoding(
        const nb::DRef<const Eigen::MatrixXf> &input_embedding_matrix,
        const FixedDimensionalEncodingConfig &config);
} // namespace multidimensional_encoding
