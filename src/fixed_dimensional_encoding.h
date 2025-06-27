#pragma once

#include <nanobind/eigen/dense.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include "fixed_dimensional_encoding_config.h"

namespace nb = nanobind;

namespace multidimensional_encoding
{
    Eigen::VectorXf GenerateFixedDimensionalEncoding(
        const nb::DRef<const Eigen::MatrixXf> &input_embedding_matrix,
        const FixedDimensionalEncodingConfig &config);

    Eigen::VectorXf GenerateQueryFixedDimensionalEncoding(
        const nb::DRef<const Eigen::MatrixXf> &input_embedding_matrix,
        const FixedDimensionalEncodingConfig &config);

    Eigen::VectorXf GenerateDocumentFixedDimensionalEncoding(
        const nb::DRef<const Eigen::MatrixXf> &input_embedding_matrix,
        const FixedDimensionalEncodingConfig &config);

    uint32_t SimHashPartitionIndex(const Eigen::VectorXf &input_vector);

    uint32_t DistanceToSimHashPartition(const Eigen::VectorXf &input_vector,
                                        uint32_t index);

    Eigen::VectorXf ApplyCountSketchToVector(
        const nb::DRef<const Eigen::VectorXf> &input_vector,
        uint32_t final_dimension, uint32_t seed);

    Eigen::MatrixXf AMSProjectionMatrixFromSeed(uint32_t seed,
                                                uint32_t dimension,
                                                uint32_t projection_dim);

    Eigen::MatrixXf SimHashMatrixFromSeed(uint32_t seed,
                                          uint32_t dimension,
                                          uint32_t num_projections);
}
