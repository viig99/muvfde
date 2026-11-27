#include "fixed_dimensional_encoding.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace multidimensional_encoding
{
    namespace
    {
        struct EncodingDimensions
        {
            Eigen::Index partitions;
            Eigen::Index projection;
            Eigen::Index total;
        };

        EncodingDimensions GetEncodingDimensions(
            Eigen::Index input_dimension, const FixedDimensionalEncodingConfig &config)
        {
            if (input_dimension <= 0)
            {
                throw std::invalid_argument("Input embeddings must have at least one column.");
            }
            if (config.num_repetitions() <= 0)
            {
                throw std::invalid_argument("num_repetitions must be positive.");
            }
            if (config.num_simhash_projections() < 0 ||
                config.num_simhash_projections() >= 31)
            {
                throw std::invalid_argument(
                    "Unsupported number of simhash projections: " +
                    std::to_string(config.num_simhash_projections()));
            }
            if (config.final_projection_dimension() < 0)
            {
                throw std::invalid_argument("final_projection_dimension must be nonnegative.");
            }

            const Eigen::Index projection_dim =
                config.projection_type() == ProjectionType::DEFAULT_IDENTITY
                    ? input_dimension
                    : config.projection_dimension();
            if (projection_dim <= 0)
            {
                throw std::invalid_argument(
                    "A positive projection dimension must be specified in the config "
                    "if a non-identity projection type is used.");
            }

            const Eigen::Index num_partitions = Eigen::Index{1} << config.num_simhash_projections();
            const Eigen::Index max_size = std::numeric_limits<Eigen::Index>::max() / sizeof(float);
            if (projection_dim > max_size / num_partitions / config.num_repetitions())
            {
                throw std::overflow_error("The uncompressed FDE is too large.");
            }
            return {num_partitions, projection_dim,
                    config.num_repetitions() * num_partitions * projection_dim};
        }

        uint32_t SimHashPartitionIndex(const Eigen::MatrixXf &sketch, Eigen::Index point)
        {
            uint32_t partition = 0;
            for (Eigen::Index i = 0; i < sketch.cols(); ++i)
            {
                partition = (partition << 1) | ((sketch(point, i) > 0) ^ (partition & 1));
            }
            return partition;
        }

        uint32_t SimHashBits(uint32_t partition)
        {
            // Each appended bit is XORed with the previous stored bit.
            return partition ^ (partition >> 1);
        }

        Eigen::MatrixXf SimHashMatrixFromSeed(uint32_t seed, Eigen::Index dimension,
                                             Eigen::Index num_projections)
        {
            std::mt19937 gen(seed);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            return Eigen::MatrixXf::NullaryExpr(dimension, num_projections, [&]()
                                              { return dist(gen); });
        }

        Eigen::MatrixXf ApplyAMSProjection(
            const nb::DRef<const Eigen::MatrixXf> &points,
            Eigen::Index projection_dim, uint32_t seed)
        {
            struct Projection
            {
                Eigen::Index bucket;
                float sign;
            };

            std::vector<Projection> projections(points.cols());
            std::mt19937 gen(seed);
            std::uniform_int_distribution<std::size_t> index_dist(0, projection_dim - 1);
            std::bernoulli_distribution sign_dist(0.5);
            for (auto &projection : projections)
            {
                projection.bucket = index_dist(gen);
                projection.sign = sign_dist(gen) ? 1.0f : -1.0f;
            }

            Eigen::MatrixXf out = Eigen::MatrixXf::Zero(points.rows(), projection_dim);
            for (Eigen::Index row = 0; row < points.rows(); ++row)
            {
                for (Eigen::Index col = 0; col < points.cols(); ++col)
                {
                    const auto &projection = projections[col];
                    out(row, projection.bucket) += projection.sign * points(row, col);
                }
            }
            return out;
        }

        Eigen::VectorXf ApplyCountSketchToVector(
            const Eigen::VectorXf &input, int32_t final_dimension, uint32_t seed)
        {
            Eigen::VectorXf out = Eigen::VectorXf::Zero(final_dimension);
            std::mt19937 gen(seed);
            std::uniform_int_distribution<int32_t> index_dist(0, final_dimension - 1);
            std::bernoulli_distribution sign_dist(0.5);
            for (Eigen::Index i = 0; i < input.size(); ++i)
            {
                const int32_t index = index_dist(gen);
                const float sign = sign_dist(gen) ? 1.0f : -1.0f;
                out[index] += sign * input[i];
            }
            return out;
        }

        struct PartitionRepresentative
        {
            uint32_t bits;
            Eigen::Index point;
        };

        Eigen::Index NearestPoint(uint32_t partition,
                                  const std::vector<PartitionRepresentative> &representatives)
        {
            const uint32_t bits = SimHashBits(partition);
            Eigen::Index nearest = representatives.front().point;
            int min_distance = std::numeric_limits<int>::max();
            for (const auto &representative : representatives)
            {
                const int distance = std::popcount(bits ^ representative.bits);
                if (distance < min_distance)
                {
                    nearest = representative.point;
                    min_distance = distance;
                    // The destination partition is empty, so the closest possible distance is one.
                    if (distance == 1)
                    {
                        break;
                    }
                }
            }
            return nearest;
        }
    }

    Eigen::VectorXf GenerateFixedDimensionalEncoding(
        const nb::DRef<const Eigen::MatrixXf> &embedding_matrix,
        const FixedDimensionalEncodingConfig &config)
    {
        const bool average = config.encoding_type() == EncodingType::AVERAGE;
        const bool fill_empty = config.fill_empty_partitions();
        if (fill_empty && !average)
        {
            throw std::invalid_argument("Query FDE encoding does not support fill_empty_partitions.");
        }

        const auto [num_partitions, projection_dim, output_size] =
            GetEncodingDimensions(embedding_matrix.cols(), config);
        const Eigen::Index num_points = embedding_matrix.rows();
        if (num_points == 0)
        {
            return Eigen::VectorXf::Zero(
                config.has_final_projection_dimension() ? config.final_projection_dimension() : output_size);
        }

        const bool use_identity = config.projection_type() == ProjectionType::DEFAULT_IDENTITY;
        const int32_t num_hashes = config.num_simhash_projections();
        const auto seed = static_cast<uint32_t>(config.seed());
        Eigen::VectorXf out = Eigen::VectorXf::Zero(output_size);
        std::vector<Eigen::Index> partition_sizes(average ? num_partitions : 0);
        std::vector<PartitionRepresentative> representatives;

        for (int repetition = 0; repetition < config.num_repetitions(); ++repetition)
        {
            const uint32_t repetition_seed = seed + repetition;
            Eigen::MatrixXf sketch;
            if (num_hashes > 0)
            {
                sketch = embedding_matrix * SimHashMatrixFromSeed(
                    repetition_seed, embedding_matrix.cols(), num_hashes);
            }

            Eigen::MatrixXf projected_matrix;
            if (!use_identity)
            {
                projected_matrix = ApplyAMSProjection(embedding_matrix, projection_dim, repetition_seed);
            }
            const auto points = use_identity ? embedding_matrix
                                            : nb::DRef<const Eigen::MatrixXf>(projected_matrix);

            std::fill(partition_sizes.begin(), partition_sizes.end(), 0);
            representatives.clear();
            const Eigen::Index repetition_offset = repetition * num_partitions * projection_dim;
            for (Eigen::Index point = 0; point < num_points; ++point)
            {
                const uint32_t partition = SimHashPartitionIndex(sketch, point);
                out.segment(repetition_offset + partition * projection_dim, projection_dim) +=
                    points.row(point).transpose();
                if (average && partition_sizes[partition]++ == 0 && fill_empty)
                {
                    // Points sharing a partition have the same Hamming distance.
                    // Keeping the first point also preserves the reference tie break.
                    representatives.push_back({SimHashBits(partition), point});
                }
            }

            if (average)
            {
                for (Eigen::Index partition = 0; partition < num_partitions; ++partition)
                {
                    auto bucket = out.segment(repetition_offset + partition * projection_dim,
                                              projection_dim);
                    const auto count = partition_sizes[partition];
                    if (count > 0)
                    {
                        bucket /= static_cast<float>(count);
                    }
                    else if (fill_empty)
                    {
                        bucket = points.row(NearestPoint(partition, representatives)).transpose();
                    }
                }
            }
        }

        if (config.has_final_projection_dimension())
        {
            return ApplyCountSketchToVector(out, config.final_projection_dimension(), seed);
        }
        return out;
    }
} // namespace multidimensional_encoding
