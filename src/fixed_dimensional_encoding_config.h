#pragma once
#include <cstdint>

namespace multidimensional_encoding
{
    enum class EncodingType
    {
        DEFAULT_SUM = 0,
        AVERAGE = 1
    };
    enum class ProjectionType
    {
        DEFAULT_IDENTITY = 0,
        AMS_SKETCH = 1
    };

    class FixedDimensionalEncodingConfig
    {
    public:
        /* ---------- enums kept identical to the proto ---------- */

        /* ---------- ctor with proto-equivalent defaults ---------- */
        FixedDimensionalEncodingConfig()
            : num_repetitions_(1),
              num_simhash_projections_(0),
              seed_(1),
              encoding_type_(EncodingType::DEFAULT_SUM),
              projection_dimension_(0),
              projection_type_(ProjectionType::DEFAULT_IDENTITY),
              fill_empty_partitions_(false),
              final_projection_dimension_(0) {}

        /* ---------- fluent setters (optional) ---------- */
        FixedDimensionalEncodingConfig &set_num_repetitions(int32_t r)
        {
            num_repetitions_ = r;
            return *this;
        }
        FixedDimensionalEncodingConfig &set_num_simhash_projections(int32_t k)
        {
            num_simhash_projections_ = k;
            return *this;
        }
        FixedDimensionalEncodingConfig &set_seed(int32_t s)
        {
            seed_ = s;
            return *this;
        }
        FixedDimensionalEncodingConfig &set_encoding_type(EncodingType e)
        {
            encoding_type_ = e;
            return *this;
        }
        FixedDimensionalEncodingConfig &set_projection_dimension(int32_t dp)
        {
            projection_dimension_ = dp;
            return *this;
        }
        FixedDimensionalEncodingConfig &set_projection_type(ProjectionType p)
        {
            projection_type_ = p;
            return *this;
        }
        FixedDimensionalEncodingConfig &enable_fill_empty(bool v = true)
        {
            fill_empty_partitions_ = v;
            return *this;
        }
        FixedDimensionalEncodingConfig &set_final_projection_dimension(int32_t d)
        {
            final_projection_dimension_ = d;
            return *this;
        }

        /* ---------- read-only accessors (mirrors proto API) ---------- */
        int32_t num_repetitions() const { return num_repetitions_; }
        int32_t num_simhash_projections() const { return num_simhash_projections_; }
        int32_t seed() const { return seed_; }
        EncodingType encoding_type() const { return encoding_type_; }
        int32_t projection_dimension() const { return projection_dimension_; }
        ProjectionType projection_type() const { return projection_type_; }
        bool fill_empty_partitions() const { return fill_empty_partitions_; }
        int32_t final_projection_dimension() const { return final_projection_dimension_; }

        /* proto-style “has_…” helpers */
        bool has_projection_dimension() const { return projection_dimension_ > 0; }
        bool has_final_projection_dimension() const { return final_projection_dimension_ > 0; }

    private:
        /* ---------- data ---------- */
        int32_t num_repetitions_;
        int32_t num_simhash_projections_;
        int32_t seed_;
        EncodingType encoding_type_;
        int32_t projection_dimension_;
        ProjectionType projection_type_;
        bool fill_empty_partitions_;
        int32_t final_projection_dimension_;
    };

} // namespace multidimensional_encoding
