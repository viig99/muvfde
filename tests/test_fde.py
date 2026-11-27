import muvfde
import numpy as np
import pytest

def test_generate_fde_for_query():

    num_repetitions = 20
    num_simhash_projections = 4
    num_projection_dimensions = 8

    fde_dimension = num_repetitions * (
        2**num_simhash_projections) * num_projection_dimensions

    config = muvfde.fixed_dimensional_encoding_config()
    config.set_num_repetitions(num_repetitions)
    config.set_num_simhash_projections(num_simhash_projections)
    config.set_seed(42)
    config.set_projection_dimension(num_projection_dimensions)
    config.set_encoding_type(muvfde.encoding_type.DEFAULT_SUM)
    config.set_projection_type(muvfde.projection_type.AMS_SKETCH)

    input_embedding_matrix = np.random.rand(32, 768).astype(
        np.float32)  # 32 points, 768 dimensions

    fde = muvfde.generate_fixed_dimensional_encoding(input_embedding_matrix,
                                                     config)
    assert fde is not None
    assert fde.shape == (fde_dimension, )


def test_generate_fde_for_document():

    num_repetitions = 20
    num_simhash_projections = 4
    num_projection_dimensions = 8

    fde_dimension = num_repetitions * (
        2**num_simhash_projections) * num_projection_dimensions

    config = muvfde.fixed_dimensional_encoding_config()
    config.set_num_repetitions(num_repetitions)
    config.set_num_simhash_projections(num_simhash_projections)
    config.set_seed(42)
    config.set_projection_dimension(num_projection_dimensions)
    config.set_encoding_type(muvfde.encoding_type.AVERAGE)
    config.enable_fill_empty(True)
    config.set_projection_type(muvfde.projection_type.AMS_SKETCH)

    input_embedding_matrix = np.random.rand(2048, 768).astype(
        np.float32)  # 2048 points, 768 dimensions

    fde = muvfde.generate_fixed_dimensional_encoding(
        input_embedding_matrix, config)
    assert fde is not None
    assert fde.shape == (fde_dimension, )


def test_aggregation_with_identity_projection():
    points = np.array([[2, 4, 8], [4, 8, 16]], dtype=np.float32)
    config = muvfde.fixed_dimensional_encoding_config().set_num_repetitions(2)

    query = muvfde.generate_fixed_dimensional_encoding(points, config)
    config.set_encoding_type(muvfde.encoding_type.AVERAGE)
    document = muvfde.generate_fixed_dimensional_encoding(points, config)

    assert query.dtype == document.dtype == np.float32
    np.testing.assert_array_equal(query, [6, 12, 24, 6, 12, 24])
    np.testing.assert_array_equal(document, [3, 6, 12, 3, 6, 12])


@pytest.mark.parametrize("hashes,fill", [(0, False), (3, False), (3, True)])
@pytest.mark.parametrize("final_dimension", [0, 5])
def test_empty_document_returns_zeros(hashes, fill, final_dimension):
    config = (
        muvfde.fixed_dimensional_encoding_config()
        .set_encoding_type(muvfde.encoding_type.AVERAGE)
        .set_num_repetitions(2)
        .set_num_simhash_projections(hashes)
        .enable_fill_empty(fill)
        .set_final_projection_dimension(final_dimension)
    )

    fde = muvfde.generate_fixed_dimensional_encoding(
        np.empty((0, 3), dtype=np.float32), config
    )
    assert fde.dtype == np.float32
    np.testing.assert_array_equal(fde, np.zeros(final_dimension or 2 * 2**hashes * 3))


def test_fill_empty_partitions_uses_nearest_simhash():
    points = np.array([[1, 2, -3], [-1, -2, 3], [2, 4, -6]], dtype=np.float32)
    config = muvfde.fixed_dimensional_encoding_config().set_num_simhash_projections(2)

    buckets = []
    for point in points:
        query = muvfde.generate_fixed_dimensional_encoding(point[None, :], config)
        buckets.append(np.flatnonzero(query.reshape(4, 3).any(axis=1)).item())
    assert buckets[0] == buckets[2] != buckets[1]

    config.set_encoding_type(muvfde.encoding_type.AVERAGE).enable_fill_empty()
    document = muvfde.generate_fixed_dimensional_encoding(points, config).reshape(4, 3)

    # The reference bucket order encodes each sign with the preceding stored bit.
    signs = [bucket ^ (bucket >> 1) for bucket in buckets]
    for bucket, row in enumerate(document):
        occupied = [i for i, point_bucket in enumerate(buckets) if point_bucket == bucket]
        if occupied:
            expected = points[occupied].mean(axis=0)
        else:
            target_signs = bucket ^ (bucket >> 1)
            nearest = min(range(len(points)), key=lambda i: (signs[i] ^ target_signs).bit_count())
            expected = points[nearest]
        np.testing.assert_array_equal(row, expected)


@pytest.mark.parametrize("use_ams", [False, True])
def test_encoding_handles_strided_inputs(use_ams):
    points = np.arange(128, dtype=np.float32).reshape(16, 8)[::-2, ::2]
    config = muvfde.fixed_dimensional_encoding_config().set_num_simhash_projections(3)
    if use_ams:
        config.set_projection_type(muvfde.projection_type.AMS_SKETCH).set_projection_dimension(3)

    encode = muvfde.generate_fixed_dimensional_encoding
    np.testing.assert_array_equal(encode(points, config), encode(points.copy(), config))


@pytest.mark.parametrize("final_dimension", [0, 5])
def test_sparse_projection_is_linear(final_dimension):
    config = (
        muvfde.fixed_dimensional_encoding_config()
        .set_projection_type(muvfde.projection_type.AMS_SKETCH)
        .set_projection_dimension(3)
        .set_final_projection_dimension(final_dimension)
    )
    basis = np.eye(8, dtype=np.float32)
    encode = muvfde.generate_fixed_dimensional_encoding
    columns = np.stack([encode(row[None, :], config) for row in basis])
    assert columns.shape == (8, final_dimension or 3)
    np.testing.assert_array_equal(np.abs(columns).sum(axis=1), np.ones(8))

    weights = np.arange(8, dtype=np.float32)
    np.testing.assert_array_equal(encode(weights[None, :], config), weights @ columns)


def test_repetition_seed_crosses_signed_limit():
    points = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.float32)
    config = (
        muvfde.fixed_dimensional_encoding_config()
        .set_num_simhash_projections(2)
        .set_projection_type(muvfde.projection_type.AMS_SKETCH)
        .set_projection_dimension(2)
        .set_seed(2**31 - 1)
        .set_num_repetitions(2)
    )
    both = muvfde.generate_fixed_dimensional_encoding(points, config)
    config.set_num_repetitions(1).set_seed(-(2**31))
    second = muvfde.generate_fixed_dimensional_encoding(points, config)

    np.testing.assert_array_equal(both[second.size:], second)


@pytest.mark.parametrize(
    "setter,value,message",
    [
        ("set_num_repetitions", 0, "num_repetitions must be positive"),
        ("set_num_repetitions", -1, "num_repetitions must be positive"),
        ("set_num_simhash_projections", -1, "Unsupported number"),
        ("set_num_simhash_projections", 31, "Unsupported number"),
        ("set_final_projection_dimension", -1, "final_projection_dimension"),
        ("enable_fill_empty", True, "does not support fill_empty_partitions"),
        ("set_projection_type", muvfde.projection_type.AMS_SKETCH, "positive projection dimension"),
    ],
)
def test_invalid_config(setter, value, message):
    config = muvfde.fixed_dimensional_encoding_config()
    getattr(config, setter)(value)
    with pytest.raises(ValueError, match=message):
        muvfde.generate_fixed_dimensional_encoding(np.ones((2, 3), dtype=np.float32), config)


def test_rejects_zero_width_input():
    with pytest.raises(ValueError, match="at least one column"):
        muvfde.generate_fixed_dimensional_encoding(
            np.empty((2, 0), dtype=np.float32), muvfde.fixed_dimensional_encoding_config()
        )


def test_rejects_overflowing_uncompressed_size():
    config = (
        muvfde.fixed_dimensional_encoding_config()
        .set_num_repetitions(2**31 - 1)
        .set_num_simhash_projections(30)
        .set_final_projection_dimension(8)
    )
    with pytest.raises(OverflowError, match="uncompressed FDE is too large"):
        muvfde.generate_fixed_dimensional_encoding(np.ones((1, 2), dtype=np.float32), config)
