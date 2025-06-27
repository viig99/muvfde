import muvfde
import numpy as np

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
