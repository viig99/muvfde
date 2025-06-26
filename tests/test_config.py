from muvfde import muvfde_ext as m


def test_default_configs():
    config = m.FixedDimensionalEncodingConfig()
    assert config.dimension == 0
    assert config.num_repetitions == 1
    assert config.num_simhash_projections == 0
    assert config.seed == 1
    assert config.projection_dimension == 0
    assert config.fill_empty_partitions is False
    assert config.final_projection_dimension == 0
