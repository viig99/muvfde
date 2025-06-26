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

def test_config_setters():
    config = m.FixedDimensionalEncodingConfig()
    config.set_dimension(10)
    config.set_num_repetitions(2)
    config.set_num_simhash_projections(5)
    config.set_seed(42)
    config.set_projection_dimension(3)
    config.enable_fill_empty(True)
    config.set_final_projection_dimension(4)

    assert config.dimension == 10
    assert config.num_repetitions == 2
    assert config.num_simhash_projections == 5
    assert config.seed == 42
    assert config.projection_dimension == 3
    assert config.fill_empty_partitions is True
    assert config.final_projection_dimension == 4
