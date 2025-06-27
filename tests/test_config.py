import muvfde

def test_default_configs():
    config = muvfde.fixed_dimensional_encoding_config()
    assert config.num_repetitions == 1
    assert config.num_simhash_projections == 0
    assert config.seed == 1
    assert config.projection_dimension == 0
    assert config.fill_empty_partitions is False
    assert config.final_projection_dimension == 0
    assert config.encoding_type == muvfde.encoding_type.DEFAULT_SUM
    assert config.projection_type == muvfde.projection_type.DEFAULT_IDENTITY

def test_config_setters():
    config = muvfde.fixed_dimensional_encoding_config()
    config.set_num_repetitions(2)
    config.set_num_simhash_projections(5)
    config.set_seed(42)
    config.set_projection_dimension(3)
    config.enable_fill_empty(True)
    config.set_final_projection_dimension(4)
    config.set_encoding_type(muvfde.encoding_type.AVERAGE)
    config.set_projection_type(muvfde.projection_type.AMS_SKETCH)

    assert config.num_repetitions == 2
    assert config.num_simhash_projections == 5
    assert config.seed == 42
    assert config.projection_dimension == 3
    assert config.fill_empty_partitions is True
    assert config.final_projection_dimension == 4
    assert config.encoding_type == muvfde.encoding_type.AVERAGE
    assert config.projection_type == muvfde.projection_type.AMS_SKETCH
