# MuvFDE

MuvFDE turns the token embeddings for a query or document into one fixed-dimensional encoding (FDE) using Google's MUVERA algorithm. Index document FDEs for maximum inner product search, then rerank the retrieved candidates with MaxSim on their original token embeddings. [Google Research overview](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/) · [MUVERA paper](https://arxiv.org/abs/2405.19504)

![PyPI version](https://img.shields.io/pypi/v/muvfde?style=flat-square)
![Tests](https://img.shields.io/github/actions/workflow/status/viig99/muvfde/pip.yml?branch=main&style=flat-square)
![Downloads](https://img.shields.io/pypi/dm/muvfde?style=flat-square)
![License](https://img.shields.io/github/license/viig99/muvfde?style=flat-square)

## Install

```sh
python -m pip install --upgrade muvfde
```

The build matrix covers standard CPython 3.10–3.15. The release workflow builds x86-64 Linux wheels for glibc and musl. Source builds require a C++20 compiler, and pip installs the build dependencies, including CMake 4.4.5 or newer and nanobind 3.1 or newer. NumPy is installed as a runtime dependency.

## Encode a query and a document

Pass a two-dimensional NumPy array of finite `float32` values with shape `(num_tokens, embedding_dim)`. Each output is a one-dimensional NumPy `float32` array. Use token embeddings from the same model and keep the dimensionality, seed, and sketch settings aligned between queries and documents.

```python
import muvfde as m
import numpy as np


def make_config(document=False):
    config = (
        m.fixed_dimensional_encoding_config()
        .set_num_repetitions(20)
        .set_num_simhash_projections(5)
        .set_seed(42)
        .set_projection_type(m.projection_type.AMS_SKETCH)
        .set_projection_dimension(16)
    )
    if document:
        config.set_encoding_type(m.encoding_type.AVERAGE).enable_fill_empty()
    return config


query_tokens = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
document_tokens = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

query_fde = m.generate_fixed_dimensional_encoding(query_tokens, make_config())
document_fde = m.generate_fixed_dimensional_encoding(document_tokens, make_config(document=True))

score = float(query_fde @ document_fde)
assert query_fde.shape == document_fde.shape == (10_240,)
```

### Configuration

| Property | Default | Behavior |
| --- | --- | --- |
| `num_repetitions` | `1` | Positive number of independently seeded repetitions. |
| `num_simhash_projections` | `0` | Number of SimHash bits (0–30). Each additional bit doubles the number of partitions per repetition. |
| `seed` | `1` | Signed 32-bit seed. Repetition seeds advance in unsigned 32-bit arithmetic. |
| `encoding_type` | `DEFAULT_SUM` | Use `DEFAULT_SUM` for queries and `AVERAGE` for documents. |
| `projection_type` | `DEFAULT_IDENTITY` | Identity retains the input width. `AMS_SKETCH` maps each input coordinate to one signed output coordinate. |
| `projection_dimension` | `0` | Required positive output width for `AMS_SKETCH`. Identity uses the input width. |
| `fill_empty_partitions` | `False` | Documents can copy the point with the closest SimHash into an empty partition. Ties use the first input row. |
| `final_projection_dimension` | `0` | A positive value applies CountSketch to the concatenated encoding. Zero disables this step. |

The uncompressed length is `num_repetitions × 2**num_simhash_projections × projection_width`. The final CountSketch changes the returned length and still processes the full uncompressed encoding. Input width must be positive. An input with zero token rows returns a zero vector of the configured output length. Invalid settings raise `ValueError`, and an unrepresentably large encoding raises `OverflowError`.

Use the raw FDE dot product for candidate retrieval. Keep the model's query/document prompts, token masks and preprocessing consistent. Choose the projection settings and candidate count using retrieval measurements on your data. Persist the model revision, preprocessing, encoder build, and config with your index. [Implementation compatibility and audit](https://github.com/viig99/muvfde/blob/main/docs/compatibility.md)

## Retrieval benchmark

The NFCorpus test benchmark uses LightOn's `LateOn-regularized`, the highest-scoring model in the linked English NanoBEIR comparison and a checkpoint trained with MUVERA compression in mind. The benchmark measures full-corpus MaxSim, FDE dot product, and MaxSim reranking of FDE candidates, with and without mean-centering the token embeddings used for FDEs. [Sentence Transformers comparison](https://huggingface.co/blog/multi-vector-encoder) · [LightOn model card](https://huggingface.co/lightonai/LateOn-regularized)

```sh
python -m pip install -e .
python -m pip install -r integeration_test/requirements.txt
python integeration_test/benchmark.py --candidates 100
```

[LateOn benchmark results](https://github.com/viig99/muvfde/blob/main/integeration_test/lateon.md) · [Historical ColBERT and Jina results](https://github.com/viig99/muvfde/blob/main/integeration_test/benchmark.md)

## Test and release

```sh
python -m pip install -e . pytest
python -m pytest
```

A push to `main` runs test CI for code and build changes. A tag push records the release version. PyPI updates after a GitHub Release is published for that tag and the Wheels workflow successfully uploads the distributions using the repository's `pypi_password` secret. The tag determines the package version through setuptools-scm. Publish `v0.0.10` against the tested commit to release `muvfde==0.0.10`. [Wheels workflow](https://github.com/viig99/muvfde/blob/main/.github/workflows/wheels.yml)
