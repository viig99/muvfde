# MuvFDE
A library to generate fixed dimensional embeddings for multi-dimensional vectors based on [Google Muvera](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/).

![Version](https://img.shields.io/pypi/v/muvfde?style=flat-square)
![UnitTests](https://img.shields.io/github/actions/workflow/status/viig99/muvfde/pip.yml?branch=main&style=flat-square)
![Downloads](https://img.shields.io/pypi/dm/muvfde?style=flat-square)
![License](https://img.shields.io/github/license/viig99/muvfde?style=flat-square)


## Installation
```shell script
pip install --upgrade muvfde
```

## Integeration Benchmark
```shell script
uv venv --python 3.12
uv pip install -e .
uv pip install -r integeration_test/requirements.txt
python integeration_test/benchmark.py
```

## Results
[Benchmark Results](https://github.com/viig99/muvfde/blob/main/integeration_test/benchmark.md)

## References
* [Google Muvera Blog](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/)
* [MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings](https://arxiv.org/abs/2405.19504)
* [Reference Fixed Dimensional embeddings code](https://github.com/google/graph-mining/tree/main/sketching/point_cloud)