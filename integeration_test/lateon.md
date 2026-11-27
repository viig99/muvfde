# LateOn-regularized on NFCorpus

The NFCorpus test split contains 323 queries and 3,633 documents. This run used the same LightOn checkpoint and token embeddings for every ranking method. FDE settings were 20 repetitions, 5 SimHash bits, a 16-dimensional AMS projection, seed 42, and document partition fill. Each FDE has 10,240 coordinates. [Model checkpoint](https://huggingface.co/lightonai/LateOn-regularized/tree/4449aa20f8ca4b580b8ab46df7599fa392e9b076) · [NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus)

| Ranking | nDCG@10 | MaxSim top-10 overlap in FDE top-100 |
| --- | ---: | ---: |
| Full-corpus MaxSim | 0.3821 | — |
| Raw FDE dot product | 0.0909 | 35.8% |
| Raw FDE + MaxSim | 0.2456 | 35.8% |
| Mean-centered FDE dot product | 0.2314 | 72.4% |
| Mean-centered FDE + MaxSim | 0.3469 | 72.4% |

Mean-centering subtracts the corpus-wide mean document token vector from both query and document tokens before FDE generation. MaxSim uses the original tokens in every row. Each reranking row uses that MaxSim score for the top 100 FDE candidates. The overlap column is the mean share of the ten highest-scoring full-corpus MaxSim documents present in those candidates.

The centered FDE recovered 72.4% of the MaxSim top ten at depth 100. Reranking those candidates recovered 90.8% of full-corpus MaxSim nDCG@10 in this run. This benchmark measures ranking quality using exhaustive FDE dot products and reuses exact MaxSim scores for reranking.

Run `python integeration_test/benchmark.py --device cuda --candidates 100` after installing the benchmark requirements. The measured run used the MuvFDE 0.0.10 development build, Python 3.13.15, Sentence Transformers 6.3.0, Transformers 5.24.1, and PyTorch 2.14.1 with an NVIDIA RTX 5070 Ti.
