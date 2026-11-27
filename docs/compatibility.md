# MUVERA implementation compatibility

## Google's C++ reference

The encoder source, header, and config proto are identical between Google's June 23, 2025 FDE commit and the final public revision reviewed here. The Bazel file changed its Protobuf and Abseil dependency names. [Original FDE commit](https://github.com/google/graph-mining/commit/621ddb4e4c3b3ba3dc7d15ad34f1e0bb39ef25d5) · [Final reviewed point-cloud source](https://github.com/google/graph-mining/tree/dca52ed1c35ca530d11ea7f76b0bf74634cdeb5c/sketching/point_cloud)

| Google reference | MuvFDE |
| --- | --- |
| Flat `vector<float>` and proto `dimension` | Two-dimensional input. The number of columns supplies the embedding dimension. |
| `num_repetitions`, `num_simhash_projections`, `seed` | Same fields and defaults. Both use MT19937 with Gaussian projections and the reference bucket order. |
| `DEFAULT_SUM` / `AVERAGE` | Sum query points or average document points in each partition. |
| `DEFAULT_IDENTITY` / `AMS_SKETCH` and `projection_dimension` | Identity or the same signed, one-destination projection per input coordinate. MuvFDE accumulates the sparse projection directly. |
| `fill_empty_partitions` | Same document-only Hamming-distance fill and first-input-row tie break. MuvFDE considers one representative per occupied bucket. |
| Optional `final_projection_dimension` | Same seeded CountSketch for a positive dimension. Python uses zero to disable it. |
| Status and `vector<float>` | Python exceptions and a one-dimensional NumPy `float32` array. |

The Google source can divide by zero when an empty document is averaged with zero SimHash bits, and it can read row `-1` when an empty document requests partition fill. Its flat input path also takes a remainder by an unchecked embedding dimension. MuvFDE returns zeros for zero token rows, requires a positive input width, validates dimensions before allocating, and checks the uncompressed length before multiplying. Repetition seeds wrap in unsigned 32-bit arithmetic. [Google encoder source](https://github.com/google/graph-mining/blob/dca52ed1c35ca530d11ea7f76b0bf74634cdeb5c/sketching/point_cloud/fixed_dimensional_encoding.cc)

The projection and partition formulas match for valid inputs. Sparse summation can change float32 rounding. C++ standard libraries may produce different random-distribution samples even with the same seed. Use a consistent encoder build and config for both sides of a persisted index.

## Rust and Weaviate

| Implementation | Hashing and projection | Aggregation and output |
| --- | --- | --- |
| Google / MuvFDE | Gaussian SimHash with the reference bucket numbering. Identity or sparse AMS, plus optional final CountSketch. | Query sum and document mean within each bucket. Optional document fill. Length `repetitions × 2**hash_bits × projection_width` before CountSketch. |
| `muvera-rs` 0.3.0 | Gaussian dot products followed by ReLU. Every token is projected onto all output coordinates. | The Rust variant labeled `Sum` takes the maximum over tokens. `Avg` takes their mean. The output length equals `buckets`, and there is no SimHash bucket assignment, repetition loop, or empty-bucket fill. |
| Weaviate | Go PCG with Box–Muller Gaussian draws. SimHash sign bits are packed directly. A dense random sign matrix projects each aggregate and scales it by `1 / sqrt(DProjections)`. | Query sum and document mean within each bucket. Documents always fill empty buckets. Length `Repetitions × 2**KSim × DProjections`. The reviewed encode path applies no final CountSketch. |

The Rust encoder's API docs still describe bucket assignment and a `buckets × dim` output. The code returns `buckets` values, and a zero-sized batch computes a zero chunk size for ndarray. The Python wrapper treats `"sum"` as max-pooling and falls back to mean for unrecognized strings. [Rust encoder](https://github.com/NewBornRustacean/muvera-rs/blob/6709414da4ce2e670184b500af573b6daf9a129c/src/encoder/fde_encoder.rs) · [Python binding](https://github.com/NewBornRustacean/muvera-rs/blob/6709414da4ce2e670184b500af573b6daf9a129c/src/python_bindings/mod.rs)

In the reviewed Weaviate encoder, calling `EncodeDoc` with an empty token slice can index row zero of that empty slice. Bucket counts use `uint16` and wrap at 65,536 tokens in a bucket. These are findings in the encoder itself. [Weaviate encoder](https://github.com/weaviate/weaviate/blob/5d678d7acf9e66f60b880ecaa1ce9952c1d4b530/adapters/repos/db/vector/multivector/muvera.go)

The implementations generate different vector spaces even when the output lengths match. Re-encode both queries and documents when switching between them.
