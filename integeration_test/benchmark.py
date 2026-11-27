"""Evaluate FDE retrieval and MaxSim reranking on the NFCorpus test split."""

import argparse
import csv
from importlib.metadata import version

from huggingface_hub import hf_hub_download
import muvfde
import numpy as np
from pyarrow.parquet import read_table
from sentence_transformers import MultiVectorEncoder


MODEL = "lightonai/LateOn-regularized"
MODEL_REVISION = "4449aa20f8ca4b580b8ab46df7599fa392e9b076"
DATASET_REVISION = "b5026a0e96e8a7ac4f95f482a596389289d46269"
QRELS_REVISION = "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65"


def load_nfcorpus():
    corpus_path = hf_hub_download(
        "BeIR/nfcorpus", "corpus/corpus-00000-of-00001.parquet",
        repo_type="dataset", revision=DATASET_REVISION,
    )
    queries_path = hf_hub_download(
        "BeIR/nfcorpus", "queries/queries-00000-of-00001.parquet",
        repo_type="dataset", revision=DATASET_REVISION,
    )
    judgments_path = hf_hub_download(
        "BeIR/nfcorpus-qrels", "test.tsv", repo_type="dataset", revision=QRELS_REVISION
    )
    corpus = read_table(corpus_path).to_pylist()
    query_text = {row["_id"]: row["text"] for row in read_table(queries_path).to_pylist()}
    with open(judgments_path, newline="", encoding="utf-8") as file:
        judgments = list(csv.DictReader(file, delimiter="\t"))

    query_ids = sorted({row["query-id"] for row in judgments})
    document_indices = {row["_id"]: i for i, row in enumerate(corpus)}
    query_indices = {query_id: i for i, query_id in enumerate(query_ids)}
    relevance = np.zeros((len(query_ids), len(corpus)), dtype=np.float32)
    for row in judgments:
        relevance[query_indices[row["query-id"]], document_indices[row["corpus-id"]]] = int(row["score"])
    documents = [f'{row["title"] or ""} {row["text"]}'.strip() for row in corpus]
    return [query_text[query_id] for query_id in query_ids], documents, relevance


def encode_fdes(embeddings, *, document, center):
    config = (
        muvfde.fixed_dimensional_encoding_config()
        .set_num_repetitions(20)
        .set_num_simhash_projections(5)
        .set_projection_type(muvfde.projection_type.AMS_SKETCH)
        .set_projection_dimension(16)
        .set_seed(42)
    )
    if document:
        config.set_encoding_type(muvfde.encoding_type.AVERAGE).enable_fill_empty()
    return np.stack([
        muvfde.generate_fixed_dimensional_encoding(embedding - center, config)
        for embedding in embeddings
    ])


def ndcg_at_10(ranking, relevance):
    gains = np.take_along_axis(relevance, ranking[:, :10], axis=1)
    ideal = np.sort(relevance, axis=1)[:, -gains.shape[1]:][:, ::-1]
    discounts = 1 / np.log2(np.arange(2, gains.shape[1] + 2))
    ideal_dcg = ideal @ discounts
    dcg = gains @ discounts
    return np.divide(dcg, ideal_dcg, out=np.zeros_like(dcg), where=ideal_dcg > 0).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--device", help="e.g. cpu or cuda (default: auto-detect)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size < 1 or args.candidates < 10:
        parser.error("--batch-size must be positive and --candidates must be at least 10")

    queries, documents, relevance = load_nfcorpus()
    if args.candidates > len(documents):
        parser.error("--candidates cannot exceed the corpus size")
    revision = args.revision or (MODEL_REVISION if args.model == MODEL else None)
    model = MultiVectorEncoder(
        args.model, revision=revision, device=args.device, similarity_fn_name="maxsim"
    )
    encode_options = dict(batch_size=args.batch_size, convert_to_numpy=True, show_progress_bar=True)
    queries = [np.asarray(x, dtype=np.float32) for x in model.encode_query(queries, **encode_options)]
    documents = [np.asarray(x, dtype=np.float32) for x in model.encode_document(documents, **encode_options)]

    print("Scoring full-corpus MaxSim...", flush=True)
    maxsim = np.concatenate([
        model.similarity(queries[start:start + 8], documents, device=str(model.device)).cpu().numpy()
        for start in range(0, len(queries), 8)
    ])
    exact_top_10 = np.argsort(-maxsim, axis=1, kind="stable")[:, :10]
    results = [("Full-corpus MaxSim", ndcg_at_10(exact_top_10, relevance), None)]

    # Use the same corpus token mean for documents and queries during FDE generation.
    center = (
        sum(document.sum(axis=0, dtype=np.float64) for document in documents)
        / sum(len(document) for document in documents)
    ).astype(np.float32)
    for label, mean in [("Raw", np.float32(0)), ("Mean-centered", center)]:
        print(f"Scoring {label.lower()} FDEs...", flush=True)
        query_fdes = encode_fdes(queries, document=False, center=mean)
        document_fdes = encode_fdes(documents, document=True, center=mean)
        candidates = np.argsort(-(query_fdes @ document_fdes.T), axis=1, kind="stable")[:, :args.candidates]
        overlap = np.mean([
            np.isin(exact_row, candidate_row)
            for exact_row, candidate_row in zip(exact_top_10, candidates, strict=True)
        ])
        # Reuse the exact scores for candidate documents to measure ranking quality.
        candidate_scores = np.take_along_axis(maxsim, candidates, axis=1)
        reranked = np.take_along_axis(
            candidates, np.argsort(-candidate_scores, axis=1, kind="stable"), axis=1
        )
        results.append((f"{label} FDE dot product", ndcg_at_10(candidates, relevance), overlap))
        results.append((f"{label} FDE + MaxSim", ndcg_at_10(reranked, relevance), overlap))

    print(
        f"\n{args.model} @ {revision or 'default revision'}, "
        f"{len(queries)} queries, {len(documents)} documents"
    )
    print(f"muvfde {version('muvfde')}, sentence-transformers {version('sentence-transformers')}")
    print(f"\n| Ranking | nDCG@10 | MaxSim top-10 overlap in FDE top-{args.candidates} |")
    print("| --- | ---: | ---: |")
    for label, ndcg, overlap in results:
        overlap_text = "—" if overlap is None else f"{overlap:.1%}"
        print(f"| {label} | {ndcg:.4f} | {overlap_text} |")


if __name__ == "__main__":
    main()
