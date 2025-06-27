import numpy as np
import mteb
import muvfde
import torch
import pandas as pd
import copy
from mteb.encoder_interface import PromptType
from mteb.models.wrapper import Wrapper
from pylate import models as colbert_model

# 1. Build two configs
query_config = muvfde.fixed_dimensional_encoding_config()
query_config.set_num_repetitions(20)
query_config.set_num_simhash_projections(5)
query_config.set_projection_dimension(8)
query_config.set_seed(1221)
query_config.set_encoding_type(muvfde.encoding_type.DEFAULT_SUM)
query_config.set_projection_type(muvfde.projection_type.AMS_SKETCH)

doc_config = muvfde.fixed_dimensional_encoding_config()
doc_config.set_num_repetitions(20)
doc_config.set_num_simhash_projections(5)
doc_config.set_projection_dimension(8)
doc_config.set_seed(1221)
doc_config.set_encoding_type(muvfde.encoding_type.AVERAGE)
doc_config.enable_fill_empty(True)
doc_config.set_projection_type(muvfde.projection_type.AMS_SKETCH)

# 2. Wrapper takes both configs
class FdeLateInteractionModel(Wrapper):
    def __init__(self, model_name: str, **kwargs):
        super().__init__()  # empty init for Wrapper
        self.colbert = colbert_model.ColBERT(model_name, revision=None, **kwargs)
        self.q_cfg = query_config
        self.d_cfg = doc_config

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        # 1) get token embeddings from the base ColBERT
        token_embs = self.colbert.encode(
            sentences,
            prompt_name=None,
            is_query=(prompt_type == PromptType.query),
            **kwargs,
        )
        # 2) pick the right config
        cfg = self.q_cfg if prompt_type == PromptType.query else self.d_cfg
        # 3) compress each [seq_len × dim] matrix → [fde_dim]
        fde_out = []
        for mat in token_embs:
            fde_out.append(
                muvfde.generate_fixed_dimensional_encoding(
                    mat.cpu().numpy().astype(np.float32), cfg
                )
            )
        return np.stack(fde_out, axis=0)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a @ b.T)

# 3. Instantiate & run
colbert_model_name = "colbert-ir/colbertv2.0"
lightonai_model_name = "lightonai/GTE-ModernColBERT-v1"
jina_model_name = "jinaai/jina-colbert-v2"
kwargs = {
    "trust_remote_code": True,
    "attend_to_expansion_tokens": True,
    "query_prefix": "[QueryMarker]",
    "document_prefix": "[DocumentMarker]",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_kwargs": {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
}
corpus_chunk_size = 32
colbert = mteb.get_model(colbert_model_name)
jina = mteb.get_model(jina_model_name, **kwargs)
fde_jina_model = FdeLateInteractionModel(jina_model_name, **kwargs)
fde_jina_model.mteb_model_meta = copy.deepcopy(jina.mteb_model_meta)
fde_jina_model.mteb_model_meta.name = "FDE Jina v2"
fde_jina_model.mteb_model_meta.revision = "no-revision"
fde_jina_model.mteb_model_meta.similarity_fn_name = "dot"
fde_jina_model.mteb_model_meta.framework = []

fde_model_colbert = FdeLateInteractionModel(colbert_model_name)
fde_model_colbert.mteb_model_meta = copy.deepcopy(colbert.mteb_model_meta)
fde_model_colbert.mteb_model_meta.name = "FDE ColBERT v2"
fde_model_colbert.mteb_model_meta.revision = "no-revision"
fde_model_colbert.mteb_model_meta.similarity_fn_name = "dot"
fde_model_colbert.mteb_model_meta.framework = []

tasks = mteb.get_tasks(tasks=["NFCorpus"], languages=["eng"], task_types=["Retrieval"])
evaluator = mteb.MTEB(tasks=tasks)

res1 = evaluator.run(colbert, eval_splits=["test"], corpus_chunk_size=corpus_chunk_size, verbosity=1)
res2 = evaluator.run(jina, eval_splits=["test"], corpus_chunk_size=corpus_chunk_size, verbosity=1)
res3 = evaluator.run(fde_jina_model,
                     eval_splits=["test"],
                     corpus_chunk_size=corpus_chunk_size,
                     verbosity=1)
res4 = evaluator.run(fde_model_colbert,
                     eval_splits=["test"],
                     corpus_chunk_size=corpus_chunk_size,
                     verbosity=1)

colbert_scores = res1[0].scores['test'][0]
jina_scores = res2[0].scores['test'][0]
fde_jina_scores = res3[0].scores['test'][0]
fde_colbert_scores = res4[0].scores['test'][0]

df = pd.DataFrame({
    'Metric': list(colbert_scores.keys()),
    'ColBERT v2': list(colbert_scores.values()),
    'FDE ColBERT v2': [fde_colbert_scores[k] for k in colbert_scores.keys()],
    'Jina ColBERT v2': [jina_scores[k] for k in colbert_scores.keys()],
    'FDE Jina v2': [fde_jina_scores[k] for k in colbert_scores.keys()],
})
# Show only .5f
df.to_markdown("integeration_test/benchmark.md", index=False, tablefmt="github")
