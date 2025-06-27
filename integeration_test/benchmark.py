import numpy as np
import mteb
import muvfde
import torch
import pandas as pd
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
colbert = mteb.get_model("colbert-ir/colbertv2.0")
fde_model = FdeLateInteractionModel("colbert-ir/colbertv2.0")

tasks = mteb.get_tasks(tasks=["NFCorpus"], languages=["eng"], task_types=["Retrieval"])
evaluator = mteb.MTEB(tasks=tasks)

# raw ColBERT
res1 = evaluator.run(colbert, eval_splits=["test"], corpus_chunk_size=1024, verbosity=1)

# FDE compressed
res2 = evaluator.run(fde_model,
                     eval_splits=["test"],
                     corpus_chunk_size=1024,
                     verbosity=3)

raw_scores = res1[0].scores['test'][0]
fde_scores = res2[0].scores['test'][0]

df = pd.DataFrame({
    'Metric': list(raw_scores.keys()),
    'Raw ColBERT': list(raw_scores.values()),
    'FDE ColBERT': [fde_scores[k] for k in raw_scores.keys()]
})
print(df.to_markdown(index=False))
