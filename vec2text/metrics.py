from typing import Dict, List

import numpy as np
import scipy.stats
import torch

from vec2text.utils import get_embeddings_openai_vanilla

class EmbeddingCosineSimilarity:
    """Computes the cosine similarity between two lists of 
    string pairs using OpenAI ada-2 embeddings.
    """
    def __call__(self, s1: List[str], s2: List[str]) -> Dict[str, float]:
        e1 = torch.tensor(get_embeddings_openai_vanilla(s1), dtype=torch.float32)
        e2 = torch.tensor(get_embeddings_openai_vanilla(s2), dtype=torch.float32)
        sims = torch.nn.functional.cosine_similarity(e1, e2, dim=1)
        return {
            "ada_emb_cos_sim_mean": sims.mean().item(),
            "ada_emb_cos_sim_sem": scipy.stats.sem(sims.numpy()),
        }