from __future__ import annotations

from functools import lru_cache

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


EMBEDDING_MODELS = {
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


@lru_cache(maxsize=4)
def get_embedding_model(name: str = "bge-base") -> BaseEmbedding:
    model_id = EMBEDDING_MODELS.get(name, name)
    return HuggingFaceEmbedding(model_name=model_id)
