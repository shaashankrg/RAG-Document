from __future__ import annotations

from pathlib import Path

import faiss
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore

from src.rag.embeddings import get_embedding_model


def build_index(
    nodes: list[TextNode],
    persist_dir: str | Path | None = None,
    embedding_model: str = "bge-base",
) -> VectorStoreIndex:
    embed_model = get_embedding_model(embedding_model)
    dim = len(embed_model.get_text_embedding("probe"))

    faiss_index = faiss.IndexFlatIP(dim)  # inner-product = cosine on normalised vecs
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_ctx,
        embed_model=embed_model,
        show_progress=True,
    )

    if persist_dir:
        index.storage_context.persist(persist_dir=str(persist_dir))

    return index


def load_index(persist_dir: str | Path, embedding_model: str = "bge-base") -> VectorStoreIndex:
    from llama_index.core import load_index_from_storage
    embed_model = get_embedding_model(embedding_model)
    vector_store = FaissVectorStore.from_persist_dir(str(persist_dir))
    storage_ctx = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=str(persist_dir),
    )
    return load_index_from_storage(storage_ctx, embed_model=embed_model)
