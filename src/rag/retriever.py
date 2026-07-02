from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore


@dataclass
class RetrieverConfig:
    top_k: int = 5
    similarity_threshold: float = 0.3


def build_retriever(index: VectorStoreIndex, config: RetrieverConfig | None = None) -> VectorIndexRetriever:
    config = config or RetrieverConfig()
    return VectorIndexRetriever(index=index, similarity_top_k=config.top_k)


def retrieve(
    retriever: VectorIndexRetriever,
    query: str,
    threshold: float = 0.3,
) -> list[NodeWithScore]:
    nodes = retriever.retrieve(query)
    return [n for n in nodes if n.score is not None and n.score >= threshold]
