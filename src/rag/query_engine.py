from __future__ import annotations

from dataclasses import dataclass, field

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from src.rag.retriever import RetrieverConfig, build_retriever


@dataclass
class QueryResult:
    answer: str
    source_nodes: list = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)

    @property
    def top_score(self) -> float:
        return max(self.retrieval_scores, default=0.0)


def build_query_engine(
    index: VectorStoreIndex,
    llm=None,
    retriever_config: RetrieverConfig | None = None,
) -> RetrieverQueryEngine:
    retriever = build_retriever(index, retriever_config)
    synthesizer = get_response_synthesizer(llm=llm, response_mode="compact")
    return RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer)


def query(engine: RetrieverQueryEngine, question: str) -> QueryResult:
    response = engine.query(question)
    scores = [n.score or 0.0 for n in response.source_nodes]
    return QueryResult(
        answer=str(response),
        source_nodes=response.source_nodes,
        retrieval_scores=scores,
    )
