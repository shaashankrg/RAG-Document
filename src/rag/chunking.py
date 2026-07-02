from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.schema import Document, TextNode


@dataclass
class ChunkConfig:
    strategy: str = "sentence"   # "sentence" | "semantic" | "fixed"
    chunk_size: int = 512
    chunk_overlap: int = 64


def build_nodes(
    texts: Sequence[str],
    metadata_list: Sequence[dict] | None = None,
    config: ChunkConfig | None = None,
) -> list[TextNode]:
    config = config or ChunkConfig()
    docs = [
        Document(text=t, metadata=meta or {})
        for t, meta in zip(texts, metadata_list or [{}] * len(texts))
    ]

    if config.strategy == "sentence":
        parser = SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    elif config.strategy == "fixed":
        from llama_index.core.node_parser import SimpleNodeParser
        parser = SimpleNodeParser.from_defaults(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {config.strategy}")

    return parser.get_nodes_from_documents(docs)
