from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve relevant chunks
        results = self.store.search(question, top_k=top_k)

        # 2. Build context from retrieved chunks
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[{i}] {result['content']}")
        context = "\n\n".join(context_parts)

        # 3. Build prompt injecting the retrieved context
        prompt = (
            f"Answer the following question using ONLY the context provided below.\n"
            f"If the context does not contain enough information, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        # 4. Call the LLM
        return self.llm_fn(prompt)
