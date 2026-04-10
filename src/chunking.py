from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        # Split on sentence boundaries: ". ", "! ", "? ", or ".\n"
        sentences = re.split(r'(?<=[.!?])(?:\s|\n)', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return []
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case: text fits in one chunk
        if len(current_text) <= self.chunk_size:
            return [current_text] if current_text.strip() else []

        # Base case: no separators left — force-split by chunk_size
        if not remaining_separators:
            chunks: list[str] = []
            for i in range(0, len(current_text), self.chunk_size):
                piece = current_text[i : i + self.chunk_size]
                if piece.strip():
                    chunks.append(piece)
            return chunks

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]

        # Split with this separator
        if sep == "":
            # Character-level split (last resort)
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                piece = current_text[i : i + self.chunk_size]
                if piece.strip():
                    chunks.append(piece)
            return chunks

        parts = current_text.split(sep)

        # If this separator didn't split anything, try the next one
        if len(parts) <= 1:
            return self._split(current_text, next_seps)

        # Merge small parts and recurse on oversized ones
        result: list[str] = []
        current_group = parts[0]
        for part in parts[1:]:
            candidate = current_group + sep + part
            if len(candidate) <= self.chunk_size:
                current_group = candidate
            else:
                if current_group.strip():
                    if len(current_group) <= self.chunk_size:
                        result.append(current_group)
                    else:
                        result.extend(self._split(current_group, next_seps))
                current_group = part
        # Don't forget the last group
        if current_group.strip():
            if len(current_group) <= self.chunk_size:
                result.append(current_group)
            else:
                result.extend(self._split(current_group, next_seps))
        return result


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_product = _dot(vec_a, vec_b)
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot_product / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            'fixed_size': FixedSizeChunker(chunk_size=chunk_size, overlap=0).chunk(text),
            'by_sentences': SentenceChunker(max_sentences_per_chunk=3).chunk(text),
            'recursive': RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }
        result = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0
            result[name] = {
                'count': count,
                'avg_length': avg_length,
                'chunks': chunks,
            }
        return result
