"""
Day 7 — Retrieval Experiment Script
Chạy thực nghiệm so sánh các chiến lược chunking trên bộ dữ liệu Y Tế BV Tâm Anh.
Sử dụng OpenAI text-embedding-3-small qua API.

Usage:
    python run_experiment.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from src.chunking import (
    ChunkingStrategyComparator,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    compute_similarity,
)
from src.embeddings import OpenAIEmbedder, _mock_embed
from src.models import Document
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent

# ──────────────────────────────────────────────
# 1. Metadata cho 7 file y tế
# ──────────────────────────────────────────────
MEDICAL_DOCS_METADATA = {
    "alzheimer": {
        "disease_name": "Bệnh Alzheimer",
        "category": "Thần kinh",
        "source": "BV Tâm Anh",
    },
    "an-khong-tieu": {
        "disease_name": "Ăn không tiêu",
        "category": "Tiêu hóa - Gan mật",
        "source": "BV Tâm Anh",
    },
    "ap-xe-hau-mon": {
        "disease_name": "Áp xe hậu môn",
        "category": "Tiêu hóa - Hậu môn trực tràng",
        "source": "BV Tâm Anh",
    },
    "ap-xe-phoi": {
        "disease_name": "Áp xe phổi",
        "category": "Hô hấp",
        "source": "BV Tâm Anh",
    },
    "ban-chan-dai-thao-duong": {
        "disease_name": "Bàn chân đái tháo đường",
        "category": "Nội tiết - Đái tháo đường",
        "source": "BV Tâm Anh",
    },
    "bang-huyet-sau-sinh": {
        "disease_name": "Băng huyết sau sinh",
        "category": "Sản phụ khoa",
        "source": "BV Tâm Anh",
    },
    "bang-quang-tang-hoat": {
        "disease_name": "Bàng quang tăng hoạt",
        "category": "Tiết niệu",
        "source": "BV Tâm Anh",
    },
}

# ──────────────────────────────────────────────
# 2. Benchmark Queries & Gold Answers
# ──────────────────────────────────────────────
BENCHMARK_QUERIES = [
    {
        "query": "Bệnh Alzheimer được chẩn đoán chia giai đoạn theo thang điểm MMSE như thế nào?",
        "gold_answer": "Alzheimer nhẹ: MMSE 21-26, trung bình: 10-20, trung bình nặng: 10-14, nặng: dưới 10.",
        "expected_source": "alzheimer",
    },
    {
        "query": "Các biện pháp cận lâm sàng nào giúp đánh giá tình trạng mạch máu và tưới máu bàn chân ở người bệnh đái tháo đường?",
        "gold_answer": "Sử dụng các biện pháp đo ABI (Ankle Brachial Index) và đo TcPO2 (transcutaneous oxygen tension).",
        "expected_source": "ban-chan-dai-thao-duong",
    },
    {
        "query": "Thời gian dùng kháng sinh điều trị áp xe phổi nguyên phát thường kéo dài bao lâu?",
        "gold_answer": "Thời gian dùng kháng sinh kéo dài từ 4 đến 6 tuần hoặc cho đến khi X-quang ngực sạch hoặc chỉ còn vết sẹo nhỏ.",
        "expected_source": "ap-xe-phoi",
    },
    {
        "query": "Nguyên nhân hàng đầu gây ra băng huyết sau sinh là gì?",
        "gold_answer": "Nguyên nhân hàng đầu (hay gặp nhất) gây băng huyết sau sinh là đờ tử cung.",
        "expected_source": "bang-huyet-sau-sinh",
    },
    {
        "query": "Ăn không tiêu thường xuyên có thể là dấu hiệu của những bệnh tiêu hóa nguy hiểm nào?",
        "gold_answer": "Viêm loét dạ dày, trào ngược dạ dày thực quản, viêm dạ dày, liệt dạ dày, thoát vị hoành, sỏi mật, viêm túi mật, viêm tụy, IBS, bệnh celiac, tắc ruột non, ung thư dạ dày.",
        "expected_source": "an-khong-tieu",
    },
]


def load_medical_docs() -> list[Document]:
    """Load & gán metadata cho 7 file y tế đã clean."""
    data_dir = Path("data")
    docs: list[Document] = []
    for filename, meta in MEDICAL_DOCS_METADATA.items():
        filepath = data_dir / f"{filename}.md"
        if not filepath.exists():
            print(f"  ⚠ File not found: {filepath}")
            continue
        content = filepath.read_text(encoding="utf-8")
        docs.append(Document(id=filename, content=content, metadata=meta))
        print(f"  ✓ {filename}.md ({len(content):,} chars) — {meta['disease_name']} [{meta['category']}]")
    return docs


def get_embedder():
    """Chọn embedder dựa trên .env"""
    provider = os.getenv("EMBEDDING_PROVIDER", "mock").strip().lower()
    if provider == "openai":
        try:
            model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            embedder = OpenAIEmbedder(model_name=model)
            print(f"  → Embedding: OpenAI ({model})")
            return embedder
        except Exception as e:
            print(f"  ⚠ OpenAI failed ({e}), falling back to mock")
    print("  → Embedding: Mock (deterministic)")
    return _mock_embed


def run_baseline_analysis(docs: list[Document]):
    """Exercise 3.1 Step 1 — Baseline chunking comparison."""
    print("\n" + "=" * 70)
    print("PHASE 1: BASELINE CHUNKING ANALYSIS")
    print("=" * 70)

    comparator = ChunkingStrategyComparator()
    # Chọn 3 tài liệu đại diện
    sample_docs = docs[:3]

    for doc in sample_docs:
        print(f"\n--- {doc.metadata['disease_name']} ({len(doc.content):,} chars) ---")
        result = comparator.compare(doc.content, chunk_size=500)
        for strategy_name, stats in result.items():
            print(f"  {strategy_name:15s} | chunks: {stats['count']:3d} | avg_len: {stats['avg_length']:.0f}")


def run_retrieval_experiment(docs: list[Document], embedder):
    """Exercise 3.2 + 3.4 — Chạy benchmark queries với OpenAI embeddings."""
    print("\n" + "=" * 70)
    print("PHASE 2: RETRIEVAL EXPERIMENT (OpenAI Embeddings)")
    print("=" * 70)

    # Dùng RecursiveChunker cho experiment chính
    chunker = RecursiveChunker(chunk_size=500)

    # Tạo store và add chunks
    store = EmbeddingStore(collection_name="medical_rag", embedding_fn=embedder)

    chunk_count = 0
    for doc in docs:
        chunks = chunker.chunk(doc.content)
        chunk_docs = []
        for i, chunk_text in enumerate(chunks):
            chunk_docs.append(Document(
                id=f"{doc.id}_chunk_{i}",
                content=chunk_text,
                metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": i},
            ))
        store.add_documents(chunk_docs)
        chunk_count += len(chunks)
        print(f"  ✓ {doc.id}: {len(chunks)} chunks")

    print(f"\n  Total chunks in store: {store.get_collection_size()}")

    # Chạy 5 benchmark queries
    print("\n" + "-" * 70)
    print("BENCHMARK QUERIES — Top-3 Results")
    print("-" * 70)

    relevant_count = 0
    for i, bq in enumerate(BENCHMARK_QUERIES, 1):
        print(f"\n  Q{i}: {bq['query']}")
        print(f"  Gold: {bq['gold_answer'][:100]}...")
        print(f"  Expected source: {bq['expected_source']}")

        results = store.search(bq["query"], top_k=3)
        hit = False
        for rank, r in enumerate(results, 1):
            doc_id = r["metadata"].get("doc_id", r.get("id", "?"))
            is_relevant = bq["expected_source"] in doc_id
            marker = "✓" if is_relevant else "✗"
            if is_relevant and rank == 1:
                hit = True
            print(f"    {rank}. [{marker}] score={r['score']:.4f} doc={doc_id}")
            print(f"       {r['content'][:120].replace(chr(10), ' ')}...")

        if hit:
            relevant_count += 1
            print(f"  → ✓ TOP-1 RELEVANT")
        else:
            print(f"  → ✗ TOP-1 MISS")

    print(f"\n{'=' * 70}")
    print(f"RETRIEVAL PRECISION: {relevant_count}/5 queries with relevant top-1 chunk")
    print(f"{'=' * 70}")

    return store


def run_similarity_predictions(embedder):
    """Exercise 3.3 — Cosine Similarity Predictions."""
    print("\n" + "=" * 70)
    print("PHASE 3: SIMILARITY PREDICTIONS")
    print("=" * 70)

    pairs = [
        ("Bệnh Alzheimer gây mất trí nhớ ở người cao tuổi",
         "Sa sút trí tuệ khiến người già hay quên",
         "high"),
        ("Triệu chứng đau bụng sau khi ăn",
         "Ăn không tiêu gây khó chịu vùng thượng vị",
         "high"),
        ("Bàng quang tăng hoạt gây tiểu gấp",
         "Phương pháp nấu phở bò truyền thống",
         "low"),
        ("Áp xe phổi điều trị bằng kháng sinh",
         "Áp xe hậu môn cần phẫu thuật dẫn lưu",
         "medium"),
        ("Băng huyết sau sinh do đờ tử cung",
         "Bàn chân đái tháo đường bị loét",
         "low"),
    ]

    for i, (sent_a, sent_b, prediction) in enumerate(pairs, 1):
        emb_a = embedder(sent_a)
        emb_b = embedder(sent_b)
        score = compute_similarity(emb_a, emb_b)
        actual = "high" if score > 0.7 else "low" if score < 0.3 else "medium"
        match = "✓" if prediction == actual else "✗"
        print(f"  Pair {i} [{match}]: prediction={prediction}, actual={actual} (score={score:.4f})")
        print(f"    A: {sent_a}")
        print(f"    B: {sent_b}")


def run_agent_demo(store, embedder):
    """Demo Agent trả lời câu hỏi dùng RAG."""
    print("\n" + "=" * 70)
    print("PHASE 4: RAG AGENT DEMO")
    print("=" * 70)

    def demo_llm(prompt: str) -> str:
        """Simple echo LLM — In production, gọi OpenAI GPT tại đây."""
        # Trích context từ prompt
        lines = prompt.split("\n")
        context_lines = [l for l in lines if l.startswith("[")]
        return f"[DEMO] Based on {len(context_lines)} retrieved chunks: {context_lines[0][:200] if context_lines else 'No context'}..."

    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)

    for i, bq in enumerate(BENCHMARK_QUERIES[:2], 1):
        print(f"\n  Q{i}: {bq['query']}")
        answer = agent.answer(bq["query"], top_k=3)
        print(f"  Agent: {answer[:200]}...")


def main():
    print("=" * 70)
    print("Day 7 — Medical RAG Retrieval Experiment")
    print("Domain: Bệnh học Y Tế (BV Đa khoa Tâm Anh)")
    print("=" * 70)

    # Load documents
    print("\n📂 Loading medical documents...")
    docs = load_medical_docs()
    print(f"\n  Total: {len(docs)} documents loaded")

    # Get embedder
    print("\n🧠 Initializing embedder...")
    embedder = get_embedder()

    # Phase 1: Baseline
    run_baseline_analysis(docs)

    # Phase 2: Retrieval
    store = run_retrieval_experiment(docs, embedder)

    # Phase 3: Similarity
    run_similarity_predictions(embedder)

    # Phase 4: Agent demo
    run_agent_demo(store, embedder)

    print("\n\n✅ Experiment complete! Use results above to fill REPORT.md")


if __name__ == "__main__":
    main()
