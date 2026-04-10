"""
Day 7 — Retrieval Experiment Script
Chạy thực nghiệm so sánh 3 chiến lược chunking trên bộ dữ liệu Y Tế BV Tâm Anh.
Sử dụng OpenAI text-embedding-3-small qua API.

Usage (PowerShell):
    $env:PYTHONIOENCODING="utf-8"; python run_experiment.py

Usage (Git Bash / Linux):
    PYTHONIOENCODING=utf-8 python run_experiment.py
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chiến lược chính của BẠN (dùng để in chi tiết top-3 và agent demo)
#   "recursive"  → RecursiveChunker(chunk_size=500)
#   "sentence"   → SentenceChunker(max_sentences_per_chunk=3)
#   "fixed"      → FixedSizeChunker(chunk_size=500, overlap=50)
CHUNKING_STRATEGY = "recursive"

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
        "filter": {"category": "Nội tiết - Đái tháo đường"},
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
        "filter": {"category": "Sản phụ khoa"},
    },
    {
        "query": "Ăn không tiêu thường xuyên có thể là dấu hiệu của những bệnh tiêu hóa nguy hiểm nào?",
        "gold_answer": "Viêm loét dạ dày, trào ngược dạ dày thực quản, viêm dạ dày, liệt dạ dày, thoát vị hoành, sỏi mật, viêm túi mật, viêm tụy, IBS, bệnh celiac, tắc ruột non, ung thư dạ dày.",
        "expected_source": "an-khong-tieu",
        "filter": {"category": "Tiêu hóa - Gan mật"},
    },
]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def make_chunker(strategy: str):
    """Trả về (chunker, tên hiển thị)."""
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=500), "RecursiveChunker(500)"
    elif strategy == "sentence":
        return SentenceChunker(max_sentences_per_chunk=3), "SentenceChunker(3 sent)"
    elif strategy == "fixed":
        return FixedSizeChunker(chunk_size=500, overlap=50), "FixedSizeChunker(500/50)"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


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


def _build_store(docs, chunker, embedder):
    """Tạo store cho 1 chunker, trả về (store, total_chunks)."""
    store = EmbeddingStore(collection_name="temp", embedding_fn=embedder)
    total = 0
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
        total += len(chunks)
    return store, total


def _eval_queries(store, queries):
    """Chạy benchmark, trả về list of {hit, score, doc, results}."""
    out = []
    for bq in queries:
        filt = bq.get("filter")
        if filt:
            results = store.search_with_filter(bq["query"], metadata_filter=filt, top_k=3)
        else:
            results = store.search(bq["query"], top_k=3)
        if results:
            top1 = results[0]
            doc_id = top1["metadata"].get("doc_id", top1.get("id", "?"))
            hit = bq["expected_source"] in doc_id
            out.append({"hit": hit, "score": top1["score"], "doc": doc_id, "results": results})
        else:
            out.append({"hit": False, "score": 0.0, "doc": "N/A", "results": []})
    return out


# ══════════════════════════════════════════════
# PHASE 1: Chunking Statistics
# ══════════════════════════════════════════════
def run_baseline_analysis(docs: list[Document]):
    """Exercise 3.1 — So sánh FixedSize vs Sentence vs Recursive."""
    print("\n")
    print("┌" + "─" * 68 + "┐")
    print("│  PHASE 1: SO SÁNH 3 CHIẾN LƯỢC CHUNKING (chunk_size=500)        │")
    print("└" + "─" * 68 + "┘")

    comparator = ChunkingStrategyComparator()

    print(f"\n  {'Tài liệu':<30s} │ {'Chiến lược':<15s} │ {'Chunks':>6s} │ {'Avg Len':>7s}")
    print("  " + "─" * 30 + "─┼─" + "─" * 15 + "─┼─" + "─" * 6 + "─┼─" + "─" * 7)

    for doc in docs:
        disease = doc.metadata["disease_name"]
        result = comparator.compare(doc.content, chunk_size=500)
        first = True
        for strategy_name, stats in result.items():
            label = disease if first else ""
            print(f"  {label:<30s} │ {strategy_name:<15s} │ {stats['count']:>6d} │ {stats['avg_length']:>7.0f}")
            first = False
        print("  " + "─" * 30 + "─┼─" + "─" * 15 + "─┼─" + "─" * 6 + "─┼─" + "─" * 7)


# ══════════════════════════════════════════════
# PHASE 2: Retrieval — So sánh 3 Chunker
# ══════════════════════════════════════════════
def run_retrieval_experiment(docs: list[Document], embedder):
    """Exercise 3.2 + 3.4 — So sánh 3 chunker trên cùng 5 queries."""
    strategies = [
        ("recursive", *make_chunker("recursive")),
        ("sentence",  *make_chunker("sentence")),
        ("fixed",     *make_chunker("fixed")),
    ]

    print("\n")
    print("┌" + "─" * 68 + "┐")
    print("│  PHASE 2: RETRIEVAL BENCHMARK — SO SÁNH 3 CHUNKER               │")
    print("└" + "─" * 68 + "┘")

    # --- Build 3 stores ---
    stores = {}
    print(f"\n  Indexing 7 tài liệu × 3 chiến lược...")
    for key, chunker, name in strategies:
        store, total = _build_store(docs, chunker, embedder)
        stores[key] = (store, name, total)
        print(f"    {name:<30s} → {total:>3d} chunks")

    # --- Evaluate all 3 ---
    all_evals = {}
    for key in ["recursive", "sentence", "fixed"]:
        store, name, total = stores[key]
        all_evals[key] = _eval_queries(store, BENCHMARK_QUERIES)

    # --- Chi tiết từng query cho chunker chính ---
    main_key = CHUNKING_STRATEGY
    main_eval = all_evals[main_key]
    main_name = stores[main_key][1]

    print(f"\n  {'─' * 66}")
    print(f"  CHI TIẾT: {main_name} (chiến lược chính)")
    print(f"  {'─' * 66}")

    for i, bq in enumerate(BENCHMARK_QUERIES):
        ev = main_eval[i]
        print(f"\n  ┌── Q{i+1} ──────────────────────────────────────────────────────")
        print(f"  │ Câu hỏi : {bq['query']}")
        print(f"  │ Đáp án  : {bq['gold_answer']}")
        print(f"  │ Nguồn   : {bq['expected_source']}")
        filt = bq.get("filter")
        print(f"  │ Filter  : {filt if filt else '(không)'}")
        print(f"  │")
        print(f"  │ Top-3 kết quả:")
        for rank, r in enumerate(ev["results"], 1):
            doc_id = r["metadata"].get("doc_id", r.get("id", "?"))
            is_rel = bq["expected_source"] in doc_id
            marker = "✅" if is_rel else "❌"
            snippet = r["content"][:90].replace("\n", " ").replace("\r", "")
            print(f"  │   {rank}. {marker}  score={r['score']:.4f}  doc={doc_id}")
            print(f'  │         "{snippet}..."')
        verdict = "✅ TOP-1 HIT" if ev["hit"] else "❌ TOP-1 MISS"
        print(f"  │")
        print(f"  └── Kết quả: {verdict}")

    # ── BẢNG SO SÁNH 3 CHUNKER ──
    print(f"\n\n  {'━' * 72}")
    print(f"  BẢNG SO SÁNH: TOP-1 SIMILARITY SCORE — 3 CHUNKER × 5 QUERIES")
    print(f"  {'━' * 72}")

    header = f"  {'Query':<8s} │ {'Recursive':>12s} │ {'Sentence':>12s} │ {'FixedSize':>12s} │ Winner"
    sep    = f"  {'─'*8}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*15}"
    print(f"\n{header}")
    print(sep)

    wins = {"recursive": 0, "sentence": 0, "fixed": 0}
    for i in range(len(BENCHMARK_QUERIES)):
        scores = {}
        for key in ["recursive", "sentence", "fixed"]:
            ev = all_evals[key][i]
            scores[key] = ev["score"]

        best_key = max(scores, key=lambda k: scores[k])
        wins[best_key] += 1
        winner_label = {"recursive": "Recursive", "sentence": "Sentence", "fixed": "FixedSize"}[best_key]

        def fmt(key):
            ev = all_evals[key][i]
            mark = "✅" if ev["hit"] else "❌"
            return f"{mark} {ev['score']:.4f}"

        print(f"  Q{i+1:<6d} │ {fmt('recursive'):>12s} │ {fmt('sentence'):>12s} │ {fmt('fixed'):>12s} │ ← {winner_label}")

    print(sep)

    # Tổng kết dòng
    def summary_row(label, fn):
        vals = []
        for key in ["recursive", "sentence", "fixed"]:
            vals.append(fn(key))
        print(f"  {label:<8s} │ {vals[0]:>12s} │ {vals[1]:>12s} │ {vals[2]:>12s} │")

    summary_row("Hits", lambda k: f"{sum(1 for e in all_evals[k] if e['hit'])}/5")
    summary_row("Avg Sc", lambda k: f"{sum(e['score'] for e in all_evals[k])/5:.4f}")
    summary_row("Wins", lambda k: f"{wins[k]}/5")

    best = max(wins, key=lambda k: wins[k])
    best_name = {"recursive": "RecursiveChunker", "sentence": "SentenceChunker", "fixed": "FixedSizeChunker"}[best]
    print(f"\n  🏆 Best Chunker by avg score wins: {best_name} ({wins[best]}/5 queries)")

    return stores[CHUNKING_STRATEGY][0]


# ══════════════════════════════════════════════
# PHASE 3: Similarity Predictions
# ══════════════════════════════════════════════
def run_similarity_predictions(embedder):
    """Exercise 3.3 — Dự đoán & đo Cosine Similarity."""
    print("\n")
    print("┌" + "─" * 68 + "┐")
    print("│  PHASE 3: SIMILARITY PREDICTIONS (5 cặp câu)                    │")
    print("└" + "─" * 68 + "┘")

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

    print(f"\n  {'#':<4s} │ {'Dự đoán':>7s} │ {'Thực tế':>7s} │ {'Score':>7s} │ {'':>3s} │ Câu A / Câu B")
    print("  " + "─" * 4 + "─┼─" + "─" * 7 + "─┼─" + "─" * 7 + "─┼─" + "─" * 7 + "─┼─" + "─" * 3 + "─┼─" + "─" * 40)

    correct = 0
    for i, (sent_a, sent_b, prediction) in enumerate(pairs, 1):
        emb_a = embedder(sent_a)
        emb_b = embedder(sent_b)
        score = compute_similarity(emb_a, emb_b)
        actual = "high" if score > 0.7 else "low" if score < 0.3 else "medium"
        is_match = prediction == actual
        icon = "✅" if is_match else "❌"
        if is_match:
            correct += 1
        print(f"  {i:<4d} │ {prediction:>7s} │ {actual:>7s} │ {score:>7.4f} │ {icon:>3s} │ {sent_a[:35]}...")
        print(f"  {'':4s} │ {'':7s} │ {'':7s} │ {'':7s} │ {'':3s} │ {sent_b[:35]}...")

    print(f"\n  Dự đoán đúng: {correct}/5 ({correct*20}%)")


# ══════════════════════════════════════════════
# PHASE 4: RAG Agent Demo
# ══════════════════════════════════════════════
def run_agent_demo(store, embedder):
    """Demo Agent trả lời câu hỏi dùng RAG."""
    print("\n")
    print("┌" + "─" * 68 + "┐")
    print("│  PHASE 4: RAG AGENT DEMO                                        │")
    print("└" + "─" * 68 + "┘")

    def demo_llm(prompt: str) -> str:
        lines = prompt.split("\n")
        context_lines = [l for l in lines if l.startswith("[")]
        return f"[DEMO] Dựa trên {len(context_lines)} chunks:\n{context_lines[0][:200] if context_lines else 'Không có context'}"

    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)

    for i, bq in enumerate(BENCHMARK_QUERIES[:2], 1):
        print(f"\n  Q{i}: {bq['query']}")
        answer = agent.answer(bq["query"], top_k=3)
        print(f"  Agent Response:")
        for line in answer.split("\n"):
            print(f"    │ {line[:100]}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  Day 7 — Medical RAG Retrieval Experiment                        ║")
    print("║  Domain : Bệnh học Y Tế (BV Đa khoa Tâm Anh)                    ║")
    print(f"║  Strategy: {CHUNKING_STRATEGY:<55s}  ║")
    print("╚" + "═" * 68 + "╝")

    # Load documents
    print(f"\n📂 Loading {len(MEDICAL_DOCS_METADATA)} medical documents...")
    docs = load_medical_docs()
    print(f"\n  → Loaded: {len(docs)} documents, total {sum(len(d.content) for d in docs):,} chars")

    # Get embedder
    print(f"\n🧠 Initializing embedder...")
    embedder = get_embedder()

    # Phase 1
    run_baseline_analysis(docs)

    # Phase 2
    store = run_retrieval_experiment(docs, embedder)

    # Phase 3
    run_similarity_predictions(embedder)

    # Phase 4
    run_agent_demo(store, embedder)

    # Done
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║  ✅ EXPERIMENT COMPLETE                                           ║")
    print("║  Copy kết quả ở trên vào file report/REPORT.md để nộp bài.       ║")
    print("╚" + "═" * 68 + "╝")
    print()


if __name__ == "__main__":
    main()
