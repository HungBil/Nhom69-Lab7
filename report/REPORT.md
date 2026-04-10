# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Đông Hưng
**MSSV:** 2A202600392
**Nhóm:** Nhóm 69
**Ngày:** 2026-04-10

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai đoạn văn bản có cosine similarity cao (gần 1.0) nghĩa là vector embedding của chúng gần như cùng hướng trong không gian nhiều chiều — tức là chúng mang ý nghĩa ngữ nghĩa tương đồng, dù có thể dùng từ ngữ khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Bệnh Alzheimer gây mất trí nhớ ở người cao tuổi"
- Sentence B: "Sa sút trí tuệ khiến người già hay quên"
- Tại sao tương đồng: Cả hai đều nói về cùng một hiện tượng (suy giảm trí nhớ ở người già) chỉ khác cách diễn đạt — "Alzheimer" vs "sa sút trí tuệ", "mất trí nhớ" vs "hay quên". Embedding model hiểu được sự tương đương ngữ nghĩa này.

**Ví dụ LOW similarity:**
- Sentence A: "Bàng quang tăng hoạt gây tiểu gấp"
- Sentence B: "Phương pháp nấu phở bò truyền thống"
- Tại sao khác: Hai câu thuộc hai domain hoàn toàn khác biệt (y tế vs ẩm thực), không chia sẻ bất kỳ khái niệm ngữ nghĩa nào. Actual score đo được: 0.2291 (low).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo **hướng** của vector, không phụ thuộc vào **độ dài** (magnitude). Hai đoạn văn có cùng ý nghĩa nhưng độ dài khác nhau sẽ có embedding vector khác magnitude nhưng cùng hướng — cosine similarity vẫn cho điểm cao, trong khi Euclidean distance sẽ bị ảnh hưởng bởi sự khác biệt magnitude đó.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> = ceil((10000 - 50) / (500 - 50))
> = ceil(9950 / 450)
> = ceil(22.11)
> **= 23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = **25 chunks** (tăng thêm 2 chunks).
> Overlap nhiều hơn giúp bảo toàn ngữ cảnh tại ranh giới giữa các chunk — một câu bị cắt đôi ở chunk trước sẽ xuất hiện đầy đủ ở chunk sau, tránh mất thông tin quan trọng khi retrieval.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Bệnh học Y Tế (BV Đa khoa Tâm Anh)

**Tại sao nhóm chọn domain này?**
> Dữ liệu y tế có cấu trúc rõ ràng (triệu chứng → chẩn đoán → điều trị), phân loại chuyên khoa tự nhiên (Thần kinh, Hô hấp, Tiêu hóa...), và yêu cầu độ chính xác cao khi retrieval — rất phù hợp để đánh giá hiệu quả của metadata filtering. Ngoài ra, nguồn dữ liệu tiếng Việt từ BV Tâm Anh công khai, chất lượng cao, dễ verify gold answers.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | alzheimer.md | BV Tâm Anh | 27,190 | disease_name, category=Thần kinh, source |
| 2 | an-khong-tieu.md | BV Tâm Anh | 12,178 | disease_name, category=Tiêu hóa - Gan mật, source |
| 3 | ap-xe-hau-mon.md | BV Tâm Anh | 9,534 | disease_name, category=Tiêu hóa - Hậu môn trực tràng, source |
| 4 | ap-xe-phoi.md | BV Tâm Anh | 14,339 | disease_name, category=Hô hấp, source |
| 5 | ban-chan-dai-thao-duong.md | BV Tâm Anh | 12,587 | disease_name, category=Nội tiết - Đái tháo đường, source |
| 6 | bang-huyet-sau-sinh.md | BV Tâm Anh | 13,386 | disease_name, category=Sản phụ khoa, source |
| 7 | bang-quang-tang-hoat.md | BV Tâm Anh | 12,484 | disease_name, category=Tiết niệu, source |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| disease_name | string | "Bệnh Alzheimer" | Cho phép lọc chính xác theo tên bệnh, tránh nhầm lẫn giữa các bệnh có triệu chứng tương tự |
| category | string | "Thần kinh" | Phân loại theo chuyên khoa — là trường filter chính, giúp thu hẹp không gian tìm kiếm đáng kể |
| source | string | "BV Tâm Anh" | Truy xuất nguồn gốc dữ liệu, hữu ích khi mở rộng sang nhiều nguồn khác nhau |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên tất cả 7 tài liệu (chunk_size=500):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Bệnh Alzheimer (27,190) | fixed_size | 55 | 494 | Thấp — cắt giữa câu |
| | by_sentences | 74 | 366 | Cao — giữ nguyên câu |
| | recursive | 73 | 371 | Cao — ưu tiên ranh giới tự nhiên |
| Ăn không tiêu (12,178) | fixed_size | 25 | 487 | Thấp |
| | by_sentences | 41 | 295 | Cao |
| | recursive | 33 | 368 | Cao |
| Áp xe hậu môn (9,534) | fixed_size | 20 | 477 | Thấp |
| | by_sentences | 26 | 365 | Cao |
| | recursive | 27 | 352 | Cao |

### Strategy Của Tôi

**Loại:** RecursiveChunker (chunk_size=500)

**Mô tả cách hoạt động:**
> RecursiveChunker thử lần lượt các separator theo thứ tự ưu tiên: `["\n\n", "\n", ". ", " ", ""]`. Đầu tiên, nó tách văn bản theo paragraph (`\n\n`). Nếu một đoạn vẫn quá dài (>500 chars), nó đệ quy xuống separator tiếp theo (`\n`, rồi `.`, rồi space, cuối cùng là character-level). Các đoạn nhỏ liên tiếp được gộp lại cho đến khi gần đầy chunk_size. Kết quả là các chunk tôn trọng ranh giới tự nhiên của văn bản (paragraph > dòng > câu).

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu y tế BV Tâm Anh có cấu trúc rõ ràng theo các heading Markdown (`##`, `###`) và paragraph. RecursiveChunker khai thác cấu trúc này bằng cách ưu tiên tách tại `\n\n` (ranh giới paragraph/section), giữ nguyên các khối nội dung y tế có liên quan với nhau (ví dụ: toàn bộ phần "Triệu chứng" nằm trong 1-2 chunks thay vì bị cắt ngang).

### So Sánh: Strategy của tôi vs Baseline

Chạy cả 3 chunker trên cùng 5 benchmark queries với cùng OpenAI `text-embedding-3-small`:

| Query | Recursive (score) | Sentence (score) | FixedSize (score) | Winner |
|-------|-------------------|------------------|-------------------|--------|
| Q1 — Alzheimer MMSE | ✅ 0.7558 | ✅ 0.7570 | ✅ **0.7760** | FixedSize |
| Q2 — Bàn chân ĐTĐ | ✅ **0.7512** | ✅ 0.7367 | ✅ 0.7161 | Recursive |
| Q3 — Áp xe phổi KS | ✅ 0.6764 | ✅ **0.7070** | ✅ 0.6616 | Sentence |
| Q4 — Băng huyết | ✅ 0.6171 | ✅ **0.6493** | ✅ 0.6218 | Sentence |
| Q5 — Ăn không tiêu | ✅ 0.5661 | ✅ **0.5677** | ✅ 0.5414 | Sentence |
| **Avg Score** | **0.6733** | **0.6835** | **0.6634** | |
| **Wins** | 1/5 | **3/5** 🏆 | 1/5 | |
| **Hits** | 5/5 | 5/5 | 5/5 | |

**Nhận xét:** Cả 3 strategy đều đạt 5/5 hits (tìm đúng tài liệu nguồn) vì OpenAI embedding đã đủ mạnh. Tuy nhiên, **SentenceChunker cho similarity score trung bình cao nhất** (0.6835) vì nó giữ nguyên ranh giới câu — mỗi chunk là một đơn vị ngữ nghĩa hoàn chỉnh.

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Embedding Model | Vector DB | Precision (no filter) | Precision (filtered) |
|-----------|----------|-----------------|-----------|----------------------|---------------------|
| Nguyễn Đông Hưng (Tôi) | RecursiveChunker(500) | OpenAI text-embedding-3-small | In-memory | **100%** (5/5) | **100%** (5/5) |
| Khuất Văn Vương | RecursiveChunker(500) | Qwen 0.8B (local) | ChromaDB | **95.2%** | **100%** |
| Lưu Lương Vi Nhân | Recursive(400) | all-MiniLM-L6-v2 | ChromaDB | **66.8%** | **100%** |
| Huỳnh Văn Nghĩa | SentenceChunker(500) | GPT-4o-mini | ChromaDB | **9.5%** | **100%** |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Kết quả nhóm cho thấy **embedding model quan trọng hơn chunking strategy**. Cùng dùng RecursiveChunker nhưng: OpenAI embedding đạt 100%, Qwen 0.8B đạt 95.2%, MiniLM đạt 66.8%. Tuy nhiên, **Metadata Filter là "lưới an toàn" cực kỳ hiệu quả** — nó kéo tất cả mọi cấu hình lên 100%, kể cả trường hợp Nghĩa chỉ đạt 9.5% khi không filter. Đối với domain y tế có phân loại chuyên khoa rõ ràng, kết hợp RecursiveChunker + metadata filtering là lựa chọn tối ưu.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex lookbehind `(?<=[.!?])(?:\s|\n)` để tách câu tại các dấu kết thúc câu (`.`, `!`, `?`) theo sau bởi whitespace hoặc newline. Sau đó gộp mỗi `max_sentences_per_chunk` câu thành 1 chunk bằng `" ".join()`. Xử lý edge case: text rỗng hoặc chỉ có whitespace trả về `[]`, strip các câu trống sau khi split.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán đệ quy: base case là text ≤ chunk_size (trả về nguyên), hoặc hết separator (force-split theo chunk_size). Recursive case: split text bằng separator hiện tại, gộp các phần nhỏ liên tiếp lại cho đến khi gần đầy chunk_size, các phần vượt quá thì đệ quy xuống separator tiếp theo. Thứ tự separator `["\n\n", "\n", ". ", " ", ""]` đảm bảo ưu tiên ranh giới ngữ nghĩa tự nhiên.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents`: Mỗi document được embed bằng `embedding_fn(doc.content)`, rồi lưu dưới dạng dict `{id, content, embedding, metadata}` vào `self._store` (list in-memory). Nếu có ChromaDB thì dùng `collection.add()`.
> `search`: Embed query, tính dot product với tất cả embeddings trong store, sort giảm dần theo score, trả về top_k kết quả.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter`: Filter **trước**, search **sau** — lọc `self._store` chỉ giữ records mà `metadata[key] == value` cho mọi key trong filter dict, rồi chạy similarity search trên tập đã lọc. Cách này giảm không gian tìm kiếm, đảm bảo kết quả luôn thuộc đúng category.
> `delete_document`: So sánh `doc_id` với từng record, giữ lại các record không khớp bằng list comprehension. Trả về `True` nếu size giảm.

### KnowledgeBaseAgent

**`answer`** — approach:
> (1) Gọi `store.search(query, top_k)` để lấy top-k chunks liên quan nhất. (2) Xây dựng prompt theo cấu trúc: mỗi chunk được format thành `[{i}] {content}` rồi nối thành context block. (3) Ghép context + câu hỏi gốc thành prompt hoàn chỉnh, gọi `llm_fn(prompt)` và trả về kết quả.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.08s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Bệnh Alzheimer gây mất trí nhớ ở người cao tuổi | Sa sút trí tuệ khiến người già hay quên | high | 0.5460 (medium) | ❌ |
| 2 | Triệu chứng đau bụng sau khi ăn | Ăn không tiêu gây khó chịu vùng thượng vị | high | 0.3516 (medium) | ❌ |
| 3 | Bàng quang tăng hoạt gây tiểu gấp | Phương pháp nấu phở bò truyền thống | low | 0.2291 (low) | ✅ |
| 4 | Áp xe phổi điều trị bằng kháng sinh | Áp xe hậu môn cần phẫu thuật dẫn lưu | medium | 0.4715 (medium) | ✅ |
| 5 | Băng huyết sau sinh do đờ tử cung | Bàn chân đái tháo đường bị loét | low | 0.3812 (medium) | ❌ |

**Dự đoán đúng: 2/5 (40%)**

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 1 bất ngờ nhất: "Alzheimer gây mất trí nhớ" vs "Sa sút trí tuệ khiến người già hay quên" — tôi dự đoán high nhưng thực tế chỉ 0.546 (medium). Điều này cho thấy embedding model `text-embedding-3-small` xử lý tiếng Việt y khoa chưa thực sự sâu: dù hai câu đồng nghĩa, model không nắm được rằng "Alzheimer" ≈ "sa sút trí tuệ" và "mất trí nhớ" ≈ "hay quên" trong ngữ cảnh y tế. Embeddings biểu diễn nghĩa dựa trên phân bố từ trong dữ liệu huấn luyện — nếu dữ liệu tiếng Việt y tế ít, model sẽ không liên kết được các khái niệm chuyên ngành đồng nghĩa.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer | Chunk nguồn |
|---|-------|-------------|-------------|
| 1 | Bệnh Alzheimer được chẩn đoán chia giai đoạn theo thang điểm MMSE như thế nào? | Alzheimer nhẹ: MMSE 21-26, trung bình: 10-20, trung bình nặng: 10-14, nặng: dưới 10. | alzheimer.md — phần Chẩn đoán (Thang MMSE) |
| 2 | Các biện pháp cận lâm sàng nào giúp đánh giá tình trạng mạch máu và tưới máu bàn chân ở người bệnh đái tháo đường? | Sử dụng các biện pháp đo ABI (Ankle Brachial Index) và đo TcPO2 (transcutaneous oxygen tension). | ban-chan-dai-thao-duong.md — phần Chẩn đoán cận lâm sàng |
| 3 | Thời gian dùng kháng sinh điều trị áp xe phổi nguyên phát thường kéo dài bao lâu? | Thời gian dùng kháng sinh kéo dài từ 4 đến 6 tuần hoặc cho đến khi X-quang ngực sạch hoặc chỉ còn vết sẹo nhỏ. | ap-xe-phoi.md — phần Điều trị nội khoa |
| 4 | Nguyên nhân hàng đầu gây ra băng huyết sau sinh là gì? | Nguyên nhân hàng đầu (hay gặp nhất) gây băng huyết sau sinh là đờ tử cung. | bang-huyet-sau-sinh.md — phần Nguyên nhân |
| 5 | Ăn không tiêu thường xuyên có thể là dấu hiệu của những bệnh tiêu hóa nguy hiểm nào? | Viêm loét dạ dày, trào ngược dạ dày thực quản, viêm dạ dày, liệt dạ dày, thoát vị hoành, sỏi mật, viêm túi mật, viêm tụy, IBS, bệnh celiac, tắc ruột non, ung thư dạ dày. | an-khong-tieu.md — phần Nguyên nhân bệnh lý |

### Kết Quả Của Tôi

Cấu hình: RecursiveChunker(500) + OpenAI text-embedding-3-small + In-memory store

| # | Query | Top-1 Chunk (tóm tắt) | Top-1 Score | Top-3 Relevant? | Agent Answer (tóm tắt) |
|---|-------|-----------------------|-------------|-----------------|------------------------|
| 1 | Alzheimer MMSE | "Thang điểm MMSE có điểm cắt chẩn đoán sa sút trí tuệ là 26..." | 0.7558 | ✅ | Dựa trên 3 chunks: trích dẫn đúng thang MMSE |
| 2 | Mạch máu bàn chân ĐTĐ (filter: Nội tiết) | "Các biện pháp cận lâm sàng giúp đánh giá tình trạng mạch máu, tưới máu bàn chân như: đo ABI..." | 0.7512 | ✅ | Trả đúng ABI và TcPO2 |
| 3 | Kháng sinh áp xe phổi | "GS.TS.BS Ngô Quý Châu cho biết, có rất nhiều phương pháp điều trị áp-xe phổi khác nhau..." | 0.6764 | ✅ | Trích dẫn đúng 4-6 tuần |
| 4 | Nguyên nhân băng huyết (filter: Sản phụ khoa) | "Băng huyết sau sinh là gì? Băng huyết sau sinh là tình trạng..." | 0.6171 | ✅ | Xác định đờ tử cung |
| 5 | Bệnh tiêu hóa nguy hiểm (filter: Tiêu hóa) | "Thường xuyên ăn không tiêu là bệnh gì? Nguyên nhân ăn không tiêu..." | 0.5661 | ✅ | Liệt kê các bệnh nguy hiểm |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

**Top-3 Analysis:** Trong hầu hết các queries, Top-2 và Top-3 đến từ cùng tài liệu nguồn với Top-1 (các section lân cận trong cùng bài viết bệnh). Q1 nổi bật với cả 3 chunks đều directly relevant (chẩn đoán MMSE + phân loại giai đoạn + triệu chứng). Q5 có Top-2/3 chứa phần mục lục và triệu chứng chung — chỉ partially relevant.

### Metadata Filter Impact

3/5 queries (Q2, Q4, Q5) sử dụng `search_with_filter()` với metadata `category`:

- **Cá nhân:** Filter không thay đổi precision (vẫn 100%) nhưng giúp tăng relevance score bằng cách loại chunks ngoài chuyên khoa — đặc biệt hiệu quả với Q5 (query chung chung dễ match cross-domain).
- **Nhóm:** Filter là yếu tố quyết định khi embedding model yếu — kéo precision từ 9.5% → 100% (Huỳnh Văn Nghĩa), 66.8% → 100% (Lưu Lương Vi Nhân). Chi tiết tại bảng so sánh nhóm ở Section 3.

> So sánh `search()` vs `search_with_filter()`: với embedding model mạnh (OpenAI), filter chủ yếu cải thiện Top-2/3 relevance. Với model yếu, filter là "lưới an toàn" bắt buộc.

### So Sánh Kết Quả Trong Nhóm

> Bảng so sánh chi tiết 4 thành viên (strategy, embedding model, vector DB, precision) tại **Section 3 — So Sánh Với Thành Viên Khác**. Kết luận: embedding model ảnh hưởng precision nhiều hơn chunking strategy, nhưng metadata filter san phẳng mọi khác biệt — đưa tất cả thành viên lên 100%.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Từ kết quả của Nghĩa (GPT-4o-mini, precision 9.5% → 100% khi có filter), tôi nhận ra rằng **Metadata Filter không chỉ là "nice-to-have" mà là thành phần sống còn** khi embedding model không đủ mạnh cho tiếng Việt chuyên ngành. Kết quả của Vương (Qwen 0.8B local, 95.2%) cũng cho thấy model nhỏ chạy local hoàn toàn khả thi cho RAG tiếng Việt nếu kết hợp đúng strategy.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Qua quá trình thảo luận, tôi nhận ra tầm quan trọng của việc **clean data trước khi embedding** — dữ liệu y tế gốc chứa nhiều HTML tags, breadcrumbs, và quảng cáo footer. Nếu không loại bỏ, các thông tin rác này sẽ "nhiễm" vào embedding và làm giảm chất lượng retrieval.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> (1) Thử **SentenceChunker** làm strategy chính thay vì Recursive, vì dữ liệu thực nghiệm cho thấy SentenceChunker thắng 3/5 queries về similarity score. (2) Thiết kế benchmark queries khó hơn — các query hiện tại quá rõ ràng (mỗi query chỉ liên quan 1 bệnh), nên cần thêm các query "cross-domain" để thực sự phân biệt được chunker tốt/xấu. (3) Tích hợp reranking layer sau retrieval để cải thiện thứ tự kết quả.

### Failure Analysis (Exercise 3.5)

**Query thất bại tiềm ẩn:** Q5 — "Ăn không tiêu thường xuyên có thể là dấu hiệu của những bệnh tiêu hóa nguy hiểm nào?"

Mặc dù Q5 trả về đúng document, nhưng **top-1 score chỉ 0.5661** — thấp nhất trong 5 queries. Chunk top-1 trả về là phần mục lục/table of contents của bài viết chứ không phải đoạn liệt kê các bệnh cụ thể. Nguyên nhân:
- Chunk chứa danh sách bệnh thực sự bị phân tán qua nhiều paragraph (mỗi bệnh là 1 heading riêng), nên không nằm gọn trong 1 chunk 500 chars.
- **Đề xuất cải thiện:** Dùng chunk_size lớn hơn (800-1000) cho loại nội dung "liệt kê nhiều item", hoặc thiết kế custom chunker tách theo heading level.

---

## Tự Đánh Giá

| Tiêu chí (theo SCORING.md) | Loại | Điểm tự đánh giá |
|----------------------------|------|-------------------|
| Core Implementation (tests) | Cá nhân | 30 / 30 |
| My Approach | Cá nhân | 9 / 10 |
| Competition Results | Cá nhân | 9 / 10 |
| Warm-up | Cá nhân | 5 / 5 |
| Similarity Predictions | Cá nhân | 4 / 5 |
| Strategy Design | Nhóm | 13 / 15 |
| Document Set Quality | Nhóm | 9 / 10 |
| Retrieval Quality | Nhóm | 9 / 10 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **92 / 100** |
