| Method | Recall@5 | MRR@10 | Hit@1 | Recall@5 (date queries) | Mean latency (ms) | p95 (ms) |
|---|---|---|---|---|---|---|
| Filename (baseline) | 0.008 | 0.002 | 0.000 | 0.000 | 2 | 2 |
| Keyword BM25 | 0.915 | 0.876 | 0.839 | 1.000 | 1 | 2 |
| Semantic | 1.000 | 0.955 | 0.915 | 1.000 | 12 | 19 |
| Hybrid | 1.000 | 0.966 | 0.932 | 1.000 | 13 | 20 |
| Agent, offline planner | 1.000 | 0.987 | 0.975 | 1.000 | 10 | 13 |

OCR: extraction success (token recall >= 0.9) 100.0% over 232 files, mean token recall 0.998, approx CER 0.056, text-free photos left empty 38/38
By source: {'pdf': 1.0, 'pdf-scanned': 1.0, 'phone': 1.0, 'receipt': 1.0, 'slide': 1.0, 'web': 0.992}
Dedup: 30/30 exact duplicates detected
Indexing: {'seconds': 0.1, 'files_per_sec': None, 'note': 'only newly indexed files are timed; use --reindex for a full timing'}
