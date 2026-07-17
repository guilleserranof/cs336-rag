# Evaluation

## Retrieval evaluation

### Method

Retrieval quality is measured against an LLM-generated ground-truth set. For a
seeded random sample of 150 knowledge-base chunks, the chat model writes 2
self-contained questions per chunk (300 questions total). Each question is
labelled with the `chunk_id` it was generated from — that chunk is the single
relevant document the retriever should surface.

Each method is scored by:

- **Hit rate@k** — fraction of questions whose relevant chunk appears in the
  top *k* results.
- **MRR** — Mean Reciprocal Rank of the relevant chunk (misses score 0).

Reproduce with:

```bash
uv run cs336-rag generate-ground-truth --sample 150 --per-chunk 2   # -> data/ground_truth.json
uv run cs336-rag evaluate-retrieval --limit 10                      # -> data/eval/retrieval_eval.json
```

The ground-truth set and the results JSON are committed, so the numbers below
are reproducible without regenerating the dataset.

### Results

| Method | Hit rate@5 | Hit rate@10 | MRR |
|---|---|---|---|
| text (BM25-style FTS) | 0.797 | 0.883 | 0.578 |
| **vector (pgvector cosine)** | **0.973** | **0.990** | **0.869** |
| hybrid (RRF of text + vector) | 0.960 | 0.983 | 0.796 |
| hybrid + rerank (cross-encoder) | 0.210 | 0.310 | 0.128 |

**Winner: `vector`.** It leads on every metric, so it is the default retrieval
method (`RETRIEVAL_METHOD=vector`, wired through `Settings.retrieval_method`).

### Discussion

- **Vector search dominates.** The qwen3 embeddings capture the paraphrase
  between a formal question and colloquial lecture speech that lexical search
  cannot. Hit rate@10 of 0.99 means the answer chunk is almost always in the
  top 10.
- **Text search is a real but weaker baseline.** Full-sentence questions rarely
  share every word with a transcript chunk, so the query is OR-matched over its
  terms and ranked by `ts_rank_cd` cover density. (ANDing every term — the naive
  `websearch_to_tsquery` default — scored only 0.017 because ~90% of questions
  matched nothing; see the retrieval-eval PR.)
- **Hybrid is marginally below pure vector here.** Reciprocal Rank Fusion weights
  both rankings equally, so blending the weaker text ranking into the much
  stronger vector ranking slightly dilutes it. Hybrid wins when the two signals
  are more balanced; on this dataset vector alone is best.
- **Reranking hurts on this corpus.** The hosted `rerank` cross-encoder is poorly
  calibrated for ASR-style transcript text: in a direct probe it scored an
  irrelevant "bananas are a good source of potassium" document at 0.82 against a
  byte-pair-encoding query. Asked to order 30 topically-similar lecture chunks it
  cannot discriminate and pushes the correct chunk out of the top 10. The rerank
  path is implemented and evaluated, but deliberately not used by default.

### Note on query rewriting

Query rewriting (`retrieval.rewrite_query`) targets messy real-world queries
(typos, abbreviations, conversational phrasing). The ground-truth questions are
already clean, well-formed sentences, so rewriting them changes little; its value
is on the live interactive path rather than this offline benchmark.
