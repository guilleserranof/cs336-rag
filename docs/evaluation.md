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
| hybrid (RRF of text + vector) | 0.963 | 0.983 | 0.796 |
| hybrid + rerank (cross-encoder) | 0.207 | 0.310 | 0.129 |

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

## Answer evaluation (prompt variants)

### Method

Three answering strategies are compared over a seeded sample of 30 evaluation
questions. Context is retrieved **once per question** and shared across variants,
so the only thing that differs is the prompt:

- **baseline** — minimal instruction: answer the question from the context.
- **grounded** — answer only from the context, cite passages inline as `[n]`,
  say so plainly when the context does not cover the question.
- **tutor** — same grounding and citation discipline, but explain for a student
  new to the topic: define terms and give intuition before specifics.

A separate judge model (`deepseek-v4-flash`, deliberately different from the
`qwen3.6` generator to reduce self-preference bias) rates each answer 1-5 on:

- **relevance** — does it directly and completely address the question?
- **groundedness** — is every claim supported by the context?
- **citation** — are claims attributed to numbered passages with `[n]` markers?

Reproduce with:

```bash
uv run cs336-rag evaluate-prompts --sample 30   # -> data/eval/answer_eval.json
```

### Results

| Variant | Relevance | Groundedness | Citation | Overall | n |
|---|---|---|---|---|---|
| baseline | 4.83 | 4.87 | 2.90 | 4.20 | 30 |
| grounded | 4.83 | 4.87 | 4.80 | 4.83 | 30 |
| **tutor** | **4.93** | **4.90** | **4.90** | **4.91** | 29 |

**Winner: `tutor`**, wired in as the default (`RAG_PROMPT_VARIANT=tutor`).

### Discussion

- **The citation axis is what separates the variants.** An earlier run judged
  only relevance and groundedness, and every variant scored 4.97-5.00 — the
  differences were within noise and the "winner" was arbitrary. Retrieval is
  strong (hit rate@10 = 0.99), so the context nearly always supports a good
  answer and both axes saturate. Citation quality is the dimension the prompts
  genuinely differ on, and the one that makes answers auditable, so it was added
  as a third axis.
- **`baseline` fails on citation (2.90).** With no instruction to cite, the model
  usually writes a fluent, correct, *unverifiable* answer. That is precisely the
  failure mode a RAG system must avoid: a reader cannot check the claim against
  the lecture.
- **`tutor` edges out `grounded`** on all three axes. Explaining terms before
  specifics appears to make answers read as more complete (relevance 4.93 vs
  4.83) without loosening grounding. The margin over `grounded` is small (0.08);
  both are defensible choices and either is far ahead of `baseline`.
- **`n=29` for `tutor`**: one answer produced no parseable judgement and was
  excluded rather than silently scored, so variants are compared only on
  questions where the judge actually returned a verdict.
