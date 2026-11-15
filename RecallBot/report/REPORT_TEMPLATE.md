# PCA MZG517 Mini Project 1 — RecallBot V12

> Draft only. Replace every bracketed item with evidence from this project's own outputs. Do not
> submit claims or values that are not reproducible from the cited run IDs, logs, question IDs, or
> turn numbers.

## 1. Variant, models, index and hardware

V12; personal budgeting coach; twist D (code-mixed input); k=6; context window=4,096 tokens;
scale=100,000 users/day. Chat model: [name/version]. Embedding model, dimension and pooling:
[measured]. Tokenizer: [measured]. Index and hardware: [measured].

## 2. Architecture

Insert the implemented architecture diagram from README and map each node to its code file and
function. List state-tracker fields and describe the versioned prompt and savings-goal skill.
Report NLU approach and `results/nlu_accuracy.csv`.

## 3. Methods and measured results

### 3(a) Chunking and indexing

Report both chunk sizes, tokenizer-counted overlap, metadata fields and index selection. Cite the
generated index metadata and benchmark run.

### 3(b) Dense, BM25 and hybrid retrieval

Insert hit rate@1/@3/@5 from `results/hit_rate.csv`, split by code and paraphrase questions. Cite
two entries from `results/rankings.json`: one hybrid fix and one remaining failure. Include the
CS4 reference test inputs and expected 4.160 score.

### 3(c) Semantic memory

Report fact extraction precision/recall against tagged source turns, supersession examples and
the post-restart forget test. Cite the corresponding memory files and run IDs.

### 3(d) Episodic memory

State the importance rule, recency half-life, relevance method and two tested weightings. Cite the
ranked E-probe output and Session A reflection reused in Session B.

### 3(e) Context engineering

Insert one measured `results/context_assembly.jsonl` row for a Session B turn. Explain model-native
versus external components, context limit and reply reserve. State the grounding rule and exact
fallback sentence.

### 3(f) Code-mixed twist

Report the count and share of code-mixed turns, tokenizer used, paired English/Hinglish token
counts, query rewrite examples and both twist probe outcomes.

### 2. Dialogue pipeline and short-term memory

Report the NLU confusion table, 20-label accuracy, state fields, confirmed write evidence, and
full/trim/summary input tokens. Include the summary-call overhead from `calls.jsonl`; identify the
first turn lost by trim and one failure per strategy.

### 4. Evaluation and error analysis

Report the four configuration results from logs, per-category probe accuracy, median latency,
input/output tokens, 15 manually rated Session B responses/configuration and judge agreement if
an LLM judge was used. Trace three actual failures to their stage with ranked evidence and a fix.
Insert five surprise probes only after they are released.

### 5. Cost and scale

Name the exact CS3 price row and show the monthly-cost equations, cached and uncached. Report
cacheable prefix tokens and minimum. Show one-year vector-count assumptions, flat/HNSW memory
estimates, measured 100,000-vector latency and real-query HNSW recall.

## Limitations and next steps

[Discuss what the measured system cannot establish and what you would change with one more week.]

## Appendix A — AI-use declaration

Use the declaration draft, verify it is accurate, and include the specific suggestion you rejected,
the evidence that showed it was unsuitable, and the change you made.

## Appendix B — Prompts

Attach the versioned system, RAG, extraction, importance, reflection, summary and judge prompts
from `skills/`.
