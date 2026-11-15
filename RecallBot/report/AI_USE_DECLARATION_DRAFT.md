# Appendix A — AI-use declaration (draft)

I used GitHub Copilot (AI assistant) to help interpret the assignment brief, scaffold the Python
project, generate synthetic budgeting-guide documents and test data, and debug the implementation.
I reviewed the assignment's restrictions and kept BM25, reciprocal-rank fusion, chunking, state
tracking, goal-write confirmation, and memory update/delete logic in project code rather than using
a RAG or memory framework.

One suggestion considered during drafting was to use a feature-hashing vector as the production
semantic embedding. That would not satisfy the assignment's stated embedding-model requirement and
would be weak evidence for paraphrase retrieval. I kept test doubles limited to unit tests and
implemented live indexing through the configured Ollama embedding endpoint (or the optional
Sentence Transformers backend). Before submission, replace this paragraph if it does not match
what you actually used, and cite the run or test that verified the final choice.

This is a draft declaration, not a claim that the required live model runs, rubric ratings, pricing
analysis, surprise probes, report, or video have already been completed.
