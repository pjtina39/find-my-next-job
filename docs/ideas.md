# Ideas

A running list of improvement ideas, newest first. An idea isn't a plan:
move it into a roadmap or an investigation when work starts.

## Precompute KB embeddings (2026-11-27)

Embed the knowledge base once and store the vectors with the KB. When new
experience or skills are added, embed only the new or changed records. The
matcher always loads the latest stored embeddings instead of re-encoding
every run.

Notes:
- Key each vector by (embedding model name, evidence id, hash of the embedded
  text). Then a model change or an edited claim forces a re-embed, and an
  old cache never gets used by mistake.
- Today `src/kb.py` treats the KB as read-only ("We never write to the KB").
  So either the KB repo owns an `embed` step (vectors versioned with the KB
  commit), or this repo keeps a cache in `private-data/` keyed by
  `kb_version` + model.
- Record the vector file's version in each prediction, next to `kb_version`
  and `embed_model`.
- Fits the experiments in
  [investigations/2026-11-27-cosine-baseline.md](investigations/2026-11-27-cosine-baseline.md)
  (Q2, Q3): each representation or model variant becomes its own cached
  vector set.
