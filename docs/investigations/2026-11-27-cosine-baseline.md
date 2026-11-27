# Investigation: can cosine similarity tell "meets the requirement" from "merely related"?

**Opened:** 2026-11-27 · **Status:** open · **Slice:** 1 (JP Morgan Applied AI/ML, Senior Associate)

Per-requirement evidence ids and scores are in the private copy
(`private-data/investigations/2026-11-27-jpm-baseline.md`, gitignored). This
public note contains no KB content and no JD text.

## Question

Can cosine similarity reliably tell apart evidence that **meets** a job
requirement from evidence that is only **semantically related** to it?

## 1. Baseline (frozen; every later change is compared against this)

| Field | Value |
|---|---|
| Prediction | `pred-20261127T151202841299` |
| Code | `20f6399` |
| KB | `cfc8f65` (4 roles, 41 evidence, 26 skills) |
| Method | `match-v0.5-embedding-only` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Thresholds | requirement: matched ≥ 0.6, partial ≥ 0.3 · fit: apply ≥ 0.6, consider ≥ 0.4 (marked "untuned for SBERT" in `src/match.py`) |
| Requirements | 24 extracted, 0 warnings; 20 scored, 2 context, 2 education (needs_human) |
| Result | **fit 0.397 → skip** · partial 18, unsupported 2, needs_human 2 · `missing_required` = [r18] |
| Label | apply → strict agree = False, lenient agree = False |
| Jev (`typesafe/jev-1.14-20261014`) | p_shortlist = 0.83 → agrees with the label |

To reproduce: `git checkout 20f6399`, KB at `cfc8f65`, then `python -m src match <job_id>`.

## 2. Observations

- **Scores are squeezed into a narrow band.** The 20 scored requirements range
  from 0.201 to 0.479 (mean 0.400, median 0.427). None can reach "matched"
  (0.6). The fit is a weighted mean of the best cosines, so it inherits this
  scale and can't reach "apply" either. The result missed "consider" by 0.003.
- **Top-k barely separates candidates.** The top 3 evidence records per
  requirement are often within 0.01–0.07 of each other (e.g.
  0.430 / 0.429 / 0.423). The ranking is close to a tie.
- **Hub evidence.** 10 of the 20 requirements pick one of the same 3 generic
  evidence records as their best match (4 / 3 / 3). That points to topical
  similarity, not real support.
- **Some top-1 matches look right, some are related but don't meet the
  requirement.** Requirements about evaluation workflows and LLMs find
  plausible evidence. "Strong analytical, problem-solving and communication
  skills" scores 0.20 (unsupported) even though the KB has a
  stakeholder-communication skill. A governance requirement matches an
  unrelated operations record.
- **The KB has the coverage.** Python/SQL (6 records), LLM, evaluation and
  agents are all present. Low scores don't mean missing evidence.
- **Input scope.** The matcher reads only `evidence.json` (claim + skill names
  appended). It doesn't use a CV, experience titles or `projects.json`. Jev sees
  the whole JD and the whole profile together.

## 3. Sub-questions, hypotheses and planned experiments

| # | Question | Hypothesis | Experiment | Metric |
|---|---|---|---|---|
| Q1 | Is retrieval working? | It partly works: topically right, poorly ranked | Hand-label a gold set for the JP Morgan job: for each requirement, which evidence meets it, which is only related, which is irrelevant | recall@1, recall@3, MRR, hub rate |
| Q2 | Is the evidence representation the problem? | Long, specific claims hide short skill signals | Text variants: claim only / claim + skills (baseline) / skills only / claim + role title / skills as separate documents | Q1 metrics + score spread |
| Q3 | Can the embedding model tell "meets" from "related"? | No: a bi-encoder's cosine measures topic, not entailment | Contrast triples (requirement, evidence that meets it, related evidence). Compare MiniLM vs mpnet vs bge/e5 (with query prefix) vs a cross-encoder reranker vs LLM entailment | AUC of meets-vs-related, mean score gap |
| Q4 | Is the gap with Jev driven by input context? | Yes: comparing one line to one claim loses the JD and role context that Jev sees | Add context: requirement + section + job title; evidence + role + company. Also try scoring against the whole JD | Q1/Q3 metrics; fit gap vs Jev |

**Thresholds come after Q1–Q4.** With n = 1 labelled job, any cutoff is luck.
Collect about 5 apply and about 5 skip labels (including clearly off-target
negative controls and near misses) before sweeping APPLY/CONSIDER.

## 4. Rules for later comparisons

- Measure every change against this baseline on the same job, KB version and
  gold set, plus the extra labelled jobs once they exist.
- Change one variable at a time: representation → model → context →
  aggregation → thresholds.
- Compare **separation** (meets vs related, apply vs skip) rather than raw
  cosine, because each model has its own scale.

## Log

- 2026-11-27: Baseline recorded. No code changes.
