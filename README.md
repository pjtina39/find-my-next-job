# find-my-next-job

Takes a job description, checks it against my career evidence, and drafts a
tailored CV and cover letter. A human makes every decision and approves every
document. It never submits anything on its own.

Status: **V0.1**, one job end to end, CLI only.

## Privacy

This repo is public, so it holds **code and synthetic test data only**.

Real data lives outside the repo:

- Career evidence: the private `career-knowledge-base` repo, read-only (`FMNJ_KB_DIR`).
- JDs, labels, predictions, CVs and feedback: a private data folder (`FMNJ_DATA_DIR`).
  The default `private-data/` is gitignored.

## How it works

```
ingest JD -> label (human, before any model) -> match -> decide (human)
          -> generate CV + cover letter -> review, edit, approve (human)
```

| Step | Module | What it does |
|---|---|---|
| ingest | `jobs.py` | Saves the JD text as-is with a stable job id |
| label | `evaluate.py` | Saves my apply/skip call **before** matching |
| match | `jobs.py`, `match.py`, `embed.py` | Splits the JD into requirements, maps JD phrases to KB skills and finds evidence for each, all with SBERT embeddings; scores fit, recommends apply/consider/skip |
| compare | `evaluate.py` | Compares the recommendation with my label |
| decide | `review.py` | My final apply/skip, plus fit and preference ratings |
| generate | `generate.py` | Writes the CV and cover letter from evidence, then checks every claim |
| review | `review.py` | Ratings, comments, edits and approval for each document |

Rules the code enforces:

- `match` will not run without a saved label. Model code never reads label values.
- `generate` will not run without a human "apply" decision.
- Every generated line lists the evidence ids it uses. The claim check flags numbers that are not in the cited evidence, claims with no evidence, and sensitive evidence.
- Evidence marked sensitive in the KB is left out unless you pass `--allow-sensitive`.
- A document with a known factual error cannot be approved until you supply an edited version.
- Records are append-only and carry job id, KB version, code version, and model/prompt version.

Rubrics and record fields: [docs/rubrics.md](docs/rubrics.md).

## How matching works

Everything is compared with SBERT embeddings (`all-MiniLM-L6-v2`). The model
turns a sentence into one 384-number vector; similar meanings give a high
cosine. There is no keyword or alias matching.

1. Each JD bullet is one requirement. We split it into short phrases
   ("Python", "SQL", "data science tooling").
2. Each phrase is matched to its closest KB skill (skill name + aliases).
   Below 0.5 cosine it stays unmapped and is listed as a possible gap.
3. For each requirement and each evidence record:
   - skill score = share of the mapped skills that the evidence has
   - sentence score = cosine(requirement, evidence claim)
   - score = 0.5 x skill score + 0.5 x sentence score
4. The requirement keeps its best evidence: >= 0.6 matched, >= 0.3 partial, else unsupported.
5. Overall fit = weighted mean over requirements (required 1.0, responsibility 0.7, preferred 0.5).
   >= 0.6 apply, >= 0.4 consider, else skip.

### Second metric: LLM judge (optional)

`match --judge-model VENDOR/MODEL` sends the JD and a CV to the model through
OpenRouter and asks one question: "should this candidate be shortlisted? Yes or No".
It reads the probability of the first answer token:

    p_yes = P(Yes) / (P(Yes) + P(No))

This is stored beside the fit score (`llm_judge` in the prediction) and compared
with the label (`judge_p_yes`, `judge_agree` in the evaluation). It does not change
the fit score or the recommendation. The CV is built from non-sensitive KB evidence
unless you pass `--cv file.txt`. The label is never in the prompt.

The model must return logprobs. Many OpenRouter models do not; then `p_yes` is
empty and a note says why.

### Third metric: Jev (optional)

`match --jev` asks TypeSafe's Jev model, through OpenRouter, the same kind of
question: should this candidate be shortlisted? Jev gets the JD and the CV as its
state and returns the probability of yes directly, so no logprobs are needed.

The question and its yes/no criteria are in `src/prompts/jev_shortlist.json`.
Edit that file to change the question. The result is stored as `jev` in the
prediction (probability, cost, latency, tokens) and compared with the label
(`jev_probability`, `jev_agree`). Like the LLM judge, it does not change the fit
score or the recommendation, and the label is never sent.

Code: `src/jev_judge.py` (the question and the call) and `src/jev_openrouter.py`
(points the LangChain TypeSafe classifier at OpenRouter). `langchain-typesafe` is
pinned to `0.0.1a3` because `jev_openrouter.py` overrides internals that changed in 0.1.0.

We do not embed the whole JD as one vector. The model cuts input at about 256
word pieces, and one vector would blur twenty requirements together.

## Setup

```bash
uv sync                                        # creates .venv; add --extra anthropic for --llm anthropic
source .venv/bin/activate                      # or prefix commands with `uv run`
export FMNJ_KB_DIR=../career-knowledge-base
export FMNJ_DATA_DIR=~/fmnj-private-data      # anywhere outside this repo
export FMNJ_CANDIDATE_NAME="Your Name"
export OPENROUTER_API_KEY=...                  # only for --llm openrouter; type it in your terminal, never commit it
uv run pytest -q
```

Both repos should sit side by side:

```bash
git clone https://github.com/pjtina39/find-my-next-job
git clone https://github.com/pjtina39/career-knowledge-base   # private
cd find-my-next-job
```

## Run

```bash
uv run python -m src ingest JOB_ID --jd jd.txt --company "Company" --title "Role" --url URL
uv run python -m src label JOB_ID apply --by me --source "decided before running the model"
uv run python -m src match JOB_ID                         # add --judge-model VENDOR/MODEL and/or --jev
uv run python -m src decide JOB_ID apply --fit 4 --pref 5 --reason "..."
uv run python -m src generate JOB_ID                       # no API needed
uv run python -m src generate JOB_ID --llm openrouter --model VENDOR/MODEL   # needs OPENROUTER_API_KEY
uv run python -m src generate JOB_ID --llm anthropic       # needs ANTHROPIC_API_KEY
uv run python -m src review JOB_ID GEN_ID --doc cv \
  --ratings factual_accuracy=5,relevance=4,evidence_selection=4,impact=4,writing_style=4,overall=4 \
  --comment "..." --edited my_cv.md --approve
uv run python -m src show JOB_ID
```

## Results

First real run done on 26 Nov 2026 (one job). Results are kept in the private
data folder, not here. One job is one case, not an accuracy number.

## Limitations

- **Requirement extraction is rule-based.** It relies on headings and bullets. Odd JD layouts need a check (`requirements.json` lists warnings).
- **Phrase splitting is crude.** It splits on commas, "and", "or". Some phrases are filler ("design", "test"); they are left out of the skill score, which also means a real unmapped skill can hide there. Check `unmapped_phrases` in the prediction.
- **The model downloads from Hugging Face on first run.** The build environment blocks that, so SBERT matching has only been tested with a fake encoder. Run it locally.
- **Thresholds and weights are guesses** (0.5 skill match, 0.5/0.5 weights, 0.6/0.3, 0.6/0.4). None are tuned for SBERT yet. Re-check them after the first SBERT run. Tune them when there are more labelled jobs.
- **Education and seniority** are not in the KB, so they come back as `needs_human`.
- **Years of experience** is an estimate from role dates, not a claim.
- **The extractive generator** reuses evidence wording. It is safe but reads like a list. The LLM option writes better prose but has not been run yet (no key in the build environment).
- **The claim check** catches numbers and missing evidence. It does not catch a reworded claim that says more than the evidence.
- **No UI, no application filling.** Planned for Friday.
