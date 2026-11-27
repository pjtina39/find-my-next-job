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
| match | `jobs.py`, `match.py`, `embed.py` | Splits the JD into requirements and finds evidence for each by SBERT embedding similarity; scores fit, recommends apply/consider/skip |
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

1. Each line under a known JD section is one requirement.
2. For each requirement and each evidence record: score = cosine(requirement, evidence claim + its skill names).
3. The requirement keeps its best evidence: >= 0.6 matched, >= 0.3 partial, else unsupported.
4. Overall fit = weighted mean over requirements (required 1.0, responsibility 0.7, preferred 0.5).
   >= 0.6 apply, >= 0.4 consider, else skip.
   A required item with no evidence (unsupported) caps the result at consider, never skip:
   no evidence in the KB is not proof the skill is missing. See `missing_required` in the prediction.

### Extra: LLM judge (optional)

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

### Second score: Jev

`match` also asks TypeSafe's Jev model, through OpenRouter: does the candidate's
experience and skill set align with the requirements of the job description? Jev gets three things as its state: the JD, the candidate's
experience (roles + non-sensitive evidence from the KB) and skills (KB skill names).
It returns the probability of yes. Needs `OPENROUTER_API_KEY`; `--no-jev` skips it.

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
uv run python -m src ingest --jd jd.txt --company "Company" --title "Role" --url URL   # saves data/jd/<company>_<role>=<uuid>.txt + an entry in data/job.json
uv run python -m src label JOB_ID apply --by me --source "decided before running the model"
uv run python -m src match JOB_ID                         # embedding score + Jev; --no-jev to skip Jev
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
- **The model downloads from Hugging Face on first run.** The build environment blocks that, so SBERT matching has only been tested with a fake encoder. Run it locally.
- **Thresholds and weights are guesses** (0.6/0.3 per requirement, 0.6/0.4 overall). None are tuned for SBERT yet. Re-check them after the first SBERT run. Tune them when there are more labelled jobs.
- **Education and seniority** are not in the KB, so they come back as `needs_human`.
- **Years of experience** is an estimate from role dates, not a claim.
- **The extractive generator** reuses evidence wording. It is safe but reads like a list. The LLM option writes better prose but has not been run yet (no key in the build environment).
- **The claim check** catches numbers and missing evidence. It does not catch a reworded claim that says more than the evidence.
- **No UI, no application filling.** Planned for Friday.
