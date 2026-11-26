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
| match | `jobs.py`, `match.py` | Splits the JD into requirements, finds evidence for each, scores fit, recommends apply/consider/skip |
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

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export FMNJ_KB_DIR=../career-knowledge-base
export FMNJ_DATA_DIR=~/fmnj-private-data      # anywhere outside this repo
export FMNJ_CANDIDATE_NAME="Your Name"
export OPENROUTER_API_KEY=...                  # only for --llm openrouter; type it in your terminal, never commit it
pytest -q
```

Both repos should sit side by side:

```bash
git clone https://github.com/pjtina39/find-my-next-job
git clone https://github.com/pjtina39/career-knowledge-base   # private
cd find-my-next-job
```

## Run

```bash
python -m fmnj ingest JOB_ID --jd jd.txt --company "Company" --title "Role" --url URL
python -m fmnj label JOB_ID apply --by me --source "decided before running the model"
python -m fmnj match JOB_ID
python -m fmnj decide JOB_ID apply --fit 4 --pref 5 --reason "..."
python -m fmnj generate JOB_ID                       # no API needed
python -m fmnj generate JOB_ID --llm openrouter --model VENDOR/MODEL   # needs OPENROUTER_API_KEY
python -m fmnj generate JOB_ID --llm anthropic       # needs ANTHROPIC_API_KEY
python -m fmnj review JOB_ID GEN_ID --doc cv \
  --ratings factual_accuracy=5,relevance=4,evidence_selection=4,impact=4,writing_style=4,overall=4 \
  --comment "..." --edited my_cv.md --approve
python -m fmnj show JOB_ID
```

## Results

First real run done on 26 Nov 2026 (one job). Results are kept in the private
data folder, not here. One job is one case, not an accuracy number.

## Limitations

- **Requirement extraction is rule-based.** It relies on headings and bullets. Odd JD layouts need a check (`requirements.json` lists warnings).
- **Matching is lexical.** Skill aliases plus TF-IDF. No embeddings yet, so synonyms the alias list doesn't know are missed. Some skill terms are broad (for example "analy" tags data analysis).
- **Thresholds are guesses.** Set by hand on one job. Tune them when there are more labelled jobs.
- **Education and seniority** are not in the KB, so they come back as `needs_human`.
- **Years of experience** is an estimate from role dates, not a claim.
- **The extractive generator** reuses evidence wording. It is safe but reads like a list. The LLM option writes better prose but has not been run yet (no key in the build environment).
- **The claim check** catches numbers and missing evidence. It does not catch a reworded claim that says more than the evidence.
- **No UI, no application filling.** Planned for Friday.
