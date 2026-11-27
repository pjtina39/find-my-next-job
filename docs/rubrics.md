# Labels, rubrics and records

## Three separate human signals

| Signal | When | Values | Stored in |
|---|---|---|---|
| Apply label | Before any model runs | apply / skip | `labels/<job_id>.json` |
| Final decision | After seeing the match | apply / skip, plus reason | `decisions/<job_id>/` |
| Fit and preference ratings | With the final decision | 1-5 each | same decision record |

- The apply label is a personal preference. It is not proof of fit.
- Fit = "could I do this job?" Preference = "do I want this job?" Keep them apart.
- Labels are never passed to extraction, matching or generation. `match` only checks a label file exists, so the order is enforced.

## Qualification-fit rubric (model)

Each JD requirement gets a status:

| Status | Meaning |
|---|---|
| matched | score >= 0.6: evidence covers the skills and the wording is close |
| partial | 0.3 to 0.6: some overlap |
| unsupported | < 0.3: nothing in the KB backs it |
| needs_human | the KB cannot answer it (education, seniority) |

Score per requirement = 0.5 x skill score + 0.5 x SBERT sentence cosine. Skill score = share of the requirement's phrases (mapped to KB skills by embedding, cosine >= 0.5) that the evidence has. See README, How matching works.
Overall fit = weighted mean (required 1.0, responsibility 0.7, preferred 0.5).
Recommendation: apply >= 0.6, consider 0.4-0.6, skip < 0.4.
If any required item is unsupported, apply is capped at consider (listed in `missing_required`).

Dimensions reported: skills, experience (years), achievements (share of matched
requirements backed by evidence with a metric), qualifications, seniority.

Thresholds are first guesses. Tune them once there are more labelled jobs.

**How "consider" is compared with the label:** we report strict agreement
(exact match) and lenient agreement (consider = apply). One job is one case,
not an accuracy figure.

## CV rubric (1-5)

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| factual_accuracy | wrong or invented facts | small errors | every fact correct and traceable |
| relevance | mostly off-target | some relevant points | leads with what the JD asks for |
| evidence_selection | weak or random evidence | ok, misses strong items | best evidence for this role |
| impact | duties only | some results | clear results and numbers |
| writing_style | hard to read | fine | clear, short, my voice |
| overall | would not send | send after edits | send as is |

## Cover letter rubric (1-5)

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| factual_accuracy | wrong or invented facts | small errors | every fact correct |
| relevance | generic | some links to the role | every paragraph ties to the role |
| personalisation | could be any company | mentions the company | specific to team and role |
| writing_style | stiff or vague | fine | clear, short, my voice |
| overall | would not send | send after edits | send as is |

**Rule:** a document with a known factual error cannot be approved until an
edited version fixes it, whatever the overall score.

## Records and versioning

Every record has a job id and a UTC timestamp. Records are append-only.

| Record | Key fields |
|---|---|
| job | job_id, company, title, source_url, requisition_id, ingested_at |
| requirements | id, text, type, phrases, years, is_education; in the prediction also skill_matches (phrase, skill_id, sim) |
| prediction | prediction_id, fit_score, recommendation, per-requirement evidence ids and source refs, kb_version, code_version, method_version |
| evaluation | prediction_id, label_id, agree_strict, agree_lenient |
| generation | generation_id, prediction_id, generator, model, prompt_version, kb_version, evidence ids per line, claim flags, excluded sensitive evidence |
| feedback | feedback_id, generation_id, document, ratings, comment, factual_errors, edited file, preference pair (chosen = edited, rejected = original), approved, evidence ids, kb_version, model and prompt versions |
