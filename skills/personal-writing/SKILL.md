---
name: personal-writing
description: Tina's writing style for tailored CVs and cover letters. Load before writing or editing any application document. It covers the fixed CV format, bullet rules, evidence selection and cover letter rules.
version: personal-writing-v0.3
---

# Personal writing style (v0.3)

Derived from Tina's own CV (KB `sources/cv-2026-11.md`) and her rules of
27 November 2026. Approved examples live in `private-data/writing-examples/`
(private, never committed). When a reviewed document is approved, add it
there and update this file if the edits show a new rule.

## 1. Errorless (the most important rule)

The whole document must have no grammar, spelling or punctuation errors.
Run this check before handing anything over:

- Every bullet starts with a capital letter and ends with a full stop.
- Sentences in the summary and role lines end with full stops. Headings and
  the skills, certifications and additional-information lines have no full stop.
- British spelling throughout: optimised, analysed, specialisation, modelling.
- Ranges use an en dash with no spaces for numbers (94–97%) and with spaces
  for dates (Jun 2025 – Now).
- No double spaces, repeated words, or stray spaces before punctuation.
- Tense: past tense for past roles; past tense for completed work in the
  current role.

## 2. CV: fixed format (November CV)

Keep these sections in this order. Don't add or rename sections, except
Technical Projects.

1. Name, then location line, then contact line, then headline
   `APPLIED AI | DATA SCIENCE | DATA ANALYTICS`
2. `## Summary`: 2–3 sentences
3. `## Technical skills`: `Programming & Data:` and `Machine Learning & AI:` lines
4. `## Work and leadership experience`: `### Company, City, Country`, then
   `**Role**, Mon YYYY – Mon YYYY`, an optional one-sentence role summary, then bullets
5. `## Technical Projects`, with `### Personal projects` first and `### Hackathons`
   underneath. Each entry follows this shape:
   - Heading line: `**Name** | Event or context, Mon YYYY | Tina's actual role`
     (e.g. Team lead, Co-developer with one engineer).
   - Bullet 1, what was built and why: the problem, the working prototype or MVP,
     and the main tech stack.
   - Bullet 2, the outcome: award, votes, metrics or later use. If there's none yet, add a `todo`.
   - `Links:` line, only for public links that look finished (a live site, or a repo
     with a README). Leave out links to internal tools and to empty repos.
   - Order: personal projects with public links first; hackathons by JD relevance.
6. `## Education`: `- Institution, Degree, YYYY – YYYY`
7. `## Certifications`: one line, separated by ` | `
8. `## Additional information`: Languages, Interests

## 3. Bullets

- **Structure:** strong verb → method, tools, size or problem tackled →
  impact or outcome (a number where there is one).
  Verbs: Led, Built, Designed, Evaluated, Analysed, Optimised, Co-developed,
  Delivered, Secured. Avoid weak openers such as Helped, Worked on,
  Responsible for, Conducted.
- **Length:** 1–2 lines, no more than about 230 characters.
- **Keywords:** reuse the JD's own words where they're true: model evaluation,
  evaluation frameworks, agentic, statistical analysis, experimentation,
  stakeholders, measurable outcomes, reduce costs, cloud-based platforms.
- **Count:** most recent and most relevant role up to 6 bullets; other roles
  3–5. If the KB has fewer true bullets for an old role, use what exists.
  Never pad.
- **Never stronger than the evidence.** "Co-developed" stays co-developed.

## 4. Team work

- A team project or piece of research Tina took part in, where someone else led
  (she's named as co-researcher or contributor), is written as
  **Co-developed** or **Co-researched**. It's shared work and fair to list.
- A colleague's solo work that Tina wasn't part of is **not** listed on the CV.
  It can be discussed in an interview as team context. Source:
  `knowledge/team-research-shared-knowledge.md` says who did what.
- Don't name clients or colleagues in a CV.

## 5. Selecting and ordering evidence for a JD

1. Map each JD requirement to the evidence that actually **meets** it.
   Keep the map in the generation's private selection note.
2. Score = sum of the weights of the requirements met (required 1.0,
   responsibility 0.7, preferred 0.5).
3. Roles in reverse chronological order. Within a role, order bullets by score;
   break ties by whether the bullet has a metric, then by recency.
4. Every required JD item should be covered by at least one bullet or skill.
   If not, add a `todo`.

## 6. Grounding and sensitivity

- Every claim about Tina comes from KB evidence and lists its id.
  Facts Tina states in conversation that aren't yet in the KB are allowed, but
  they're logged in the selection note and should be added to the KB.
- Education, certifications, languages and interests come from the source CV.
- Sensitive evidence that already appears on Tina's own CV (e.g. her own
  financial results) may be used with `--allow-sensitive`. Client names,
  colleague names and internal tool names stay out.

## 7. Cover letter

- About 250–350 words in 4–5 short paragraphs. Open with "Dear Hiring Team," unless a
  name is known.
- Paragraph 1: the role and team by name, and who Tina is now.
- Paragraphs 2–3: each answers one theme the JD stresses most, with 1–2 pieces
  of evidence and their results. Use the JD's words for the theme.
- Paragraph 4: how Tina works with people, from evidence.
- Motivation must come from Tina. Leave a `todo` until she writes it.
- Close with "Thank you for your time. I would welcome the chance to talk." /
  "Kind regards," and her name.
- Don't repeat CV bullets word for word, and don't start consecutive
  sentences or paragraphs the same way.

## Voice (all documents)

Plain, direct and professional. Name concrete things: the model, the data,
the tool, the result. Avoid: passionate, cutting-edge, leverage (as a verb),
synergy, spearheaded, results-driven, dynamic.
