# Pre-registration: rerun the writer for coverage, now that we know what was scored

Written 2026-11-08, before any arm below has produced an answer. The only numbers seen when this
was fixed are the official scores (`FINDINGS_official_generation.md`) and a one-topic smoke test of
the judge on the shipped answer for dev topic 14.

## Why

The official generation measures are coverage of expected items (rubric / nugget `all_strict`,
`vital_strict`) and human preference. Citation support, which our pipeline was built and gated on,
was not scored. All four runs are in the bottom half. Our answers use about a third of the
1,024-word budget: the shipped writer caps answers at 450 words before the gap round, tells the model
to "silently omit" facets it cannot support well and that "a shorter fully-grounded answer beats a
longer hedged one", and shows it only the first 700 characters of each ~24 KB document. Every one of
those choices was made for support. This experiment asks whether undoing them raises coverage.

## What can and cannot be measured

- No nuggets, rubrics or assignments for the official 119 topics have been distributed, so a new
  answer to an official topic **cannot be scored officially**. Everything here is measured on
  **dev22**, the organisers' 22 RAG25 development topics, against the organisers' official dev
  nuggets (`data/rag/nuggets/rag25-dev-nuggets.jsonl`: human-written, 29–90 per topic, 74% vital).
- Assignment is by a local model, not the organisers' GPT-6.1 Sol: `eval/nugget_local_judge.py`,
  RAGDoll's own assigner prompt, parser and metric, judge `qwen3:32b` (the writer is
  `qwen3.5:122b`, so no model judges its own answers). This judge is not validated against the
  official one. Results are therefore "coverage as our local judge sees it, on official nuggets".
- Only the writer changes. Blueprint, pool and facet rankings come from the existing dev22 cache,
  so every arm writes from identical evidence.

## Arms (fixed now; `systems/rag/agentic_v2/synthesize.py` ARMS, `scripts/run_synth_arms.py`)

| arm | what changes relative to shipped |
|---|---|
| `shipped` | nothing: the existing `runs/rag/bp_agentic_v2_dev22.jsonl` answers |
| `rerun` | nothing, generated again: the **null control** for generator noise |
| `budget` | 950-word cap instead of 450, sentence target doubled, prompt unchanged |
| `coverage` | `budget` + a prompt that asks to cover every facet and question with specific facts, 600–900 words, instead of "silently omit" / "shorter beats longer" |
| `evidence` | `coverage` + each document shown as its best-matching 1,000 characters (most distinct query terms) instead of its first 700 |

All arms keep: one citation or more on every sentence (uncited sentences are still dropped), no
meta-commentary, the coverage self-check and one gap round, and a final trim to 1,000 words so
nothing can breach the official 1,024-word cap. That `shipped` is unchanged was checked offline:
on three topics the refactored code sends byte-identical prompts and returns identical answers.

## Primary test (one cell)

- **Contrast**: `evidence` vs `shipped`.
- **Measure**: `strict_all_score` (= official `all_strict`), mean over the 22 dev topics.
- **Pairing**: per topic. **Decision rule**: |Δ| > 2 SE of the paired differences is an effect.
- **Noise check, stated now**: `rerun` vs `shipped` on the same measure. If that null contrast is
  itself beyond 2 SE, the generator is too noisy at n = 22 and the primary result is reported as
  uninterpretable, whatever it shows.

Secondary and descriptive: `strict_vital_score`; `budget` and `coverage` vs `shipped` (which of the
three changes carries any effect); words per answer; the existing probe, 32B and v3 dev answers
scored by the same judge (does the judge order v2 above probe on dev, as the official NIST-nugget
scores do on the 119?). Secondary cells are not promoted to the headline if the primary is null.

## What a result licenses

- If the primary passes: regenerate the official 119 answers with the `evidence` arm from the
  existing `official119` cache (writer stage only, about 1–1.5 h), validate them, and keep them as a
  post-hoc run to be scored if the organisers release assignments or nuggets. Until then, the claim
  is a dev-set claim only, and it must say "local judge".
- If it fails: report the null, do not search further arms on the same 22 topics, and keep the
  paper's account as it is.
- Either way, the trade-off is reported: citation support of the `evidence` answers with the
  existing local support judge, because a coverage gain bought with ungrounded sentences is not the
  same result.

## Stated in advance

- No early stopping; every arm runs on all 22 topics.
- No arm is added or edited after the first arm output exists. A bug fix that changes outputs
  voids the run, and the voided numbers are reported too.
- Human preference, half of the official evaluation, cannot be measured here. Longer answers could
  help or hurt it; this experiment says nothing about that.

## Operational log (added after the run started; none of these changes an arm)

- 2026-11-08 13:48: stage 2 failed to load the writer, because the 32b judge had not yet released
  its VRAM. The script now waits for a free GPU between models.
- 2026-11-08 14:16: the writer's runner died of CUDA out-of-memory mid-generation on dev topic 213
  of the `rerun` arm, after 4 topics. Ollama had put 40 of 49 layers on the GPU. From here on the
  writer is `qwen3.5:122b-gpu36`, the same weights with `PARAMETER num_gpu 36`, and a crashed topic
  is retried. The 4 `rerun` answers already written (topics 14, 144, 161, 200) are kept.
- 2026-11-08 19:09: the `coverage` arm stalled on dev topic 273 and hit the 30-minute request
  timeout, after 9 of 22 topics. A BLAST job from another project (8 processes on 16 cores) had
  started at 18:40 and starved the writer's CPU-side layers. Timeouts are now retried like crashes;
  the run resumed from topic 273. Completed arms (`rerun`, `budget`) and topics are kept.
- 2026-11-08 20:45: paused the writer to score the three completed arms (`rerun`, `budget`,
  `coverage`) for a lab presentation on 9 November. This is an early look at the null control
  and two secondary arms. The primary arm (`evidence`) was partly generated and none of it has
  been scored or read; the arms and the decision rule are unchanged.

## Result (2026-11-08 22:11, `run_post_results_rerun.sh` stage 4)

Primary cell, `evidence` vs `shipped`, `strict_all`: **0.3560 → 0.4881, +0.1321, SE 0.0195, 6.8 SE.**
Null control `rerun` vs `shipped`: +0.0107, 1.4 SE, inside 2 SE, so the primary is interpretable.
Secondary: `budget` +0.0225 (1.3 SE); `coverage` +0.0828 (4.8 SE); `strict_vital` for `evidence`
+0.1166 (6.4 SE). Median words 268 → 636. Local judge (qwen3:32b) on official dev nuggets.
Still owed, as stated above: citation support of the `evidence` answers, then the official119
rewrite (writer stage only, from cache).

## Addendum A: the lab's multi-pass idea (written 2026-11-09 15:36, before any `multipass` answer exists; job started 15:37)

At the lab presentation of 9 November the team proposed to "give weight to sub-topics and do
several passes, each time reviewing if some of the topic is still to be covered". The shipped
writer reviews its draft once and repairs only sub-topics it finds wholly uncovered; it did so on 5
of 22 dev topics (23%), and the `evidence` arm on 12 (55%) (`team_ideas_2026_11_09.py` block 2).

Known when this was written: the Result above; that the v3 agent loop, which works sub-topic by
sub-topic, scores 0.3836 against 0.3560 for shipped (+0.0275, 1.4 SE) on the same judge. No
`multipass` answer has been generated or seen.

### Arm (`synthesize.py` ARMS["multipass"], `_synthesize_multipass`)

`evidence`, plus:
1. one call gives each sub-topic a weight, 3 central / 2 important / 1 background; the weights
   are shown in the facet list, with an instruction to give space in proportion to them;
2. up to **three** review passes: the reviewer lists sub-topics still uncovered and those too thin
   for their weight (thin is acted on only for weight >= 2); each such sub-topic gets up to four
   further documents from its own ranking, never one already shown (at most 16 per pass), and a
   repair call appends only new cited sentences. A pass that finds nothing, or has no unseen
   document left, ends the loop. Same final trim to 1,000 words.

Writer `qwen3.5:122b-gpu36`, temperature 0.15, same evidence cache, same parser and citation rules.

### Primary cell

- **Contrast**: `multipass` vs `evidence` (the best writer so far, not shipped), `strict_all`,
  paired per topic over the 22 dev topics. **Decision rule**: |Δ| > 2 SE of the paired differences.
- Noise reference: the null control above (+0.0107, 1.4 SE). No new null control is run.

Secondary and descriptive: `strict_vital`; `multipass` vs shipped; words per answer (longer answers
can cover more for that reason alone, so a pass must be read next to the length change); passes used
and sentences added per pass.

### Owed by the first pre-registration, run in the same job

Citation support of the `shipped`, `evidence` and `multipass` dev answers with the local support
judge (`eval/ragdoll_local_judge.py`, RAGDoll's support prompt and metric, `qwen3:32b`, 4,000-character
truncation): mean weighted precision per run, reported whatever it shows.

### Stated in advance

- These 22 topics have now served five writer arms; this is a sixth comparison on them. A pass is a
  development-set result under our local judge and needs the official nuggets to be confirmed.
- If the primary passes, `multipass` joins `evidence` as a candidate for the official119 rewrite
  (writer stage only). If it fails, `evidence` stays the candidate and no further arm is tried on
  these topics.
- No early stopping; no edit to the arm once the first answer exists; a bug fix that changes
  outputs voids the run, and the voided numbers are reported too.
- Driver: `scripts/run_multipass_dev22.sh` (log in `logs/rag/`).

### Result of addendum A (2026-11-09 18:01, `run_multipass_dev22.sh` stage 4)

Primary cell, `multipass` vs `evidence`, `strict_all`: **0.4881 → 0.4904, +0.0023, SE 0.0157,
0.1 SE, 9–11. Null.** `strict_vital` +0.0138 (0.9 SE). Median words 636 → 762. Against shipped,
`multipass` is +0.1344 (8.3 SE), no different from `evidence`. Per the rule stated above, `evidence`
stays the candidate and no further arm is tried on these 22 topics.

What the arm actually did (diagnostics in `answer_multipass.json`): the model gave weight 3 to 103
of the 111 sub-topics and weight 2 to the other 8, so the weights changed almost nothing. A review
found something to repair on 9 of 22 topics; repairs added sentences on 8 (one pass on 4, two on 4),
80 sentences in all. The idea is therefore tested only weakly. Weights that the model must ration (a
fixed budget to split) would be the version worth trying, on other topics.

### Citation support, owed by the first pre-registration (same job, stage 3)

Local support judge (`qwen3:32b`, RAGDoll prompt and metric, 4,000-character truncation), mean
weighted precision over the 22 dev topics, paired against shipped:

| run | weighted | Δ (SE) | hard | Δ (SE) |
|---|---|---|---|---|
| shipped | 0.901 | — | 0.802 | — |
| evidence | 0.809 | −0.092 (0.016), 20 of 22 topics lower | 0.649 | −0.153 (0.032) |
| multipass | 0.818 | −0.083 (0.020) | 0.675 | −0.127 (0.040) |

The coverage gain is bought partly with grounding: +0.132 coverage for −0.092 support, both beyond
5 SE. The two measures pull against each other, as the paper argues; the rewrite is not a free win.

### Operational log, continued

- 2026-11-09 20:07: as licensed above, started the `evidence` rewrite of the official 119 answers
  (writer stage only, existing `official119` cache): `scripts/run_evidence_official119.sh`, log
  `logs/rag/evidence_official119_*.log`. At ~4–5 min per topic this takes 8–9 h, not the 1–1.5 h
  estimated above. Output `runs/rag/bp_v2_evidence_official119.jsonl`, validated at the end; a
  post-hoc run, never filed.

## Addendum B: coverage counting only grounded sentences (analysis only; defined 2026-11-09 20:40, before any filtered answer was scored)

The official generation measures credit an item when the answer *text* supports it; the organisers'
description of the scores never mentions citations. So a sentence whose citation does not back it
earns the same coverage as one that is grounded. To measure how much of our coverage rests on such
sentences: for shipped, `evidence` and `multipass` on dev22, drop every sentence none of whose
citations got **full** support from our support judge (`_grounded`; a looser `_supported` keeps
partial support too), and score the remaining answers with the same nugget judge
(`eval/analysis/grounded_coverage_dev22.py`). Reported: all_strict as written vs grounded per run,
and `evidence` − shipped under each, paired, with SE. No decision rides on it; it measures the cost of
an unscored property. Known before scoring (from the build step only): sentences kept by `_grounded`
are 83% (shipped), 65% (evidence), 68% (multipass); `_supported` keeps 96–100%.

### Operational log, continued

- 2026-11-09 20:42: stopped the official119 `evidence` rewrite after 8 topics, none of them among
  the 42 narratives the organisers judged for generation (batch 1: 17, batch 2: 25; only these can
  ever be scored). Restarted with those 42 first (`--only`, run file `_judged42`), then addendum B
  on the GPU, then the other 77. No arm or setting changed; the topic in progress was redone.

### Result of addendum B (2026-11-10 10:05, `run_evidence_official119.sh` stage 2)

all_strict on the official dev nuggets, same local judge; `_grounded` keeps only sentences with at
least one fully supporting citation (our support judge), `_supported` also keeps partial support:

| run | as written | supported | grounded | lost to ungrounded sentences (paired, SE) |
|---|---|---|---|---|
| shipped | 0.3560 | 0.3560 | 0.3046 | −0.0514 (0.0111) |
| evidence | 0.4881 | 0.4799 | 0.3818 | −0.1063 (0.0208) |
| multipass | 0.4904 | 0.4778 | 0.3828 | −0.1075 (0.0225) |

`evidence` − shipped: as written +0.1321 (6.8 SE); supported +0.1239 (6.1 SE); **grounded +0.0772
(3.4 SE)**. About two fifths of the rewrite's coverage gain (0.055 of 0.132) rests on sentences that
no citation fully supports; the rest survives. Even the shipped answers owe 0.05 of their coverage
to such sentences. Coverage as officially defined cannot see either.

- 2026-11-10 09:52: relaunched from stage 2 (the first launch stopped after stage 1 at 23:16: the
  validator fails a 42-topic file on 119-topic coverage, and `set -e` made that fatal; fixed). The
  42 judged narratives were all written and pass every other check. Stage 3 (the other 68 topics;
  51 of 119 were done) started 10:05.
