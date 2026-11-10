# BiTeM at the TREC 2026 RAG Track

Code, runs and analysis scripts for the notebook paper

> **BiTeM at the TREC 2026 RAG Track: Facet Retrieval, Citation Support and the Limits of
> Self-Built Judges.** Esteban Gaillac, Julien Knafou, Paul Van Rijen, Alexandre Flament, Patrick
> Ruch. HES-SO and SIB Swiss Institute of Bioinformatics, Geneva. TREC 2026.

The paper source is in [`paper/`](paper/). Every number in it is produced by a script in this
repository, and the LaTeX comment next to each claim names the script or file.

## What is here

| path | content |
|---|---|
| `systems/rag/agentic_v2/` | the pipeline: search plan (`blueprint.py`), pooling (`harvest.py`), reranking (`tighten.py`), anchor-first fusion (`fuse.py`), writing and review (`synthesize.py`), the probe-guided plan (`probe_blueprint.py`), the agent loop (`agent_loop.py`), orchestration (`pipeline.py`) |
| `systems/rag/` | API retrieval client, cross-encoder reranker, answer parser |
| `scripts/` | the depth rule (`build_variable_depth_runs.py`), the writer variants (`run_synth_arms.py`) and the drivers of the post-results experiments |
| `eval/` | our judges: relevance labeller (`umbrela_local_label.py`), support judge (`ragdoll_local_judge.py`), nugget judge (`nugget_local_judge.py`), pairwise judge (`pairwise_arena_local.py`), defect detector, dev evaluation |
| `eval/analysis/` | every analysis in the paper, with its JSON output, and the design of the post-results experiments as fixed in writing beforehand (`PREREG_post_results_rerun.md`) |
| `runs/submitted/` | the seven runs we submitted: `bp_agentic_retrieval`, `bp_probe_ret`, `BiTeM_deep_v2`, `BiTeM_deep_probe`, `BiTeM_poolenrich` (retrieval) and `bp_agentic_v2`, `bp_probe_guided`, `bp_manual_curated`, `bp_support_argmax` (generation) |
| `runs/dev22/` | development runs on the 22 development topics, including the post-results writer variants |
| `runs/bp_v2_evidence_official119_judged42.jsonl` | the post-results rewrite of our answers to the 42 test topics judged for generation; never submitted |
| `trec_shared/` | small shared helpers (run and qrels I/O, Ollama client, JSON parsing) |
| `paper/` | LaTeX source, bibliography, figures and the script that draws them |

## What is not here

- **Corpus text.** The pipeline caches ClimbMix documents per topic (`data/rag/cache/`); these are
  not redistributed. Rebuild them with the pipeline and an API token.
- **The organisers' data and scores.** Topics, development labels and nuggets come from
  [TREC-RAG/trec-rag-data](https://github.com/TREC-RAG/trec-rag-data); the official score files
  were sent to participants and are expected under `data/rag/official_scores/`.
- **RAGDoll.** Our judges import the organisers' prompts and metrics from
  [castorini/RAGDoll](https://github.com/castorini/RAGDoll); clone it to `eval/RAGDoll/`.

## Requirements

Python 3.10 with `torch`, `sentence-transformers`, `faiss-cpu`, `numpy`, `scipy`, `scikit-learn`,
`matplotlib`, `requests`, `httpx` and `ollama`. Local models served by Ollama: `qwen3.5:122b`
(plan, writing, pairwise judge) and `qwen3:32b` (relevance, support and nugget judges). A search
token for the ClimbMix API, in `PYSERINI_API_TOKEN` or `.env.local`.

## Running

```bash
python systems/rag/agentic_v2/pipeline.py --phase all --topic-set dev22          # or official119
python scripts/build_variable_depth_runs.py                                       # the depth rule
python eval/analysis/paper_revision_checks.py                                     # development figures
python eval/analysis/official_retrieval_scores.py                                 # needs the official scores
python paper/figures/make_paper_figures.py                                        # Figures 2-6
```

Each stage writes its output to disk, and re-running a command recomputes only what is missing.

## Licence

No licence has been chosen yet; until one is, contact the authors before reusing the code.
