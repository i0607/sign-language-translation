# Stage-2 multilingual cascade

Text-only second stage: English (or German) SLT pivots → NLLB-200 → French / Arabic.

## Scripts

| Script | Role |
|--------|------|
| `scripts/01_translate_from_pivot.sh` | NLLB batch translate one pivot file |
| `scripts/02_run_cascade_full.sh` | End-to-end export + FR (+ optional AR) |
| `scripts/03_paper_evaluation_tasks.sh` | COMET-QE, error tags, human sample, bootstrap |
| `scripts/translate_pivot_text.py` | Python NLLB driver |
| `scripts/score_comet_kiwi.py` | Reference-free COMET-QE |
| `scripts/analyze_cascade_errors.py` | Heuristic tags (leak / truncation / missing-clause) |
| `scripts/sample_human_eval.py` | Sample 20 pairs for human rating |

Pivot export lives in repo root: `scripts/export_pivot_for_nllb.py`.

See root `REPRODUCE.md` §5.
