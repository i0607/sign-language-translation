# Reproduce paper experiments (UAIS)

All commands assume the repository root and an activated `venv` unless noted.

```bash
cd /path/to/slt
source venv/bin/activate
```

---

## 0. Prerequisites

1. **Python 3.8+**, then:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **PHOENIX-2014-T** pre-extracted 1024-d features in Joey/SignJoey format:
   ```
   data/PHOENIX2014T/phoenix14t.pami0.{train,dev,test}
   ```
3. **PHOENIX-2014-T release-v3** annotations (for English columns), e.g.:
   ```
   /path/to/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/
   ```
4. Optional Stage-2 env (NLLB / COMET):
   ```bash
   python3 -m venv venv-stage2 && source venv-stage2/bin/activate
   pip install -r requirements-stage2.txt
   ```

Hardware note in the paper: Apple Silicon Mac, **CPU** (`use_cuda: false` in YAML). GPU works if you set `use_cuda: true`.

---

## 1. Build English silver-standard dataset

Uses release-v3 columns `orth_english` / `translation_english` (no live MT required if those columns already exist):

```bash
python -u scripts/create_english_dataset.py \
  --from-release-v3 \
  --release-dir /path/to/PHOENIX-2014-T-release-v3 \
  --input-dir ./data/PHOENIX2014T \
  --output-dir ./data/PHOENIX2014T_ENGLISH \
  --release-gloss-col orth_english \
  --release-text-col translation_english \
  --splits train dev test
```

Outputs:
- `data/PHOENIX2014T_ENGLISH/phoenix14t_english.{train,dev,test}`

---

## 2. Stage-1 monolingual models

| Paper name | Config | Model directory |
|------------|--------|-----------------|
| Model-ODE (best DE) | `configs/sign_improved_v4.yaml` | `sign_sample_model_improved_v4` |
| Model-OEN (best EN) | `configs/sign_improved_v4_english.yaml` | `sign_sample_model_improved_v4_english` |
| Model-A (ablation) | `configs/sign_improved_v5.yaml` | `sign_sample_model_improved_v5` |

```bash
# Model-ODE
python -m signjoey train configs/sign_improved_v4.yaml

# Model-OEN (needs English data from §1)
python -m signjoey train configs/sign_improved_v4_english.yaml

# Model-A
python -m signjoey train configs/sign_improved_v5.yaml
```

Exploratory configs (optional): `sign_improved_v1.yaml` … `v3.yaml`.

After training, SignJoey writes best checkpoints and runs decode-grid evaluation on test when configured.

---

## 3. Build bilingual DE+EN manifests

```bash
python -u scripts/build_de_en_dataset_v2.py \
  --de-dir ./data/PHOENIX2014T \
  --en-dir ./data/PHOENIX2014T_ENGLISH \
  --out-dir ./data/PHOENIX2014T_DE_EN_V2 \
  --splits train dev test
```

Outputs: `data/PHOENIX2014T_DE_EN_V2/phoenix14t_de_en_v2.{train,dev,test}`

---

## 4. Unified DE+EN models (v2 → v3 → v4)

Train **in order** (each fine-tune loads the previous `best.ckpt`):

```bash
# v2 from scratch (shared softmax)
python -m signjoey train configs/sign_de_en_v2.yaml

# v3 fine-tune from v2
python -m signjoey train configs/sign_de_en_v3_finetune.yaml

# v4 language-specific translation heads from v3
python -m signjoey train configs/sign_de_en_v4_langhead_finetune.yaml
```

Model dirs (see YAML `training.model_dir`):
- `sign_sample_model_de_en_v2`
- `sign_sample_model_de_en_v3`
- `sign_sample_model_de_en_v4_langhead`

Optional per-language metrics:

```bash
python scripts/eval_de_en_per_language.py \
  --model-dir sign_sample_model_de_en_v4_langhead
```

---

## 5. Stage-2 cascade (English pivot → FR / AR)

```bash
source venv-stage2/bin/activate   # or venv with transformers installed

# Export English pivots from DE+EN-v4 test hypotheses
python scripts/export_pivot_for_nllb.py \
  --model-dir sign_sample_model_de_en_v4_langhead \
  --split test \
  --lang en \
  --output experiments/stage2_multilingual/data/pivot_en_test.txt \
  --ids-output experiments/stage2_multilingual/data/pivot_en_test_ids.txt

# French
bash experiments/stage2_multilingual/scripts/01_translate_from_pivot.sh \
  experiments/stage2_multilingual/data/pivot_en_test.txt \
  experiments/stage2_multilingual/outputs/predictions_fra_test.txt \
  fra_Latn eng_Latn

# Arabic
bash experiments/stage2_multilingual/scripts/01_translate_from_pivot.sh \
  experiments/stage2_multilingual/data/pivot_en_test.txt \
  experiments/stage2_multilingual/outputs/predictions_ar_test.txt \
  arb_Arab eng_Latn
```

Paper evaluation bundle (COMET-QE, error tags, human-eval sample, bootstrap CI):

```bash
bash experiments/stage2_multilingual/scripts/03_paper_evaluation_tasks.sh
```

Human rating CSVs are written under `experiments/stage2_multilingual/outputs/` — fill them manually; they are not regenerated as gold answers.

---

## 6. Bootstrap CIs only

```bash
python scripts/bootstrap_metrics_ci.py --n-bootstrap 1000 --seed 42
```

---

## Expected artefacts (after a full run)

| Artefact | Location |
|----------|----------|
| Checkpoints | `sign_sample_model_*/best.ckpt` |
| Test hyps | `sign_sample_model_*/best.*.test.txt` |
| EN pivots | `experiments/stage2_multilingual/data/pivot_en_test.txt` |
| FR / AR MT | `experiments/stage2_multilingual/outputs/predictions_*.txt` |
| COMET / errors | `experiments/stage2_multilingual/outputs/*.json` |

These directories are **gitignored** on purpose.

---

## Config → paper name map

| Config | Paper |
|--------|-------|
| `sign_improved_v4.yaml` | Model-ODE |
| `sign_improved_v4_english.yaml` | Model-OEN |
| `sign_improved_v5.yaml` | Model-A |
| `sign_de_en_v2.yaml` | DE+EN-v2 |
| `sign_de_en_v3_finetune.yaml` | DE+EN-v3 |
| `sign_de_en_v4_langhead_finetune.yaml` | DE+EN-v4 |

---

## Pack a clean zip for sharing

```bash
bash scripts/pack_clean_release.sh
```

Creates `../SLTR-clean-release.tar.gz` next to this repo (code + configs + scripts only).
