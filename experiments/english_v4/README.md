# English SLTR Pipeline (PHOENIX release-v3)

This directory contains a structured, reproducible workflow to:

1. Build an English pre-extracted dataset from PHOENIX release-v3 annotations
2. Train SLTR with v4 hyperparameters on that dataset
3. Generate plots and JSON metrics

## Directory layout

- `configs/sign_improved_v4_english.yaml`: training config for English dataset
- `scripts/01_build_english_dataset.sh`: create English pre-extracted train/dev/test files
- `scripts/02_train_english_v4.sh`: run training
- `scripts/03_generate_plots.sh`: generate plots and JSON after training

## Step-by-step

### 1) Build English pre-extracted dataset

```bash
cd .
bash experiments/english_v4/scripts/01_build_english_dataset.sh
```

If your release-v3 has dedicated English columns, pass them explicitly:

```bash
cd .
source venv/bin/activate
python -u create_english_dataset.py \
  --from-release-v3 \
  --release-dir /path/to/PHOENIX-2014-T-release-v3 \
  --input-dir ./data/PHOENIX2014T \
  --output-dir ./data/PHOENIX2014T_ENGLISH \
  --release-gloss-col orth_english \
  --release-text-col translation_english \
  --splits train dev test
```

Output files:

- `data/PHOENIX2014T_ENGLISH/phoenix14t_english.train`
- `data/PHOENIX2014T_ENGLISH/phoenix14t_english.dev`
- `data/PHOENIX2014T_ENGLISH/phoenix14t_english.test`

### 2) Train the model

```bash
cd .
bash experiments/english_v4/scripts/02_train_english_v4.sh
```

Model directory:

- `sign_sample_model_improved_v4_english`

### 3) Generate plots and JSON metrics

```bash
cd .
bash experiments/english_v4/scripts/03_generate_plots.sh
```

Generated assets:

- `sign_sample_model_improved_v4_english/plots_and_json/`

## Notes

- This pipeline expects PHOENIX release-v3 at:
  `/path/to/PHOENIX-2014-T-release-v3`
- Script auto-detects gloss/text columns from release CSV (prefers English-style names if available).
- For your current release files with only `orth` and `translation`, these columns are used directly unless you override.
- Sign features stay unchanged; only labels are rebuilt from release-v3 annotations.
