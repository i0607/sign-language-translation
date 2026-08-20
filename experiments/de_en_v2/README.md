# Stage 1.5 - DE+EN Unified SLTR (v2)

This is the minimal corrective iteration after `de_en_v1` underperformed.

## What changed

- Strict bilingual pairing: only keep samples that exist in both DE and EN.
- Balanced duplication: each matched sign yields exactly two outputs (`__de`, `__en`).
- Stronger language conditioning: prepend `<de>` / `<en>` to both `text` and `gloss`.
- Fresh model directory (`sign_sample_model_de_en_v2`) for clean comparison.

## Run

```bash
cd .
bash experiments/de_en_v2/scripts/01_build_de_en_dataset_v2.sh
bash experiments/de_en_v2/scripts/02_train_de_en_v2.sh
bash experiments/de_en_v2/scripts/03_generate_plots.sh
```

Resume training (same `model_dir`, loads `best.ckpt`, does not wipe the directory):

```bash
cd .
bash experiments/de_en_v2/scripts/02_train_de_en_v2_resume.sh
```

## Success criteria

- Translation: BLEU-4 (dev) should clearly improve over `de_en_v1` baseline.
- Recognition: WER (dev) should decrease from `de_en_v1` baseline.
- Tag behavior: first generated token should match requested language tag for most samples.
