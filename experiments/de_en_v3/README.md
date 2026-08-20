# Stage 1.6 - DE+EN Unified SLTR (v3 Fine-tune)

This iteration starts from the best `de_en_v2` checkpoint and fine-tunes for better combined DE+EN quality.

## Why v3 (fine-tune)

- Keep bilingual data setup from v2 (`PHOENIX2014T_DE_EN_V2`).
- Initialize from `sign_sample_model_de_en_v2/best.ckpt` for stable continuation.
- Use a lower LR with scheduler/optimizer reset to avoid carrying stale optimization dynamics.
- Slightly rebalance loss weights:
  - recognition: `2.2` (from 2.5)
  - translation: `1.2` (from 1.0)

## Run

```bash
cd .
bash experiments/de_en_v3/scripts/01_train_de_en_v3_finetune.sh
```

## Generate plots

```bash
cd .
bash experiments/de_en_v3/scripts/02_generate_plots.sh
```

## Success criteria

- Improve over `de_en_v2` best validation BLEU-4 (`11.44`) without regressing WER trend.
- Maintain bilingual tag behavior (`<de>` / `<en>` consistency).
- Produce better tuned-beam DEV and TEST metrics than v2 final eval.
