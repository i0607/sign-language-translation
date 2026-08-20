# Stage 1.7 - DE+EN Unified SLTR (v4 Lang-Head Ablation)

This iteration keeps the shared encoder-decoder backbone and adds language-specific
translation output heads to reduce DE/EN cross-language interference.

## What changes in v4

- Shared visual encoder and shared text decoder remain unchanged.
- Translation projection is split into language-conditioned heads:
  - DE head for `<de>`
  - EN head for `<en>`
- Training starts from `de_en_v3` best checkpoint with non-strict loading enabled
  so new head parameters are initialized and then fine-tuned.

## Run

```bash
cd .
bash experiments/de_en_v4_langhead/scripts/01_train_de_en_v4_langhead_finetune.sh
```

## Success criteria

- Improve combined DEV/TEST BLEU-4 over `de_en_v3`.
- Reduce language-mix outputs (wrong target language generation).
- Keep WER stable (no severe regression from `de_en_v3`).

## Final results (2026-05-07 run)

Training best checkpoint by validation greedy BLEU-4: **step 13,600** (dev BLEU-4 **23.18**). Grid on dev selects **CTC BW=10, TX BW=8, \(\alpha=3\)**.

| Split | WER \(\downarrow\) | BLEU-4 \(\uparrow\) | CHRF | ROUGE |
|-------|---:|---:|---:|---:|
| Dev | 66.97 | 24.23 | 47.36 | 50.82 |
| Test | **66.04** | **24.67** | 47.98 | 50.70 |

Test WER breakdown: DEL 26.91, INS 1.18, SUB 37.96.
