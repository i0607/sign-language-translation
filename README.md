# Sign Language Transformers (Joey-Sign)

PyTorch code for joint continuous **sign language recognition** (gloss, CTC) and **sign-to-text translation** (transformer encoder–decoder), as in Camgoz et al., CVPR 2020. This tree is based on the public SignJoey release and includes small extensions used in our multilingual DE/EN work (for example optional language-specific decoder heads via `lang_head_tokens` in the decoder section of a config).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Populate `data/` as described in `data/README.md`.

## Train and evaluate

```bash
python -m signjoey train configs/sign.yaml
```

Adjust `data_path`, splits, and `model_dir` in the YAML as needed. After training, the driver runs dev/test evaluation when test data are listed in the config.

## Configuration

Files under `configs/` follow the same schema as the upstream project. Training hyperparameters, loss weights, and architecture blocks (`encoder`, `decoder`) are set there. For bilingual setups, vocabulary must contain any control tokens (for example `<de>`, `<en>`) referenced in the data.

## License

See `LICENSE`. When publishing work built on this code, cite the original Sign Language Transformers paper and respect dataset terms for PHOENIX-2014-T.
