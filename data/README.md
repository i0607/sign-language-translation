# Data directory

Place **PHOENIX-2014-T** preprocessing artifacts here so paths in `configs/*.yaml` resolve (default: `./data/`).

Typical layout:

- `PHOENIX2014T/` — German splits and feature files as distributed with the official Sign Language Transformer release.
- `PHOENIX2014T_DE_EN_V2/` — bilingual German/English split files expected by `configs/sign_de_en_*.yaml` (features are still the same sign stream; only the text side and manifest differ).
- Other English-only or custom layouts — add a subfolder and set `data.train` / `data.dev` / `data.test` in your YAML accordingly.

Features are not included in this repository because of size and licensing; obtain them from the original CVPR 2020 project page and dataset documentation.
