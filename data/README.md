# Data directory

Place **PHOENIX-2014-T** preprocessing artifacts here so paths in `configs/*.yaml` resolve (default: `./data/`).

## Required layout

```
data/
├── PHOENIX2014T/
│   ├── phoenix14t.pami0.train
│   ├── phoenix14t.pami0.dev
│   └── phoenix14t.pami0.test
├── PHOENIX2014T_ENGLISH/          # built by scripts/create_english_dataset.py
│   ├── phoenix14t_english.train
│   ├── phoenix14t_english.dev
│   └── phoenix14t_english.test
└── PHOENIX2014T_DE_EN_V2/         # built by scripts/build_de_en_dataset_v2.py
    ├── phoenix14t_de_en_v2.train
    ├── phoenix14t_de_en_v2.dev
    └── phoenix14t_de_en_v2.test
```

## How to obtain German features

Features are **not** included in this repository (size + licensing). Obtain the official SignJoey / Camgoz et al. (CVPR 2020) PHOENIX-2014-T release pickles and place them under `PHOENIX2014T/` as above.

English and bilingual folders are **generated** (see `REPRODUCE.md`); do not commit them.
