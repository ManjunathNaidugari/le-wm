# Repository audit — 2026-10-05

Audited tracked files, Python imports, Hydra targets/config paths, tests, scripts,
and notebook/config references before cleanup. No external data was inspected or deleted.

| Classification | Paths | Decision / evidence |
|---|---|---|
| CURRENT | `envs/`, `data/`, `datasets/`, `scripts/`, `tests/`, `environment-habitat.yml` | Habitat simulation, collection, transition loading and tests; preserve. New PointNav/QC modules belong here. |
| CURRENT | `README.md`, `.gitignore`, `LICENSE`, `docs/` | Rewrite project instructions, expand artifact ignores, retain upstream license and attribution. |
| FUTURE | `models/planner.py`, `models/__init__.py` | Unused coordinate encoder draft; potentially useful as a reference. Retain, not a working planner or thesis baseline. Its old `(x,z,yaw)` assumptions differ from current relative goals. |
| LEGACY | `train.py`, `eval.py`, `jepa.py`, `module.py`, `utils.py`, `config/` | Original LeWM stack. `train.py` imports module/utils; Hydra targets reference jepa/module and train/eval configs. Preserve together; not imported by current Habitat pipeline. No V-JEPA implementation. |
| LEGACY | `configs/habitat_coord.yaml`, `configs/habitat_colab.yaml`, `notebooks/habitat_navigation_colab.ipynb`, `README_COLAB.md` | Unvalidated training drafts with cross-references and obsolete commands. Leave untouched except README_COLAB link repair. Do not run for current work. |
| LEGACY (removed) | `assets/lewm.gif` | Original promotional animation, referenced only by the replaced upstream README; unrelated to current navigation pipeline. |
| GENERATED | trajectory `.pt`, checkpoints, scene assets, videos, representations, caches, results, RunPod outputs | None found among tracked data artifacts. Ignore future outputs; retain source directories `data/` and `datasets/`. Never delete external `/workspace` data. |

No code module was removed solely because it is unused. Current executables do
not depend on the original LeWM training stack. The upstream MIT license remains.
