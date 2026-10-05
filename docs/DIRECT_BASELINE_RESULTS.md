# Direct baseline evidence and handoff — 5 October 2026

**Implementation is ready for RunPod verification; runtime acceptance is pending.**
No real trajectory audit, pretrained weight inference, GPU training or learned
Habitat rollout was executed here. See [the ordered guide](DIRECT_BASELINE.md).

## Preservation and scope

Inspected the requested GitHub/local `codex/gibson-pilot` branch, which matched
`f6d260b` and had a clean working tree. Read repository guidance, README, milestones,
pilot guide/results and collector/action/coordinate/schema implementation first.
Created `codex/vjepa-direct-baseline` from that branch. Existing tracked local
environment and legacy files were preserved. No push or merge was performed.

| Changed files | Purpose |
| --- | --- |
| `src/baseline/audit.py`, `common.py` | Raw pilot/collection audit, explicit inclusion records, combined audits, hashes and atomic artifacts |
| `src/baseline/features.py`, `worker.py`, `cache.py` | Frozen official backbone adapter, shared causal preprocessing, two-Python binary bridge and resumable hash-bound chunks |
| `src/baseline/splits.py` | Actual local asset inventory and immutable building-disjoint official definitions |
| `src/baseline/policy.py`, `training.py` | Small behavior-cloning head, goal-only ablation, training-only statistics, metrics and labeled tiny overfit |
| `src/baseline/execution.py` | Restricted learned rollout and separate resumable expert collection with attempt accounting/videos/latency |
| `src/baseline/cli.py`, `__init__.py`, `scripts/direct_baseline.py`, `scripts/feature_worker.py` | Commands and executable entry points |
| `configs/vjepa_features.yaml`, `configs/direct_policy.yaml` | Explicit encoder temporal/pooling and training defaults |
| `environment-vjepa-features.yml` | Separate pinned candidate feature runtime |
| `environment-gibson-pilot.yml`, `scripts/check_pilot_environment.py` | Required Pillow10.4.0 pin and check; earlier Habitat pins preserved |
| `pyproject.toml` | Baseline package and console command |
| `tests/test_direct_baseline.py` | 21 synthetic contract/integration tests |
| `README.md`, `docs/MILESTONES.md`, `docs/GIBSON_PILOT_RESULTS.md` | Current implementation boundary and clearly attributed user-reported pilot evidence |
| `docs/DIRECT_BASELINE.md`, this file | Commands, scientific choices, evidence and pending acceptance |

The pilot's limits/behavior and expert evaluator were left intact. No predictive
world model, motion head, planner, dataset download or unrelated cleanup was added.

## Local verification

Host: Darwin arm64 / Python3.10.7; existing navigation environment Torch2.3.1,
NumPy1.26.4, ImageIO2.34.2. Habitat and official assets are absent. These versions
were not upgraded to the Linux pins.

Before changes:

```bash
.venv-navigation/bin/python -m unittest discover -s tests -v
```

Observed **41 discovered, 38 passed, three native tests skipped** (1.957 seconds).

Final verification used a temporary second Python environment containing a built
wheel of this repository, not just a second reference to the first interpreter:

```bash
BASELINE_TEST_ENCODER_PYTHON=/private/tmp/lewm-vjepa-wiring/bin/python \
  .venv-navigation/bin/python -m unittest discover -s tests -v
.venv-navigation/bin/python -m compileall -q src scripts tests
git diff --check
.venv-navigation/bin/python scripts/direct_baseline.py --help
```

The final suite discovered **62 tests: 59 passed, three skipped** (4.789 seconds). Compile, whitespace
and command help checks passed. The skipped checks require native Habitat APIs,
real direct-Sim Gibson data, and the real Lab episode/repeat runtime. Existing
ImageIO/FFmpeg resource warnings remain; fixture encode/decode assertions pass.
The binary bridge required running outside the Mac sandbox because OpenMP shared
memory was initially blocked. An old temporary setuptools installation also
produced an unusable legacy editable package; installing a wheel corrected that,
and worker startup now uses `python -m ...`, supporting both package forms.

The 21 new tests cover causal padding/stride/buffer bounds and episode reset;
future-frame independence; frozen evaluation/no gradients; raw RGB/action/goal
alignment; explicit exclusions and missing files; interrupted chunk resume;
source/encoder/config/corruption rejection; real-file availability intersections;
building/episode leakage and mandatory pilot-final protection; collection attempt
retry/settings/ledger integrity; two-Python protocol consistency; a two-transition
fixture overfit/checkpoint round trip; goal-only visual independence; training-only
normalization with a deliberately very different development goal; restricted
online inputs; learned-controller STOP/latency; and video failure retaining
simulation metrics. The tiny fixture's fitting accuracy is not a navigation result.

Additional checks:

- Built and installed the package wheel in `/private/tmp/lewm-vjepa-wiring`.
  A fresh process successfully imported the worker while Habitat, Habitat-Sim and
  ImageIO imports were explicitly made unavailable. No Habitat installation is
  required by the encoder worker itself.
- Cloned the official V-JEPA repository into `/private/tmp`, inspected the exact
  pinned factory/checkpoint keys/patch ordering/preprocessor/training config, and
  ran its actual `torch.hub.load(..., 'vjepa2_preprocessor', source='local', crop_size=256)`.
  Four synthetic 224×320 RGB frames produced `[3,4,256,256]`; a constant 128 pixel
  yielded normalized channels approximately `[0.074065,0.205182,0.426493]`.
  This check used inherited Mac Torch2.3.1/torchvision0.18.1 and temporarily installed
  timm1.0.9/einops0.8.0, with inherited OpenCV4.10.0/Pillow10.2.0, not the full
  pinned Linux runtime. Source inspection also
  identified the required OpenCV preprocessing import, now pinned in the separate
  environment. The 5.13GB pretrained checkpoint was **not** downloaded or loaded.
- Default policy size: **170,788 trainable parameters**. The frozen backbone is
  outside the optimizer, running only in the worker.

## Pending RunPod evidence

1. Recreate lost Conda environments on local disk, check native imports and pins
   (including Pillow10.4.0), verify CUDA and package dependency consistency.
2. Audit the actual `/workspace/pilot-runs/three-scene-15` trajectories and review
   flagged RGB frame indices. User-reported 15/15 success does not certify images.
3. Verify official source/checkpoint identity, real weight keys, preprocessing and
   CUDA forward with the pinned worker. Measure actual memory and latency.
4. Extract pilot features; interrupt/resume extraction; fit the labeled tiny subset
   with real features; run the learned policy through the real Habitat+worker bridge
   and decode/review the exported videos. Inspect STOP and runtime/navigation errors.
5. Inspect available scenes/official episodes before choosing actual 10/5/final
   building lists. Collect/audit/train on those saved definitions; select checkpoints
   using development only; compare goal-only and visual policies before final testing.

Keep the final buildings untouched until the evaluation protocol is fixed. The
64-frame video pretraining cadence differs from action-boundary simulator sampling;
coarse spatial pooling, inference cost, coordinate-goal informativeness, class
imbalance and behavior-cloning distribution shift limit interpretation. Neither
fixture overfit nor training-building rollout measures held-out generalization.
