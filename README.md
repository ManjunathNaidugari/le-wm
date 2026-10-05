# Goal-conditioned indoor navigation using visual latent representations

Bachelor's thesis comparing two planned approaches to robot-relative coordinate goals:

- **Direct goal-conditioned policy:** visual representation + relative goal → discrete action.
- **Action-conditioned latent predictor:** `z_t + action → predicted z_(t+1)`, used with short-horizon planning.

**The first frozen V-JEPA 2 direct baseline is implemented for runtime verification.**
Its pretrained GPU extraction, real-data training and learned Habitat execution
remain unverified. Follow [the direct baseline guide](docs/DIRECT_BASELINE.md).
The predictive model and planner remain future work. Legacy LeWM code is retained for
reference, not as the thesis architecture; see [repository audit](docs/REPOSITORY_AUDIT.md).
The original upstream MIT [license](LICENSE) is preserved.

## Status

| Milestone | Status |
|---|---|
| Habitat-Sim smoke test | Complete; verified on Linux / RTX 3090 / EGL |
| Deterministic expert collection | Complete |
| RGB/data validation | In progress; heuristic QC requires real-data review |
| Gibson PointNav integration | User reports 15 successful RunPod pilot episodes; comprehensive visual acceptance pending |
| V-JEPA extraction | Implemented; official preprocessing checked locally, pretrained GPU execution pending |
| Direct baseline | Implemented; synthetic wiring/overfit tests pass, real-data training and closed-loop checks pending |
| Latent predictor/planner | Later |

Navigation success does not establish valid rendering. Previously observed dark/clipped
castle episodes motivate automated RGB checks and selective video review.

## Environment and basic commands

Linux x86_64 with working EGL graphics drivers and Conda. Keep the existing
`lewm-habitat-smoke` environment name for compatibility. Habitat-Sim is pinned to
0.3.3; no Habitat-Lab installation or upstream training dependencies are required.

```bash
conda env create -f environment-habitat.yml  # once; skip on configured RunPod
conda activate lewm-habitat-smoke
python -m unittest discover -s tests -v
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/habitat_data
python scripts/smoke_habitat.py --output-dir /workspace/smoke_check
python scripts/collect_habitat_dataset.py --num-episodes 10 --seed 42 --output-dir /workspace/castle_check
python scripts/inspect_habitat_dataset.py --dataset-dir /workspace/castle_check --qc
python scripts/inspect_habitat_dataset.py --dataset-dir /workspace/castle_check --episode 0
```

Run from the repository root. Output directories must be new; no automatic resume
or overwrite. Missing Habitat/assets cause errors, never synthetic observations.
Smoke exits 0 on success, 1 on rollout failure, 2 on setup error. Collection exits
0 when all requested rollouts are saved (including failures), 2 on error.

## Current Gibson pilot

Use the [Gibson pilot guide](docs/GIBSON_PILOT.md) for the new isolated
Habitat-Sim **0.3.3** + Habitat-Lab **0.3.3** reference workflow. It selects
explicit **train** buildings, preserves official episode definitions, records
every attempted rollout and automatically exports RGB + top-down MP4s. Keep
future final evaluation buildings separate. The direct-Sim environment above
remains available for the existing commands.

```bash
python scripts/gibson_pilot.py run \
  --episode-data /workspace/datasets/pointnav/gibson/v1/train/train.json.gz \
  --scene-data-dir /workspace/datasets/scene_datasets \
  --scenes Allensville --episode-id 0 \
  --output-dir /workspace/pilot-runs/one
```

Create `environment-gibson-pilot.yml` and supply the matching external assets
first. See the guide for batch, validation, export and repeatability commands,
[verification evidence](docs/GIBSON_PILOT_RESULTS.md), and
[milestone boundaries](docs/MILESTONES.md). Comprehensive visual acceptance
remains pending. The user reports successful Linux execution on
Adrian, Albertville and Anaheim; that report is recorded separately from local
verification. Local fixture tests cannot establish rendering or learned performance.

## Existing direct-Sim PointNav tools

Supply externally obtained Gibson scene meshes and official Habitat PointNav-v1
JSON/JSON.gz episodes. No proprietary assets are included or downloaded by collection.
See [PointNav and QC details](docs/HABITAT_DATA.md) for layouts, assumptions and thresholds.

```bash
python scripts/collect_habitat_dataset.py --dataset gibson \
  --scene-data-dir /workspace/datasets/scene_datasets \
  --episode-data /workspace/datasets/pointnav/gibson/v1/train/train.json.gz \
  --num-episodes 20 --seed 42 --output-dir /workspace/gibson_test
python scripts/inspect_habitat_dataset.py --dataset-dir /workspace/gibson_test --qc
python scripts/inspect_habitat_dataset.py --dataset-dir /workspace/gibson_test --qc --export-flagged
python scripts/inspect_habitat_dataset.py --dataset-dir /workspace/gibson_test --episode 0 5
```

Inspect success rate, scene counts, QC report and a few videos before scaling up.
Official starts, quaternion rotations and goals are preserved. No random replacement
is permitted. The seed controls deterministic selection and simulator RNGs.

## Data contract

Actions are **FORWARD=0, TURN_LEFT=1, TURN_RIGHT=2, STOP=3**. For T actions,
RGB uint8 observations `[T+1,3,H,W]`, positions `[T+1,3]`, headings `[T+1]`, and
relative goals `[T+1,3]` include initial and final states. Actions/collisions have T
entries. Transition t pairs state t with action t and state t+1.

World coordinates are XYZ, Y up. Heading zero faces -Z; positive turns toward -X.
Robot-relative goals are `(forward, left, up)` in metres. Files also store the fixed
goal, initial geodesic path, metrics and provenance. `manifest.json` is updated
after each episode; interrupted runs are marked incomplete. Failed rollouts are kept.
`HabitatTrajectoryDataset(..., success_only=True)` excludes navigation failures;
QC flags are separate and must be reviewed before future training.

## Storage

| Location | Purpose |
|---|---|
| GitHub | Source code, tests, configuration, documentation |
| RunPod `/root` | Disposable runtime, repository, environment |
| RunPod `/workspace` | Persistent datasets, checkpoints, representations, results |

Keep large artifacts outside Git. No cleanup command in this project deletes
external datasets. All legacy training/Colab instructions are unsupported for this milestone.
