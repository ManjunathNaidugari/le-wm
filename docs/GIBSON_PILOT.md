# Gibson pilot: setup and execution

This command runs official development PointNav episodes through Habitat-Lab
0.3.3, with Habitat-Sim 0.3.3 rendering the scene. It reuses the existing trajectory
recorder and schema. The existing direct Habitat-Sim commands remain available.
Nothing downloads assets, provisions a pod or collects training data automatically.

## 1. Supply the smallest assets

First inspect your existing external data folder. The current Mac checkout has
no Gibson meshes/episodes, and `/workspace/datasets` is absent. Do not download
anything already present on the GPU machine.

The official [Habitat task dataset table](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md)
lists [pointnav_gibson_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip)
(385 MB). Obtain that archive manually, and retain its **train** index plus the
selected building shards. Match the extracted inner folder structure, rather
than nesting `v1` twice:

```text
/workspace/datasets/pointnav/gibson/v1/train/train.json.gz
/workspace/datasets/pointnav/gibson/v1/train/content/Allensville.json.gz
```

Scene meshes require your agreement to Gibson's terms. Follow the official
[Gibson data instructions](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md)
and [access form](https://forms.gle/36TW9uVpjrE1Mkf9A). Request/select the
**Gibson Database for Habitat-sim** archive (Habitat table lists about 1.5 GB),
which contains `<building>.glb`, rather than the original GibsonEnv database.
The gated form determines the download URL and archive filename; those are not
publicly documented, so a filename is deliberately not invented here.
Extract/copy initially only:

```text
/workspace/datasets/scene_datasets/gibson/Allensville.glb
```

For a 15-episode, three-building pilot, also copy `Beechwood.glb` and
`Benevolence.glb`, and their matching train content shards. These are train
buildings in [Gibson's official metadata](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/data.json).
Confirm their presence in your actual official Habitat archive before running;
if a building is absent, report that mismatch and explicitly choose another
train building. No semantic meshes, GibsonEnv installation or habitat-baselines
are required. Lab's default navmesh configuration uses the effective agent
height/radius and can build navigation data; no separately acquired `.navmesh`
is needed. Supply embedded textures/dependencies if your asset package includes
them. A disconnected or unnavigable official start/goal is recorded as an error,
not snapped, resampled or replaced.

Pass the **parent of `gibson/`** as `--scene-data-dir`. Only the exact
`data/scene_datasets/` prefix is removed from official scene IDs. Selected
buildings do not require other scene meshes or readable unrelated shards.
You may pass a selected `train/content/<building>.json.gz` directly.

## 2. Create a separate Linux environment on RunPod

Use an existing Linux x86_64 NVIDIA GPU pod with persistent `/workspace` storage.
The user must select/provision it; this implementation does not do that. Make
the **current working files** available there: this local branch contains
uncommitted code, so cloning GitHub alone will not include the pilot. Copy this
working checkout (excluding `.git`, virtual environments and assets), or commit
locally and transfer it yourself. No push/merge was performed.

From that copied repository root, with Conda available:

```bash
nvidia-smi
conda env create -f environment-gibson-pilot.yml
conda activate gibson-pilot
python -m pip install -e . --no-deps
python -m pip check
python scripts/check_pilot_environment.py
python -m pip freeze > /workspace/gibson-pilot-package-freeze.txt
```

Keep your existing environment. The new environment pins Python 3.9.19,
Habitat-Sim 0.3.3/headless, NumPy 1.26.4 and Habitat-Lab v0.3.3 at immutable
commit `094d6be2f9d057e4781a68ae792132895fd4d3d0`, installing **only** its
`habitat-lab` subpackage. Hydra/OmegaConf, OpenCV, quaternion, SciPy and Numba
pins avoid recent NumPy-2/Python-version conflicts. Headless rendering uses EGL
and the pod's NVIDIA driver; it is unavailable on this Apple Silicon machine.
No Bullet physics, baseline trainer or GibsonEnv package is needed.

The local isolated check successfully composed the actual upstream Lab config
with the pinned Hydra/OmegaConf/NumPy versions. Full Conda solving, native
imports and GPU rendering **still require verification on Linux**. If environment
creation or the check fails, preserve the error output before changing pins.

## 3. Check and run one episode first

Commands below run from the repository root. Pick an explicit ID from the selected
official shard; `0` is an example and the command fails if it is absent. The check
does not start simulation or write outputs:

```bash
python scripts/gibson_pilot.py check \
  --episode-data /workspace/datasets/pointnav/gibson/v1/train/train.json.gz \
  --scene-data-dir /workspace/datasets/scene_datasets \
  --scenes Allensville --episode-id 0

python scripts/gibson_pilot.py run \
  --episode-data /workspace/datasets/pointnav/gibson/v1/train/train.json.gz \
  --scene-data-dir /workspace/datasets/scene_datasets \
  --scenes Allensville --episode-id 0 \
  --output-dir /workspace/pilot-runs/one

python scripts/gibson_pilot.py validate /workspace/pilot-runs/one
```

The runtime check is separate: `check` reports file/module presence, not version
or EGL correctness. `check_pilot_environment.py` verifies native imports and
configuration. The actual `run` establishes whether the chosen scene renders.
`run` automatically saves `episode_000000.pt`, `episode_000000.mp4` and
`manifest.json`. The MP4 has first-person RGB beside Habitat's colored top-down
map, with its agent, trail, source, goal and shortest path. All T+1 states are
included, including the initial state and the terminal STOP observation.
Open this first MP4 and inspect several frames before proceeding to the batch.
Camera/agent settings live in `configs/simulator.yaml`; navigation settings live
in `configs/gibson_pilot.yaml`. Pass `--simulator-config`, `--navigation-config`,
`--seed`, or `--max-steps` to override them. Fractional turn/HFOV angles are
rejected because Lab 0.3.3's schema requires integers; they are never rounded.

Output folders must be new and separate from source directories. Existing folders
or videos are refused. Exit codes: 0 = all attempted episodes succeeded with
video; 1 = executed navigation failure/repeat mismatch; 2 = setup/runtime/video
error. Every planned attempt gets an entry, so inspect the manifest even after
a nonzero exit. Interrupted runs retain pending/running entries and
`complete=false`; no automatic resume or episode replacement is performed.

## 4. Small batch after the first real video works

```bash
python scripts/gibson_pilot.py run \
  --episode-data /workspace/datasets/pointnav/gibson/v1/train/train.json.gz \
  --scene-data-dir /workspace/datasets/scene_datasets \
  --scenes Allensville Beechwood Benevolence --num-episodes 15 \
  --output-dir /workspace/pilot-runs/batch
python scripts/gibson_pilot.py validate /workspace/pilot-runs/batch
```

Selection is seeded, unique and round-robin across the selected buildings. The
pilot caps execution at 20 episodes and three buildings. The manifest preserves
the selected source IDs, successes, failures, reasons, steps, collision counts,
travelled distance, final geodesic distance, simulation time, total time and
trajectory/video bytes for **every** attempt. Simulator initialization, navmesh
work and first scene hashing are included in `simulation_seconds`; encoding is
included in `total_seconds`. The runner currently creates one simulator per
episode; timings should not be treated as optimized bulk collection throughput.
Inspect several MP4s from different buildings and both successful/failed rollouts.
Do not silently discard failures or switch to easier episodes. Reuse the existing
RGB quality helper `jepa_navigation.data.rgb_qc.rgb_quality(trajectory)` on
individual saved trajectories if needed; there is no new image quality subsystem.
The legacy dataset inspection CLI expects the old manifest layout, so use the
pilot's `validate` command for this attempt ledger.

## 5. Export and compare a repeated episode

Video can be exported again from saved real frames without rerunning simulation:

```bash
python scripts/gibson_pilot.py export \
  /workspace/pilot-runs/one/episode_000000.pt \
  --output /workspace/pilot-runs/one/reexport.mp4 --fps 10

python scripts/gibson_pilot.py run \
  --episode-data /workspace/datasets/pointnav/gibson/v1/train/train.json.gz \
  --scene-data-dir /workspace/datasets/scene_datasets \
  --scenes Allensville --episode-id 0 \
  --output-dir /workspace/pilot-runs/one-repeat

python scripts/gibson_pilot.py compare \
  /workspace/pilot-runs/one/episode_000000.pt \
  /workspace/pilot-runs/one-repeat/episode_000000.pt
```

Repeat selected batch episodes too, using their recorded scene/source ID and
the same seed/settings. Each source episode uses the global seed directly,
so repeating it alone does not change its seed due to batch position.
Comparison requires matching source/scene hashes, definition, versions and
effective settings (except output destination). Actions, collisions and RGB
must match exactly. Poses/quaternions/headings/goals report exact equality plus
an absolute tolerance of `1e-6` by default; `--atol 0` requires exact poses too.
Metrics must match. Differences cause a nonzero exit and a field-by-field report.

## Coordinate/action/termination mapping

| Meaning | Existing saved action | Habitat-Lab 0.3.3 action |
| --- | ---: | ---: |
| Forward | 0 | 1 |
| Left | 1 | 2 |
| Right | 2 | 3 |
| STOP | 3 | 0 |

Both use a Y-up world and official XYZW start quaternions. Yaw zero faces -Z;
positive yaw turns left toward -X. The recorder stores relative goals in metres
as **(forward, left, up)**. This is explicitly different from Lab's optional
polar pointgoal sensor; the pilot computes the existing Cartesian representation
from the full pose, and disables that unused task sensor. Raw RGB remains
`uint8 [T+1,3,H,W]`, actions `int64 [T]`, poses/headings/relative goals T+1,
and collisions `bool [T]`. Additional XYZW rotations and composite video frames
also contain T+1 entries. Existing schema version 1 consumers can ignore these
additional fields.

The old direct-Sim collector uses `STOP and distance <= radius`. The new reference
pilot reads Lab's **strict** `STOP and geodesic distance < success_distance`.
It explicitly sets both the follower stopping radius and success distance to the
official goal radius when supplied, otherwise configured 0.2 m. The official
[shortest-path example](https://github.com/facebookresearch/habitat-lab/blob/v0.3.3/examples/shortest_path_follower_example.py)
instead uses forward-step size as the fallback follower radius, which can differ
from the success measure. Keeping the pilot thresholds equal avoids that gap;
the choice is saved in metadata. STOP terminates without movement; the final
state/observation is duplicated. Movement never auto-terminates on proximity.
500 actions without STOP is a timeout, not success. Follower errors are preserved
as partial trajectories (Lab's automatic error-to-STOP behavior is disabled).

`collect_episode(env, controller=None)` accepts a callable
`controller(current_state) -> Action`; recording/evaluation stay unchanged when
a future learned policy replaces the expert. No learned policy is implemented.

## Acceptance and tests

```bash
python -m unittest discover -s tests -v

HABITAT_GIBSON_SCENE_DIR=/workspace/datasets/scene_datasets \
HABITAT_GIBSON_EPISODES=/workspace/datasets/pointnav/gibson/v1/train/content/Allensville.json.gz \
GIBSON_PILOT_SCENE=Allensville \
python -m unittest discover -s tests -p test_gibson_pilot.py -v
```

The opt-in real test executes the same supplied episode twice and requires
success and matching rollouts. Unit fixtures only verify contracts; their test
MP4s are not simulation evidence. `validate` checks every saved trajectory,
initial definition, finite values, quaternion/relative-goal alignment, STOP,
metric agreement, manifest values, storage sizes, decoded MP4 dimensions and
T+1 frame count. It never converts a unit test into real-data acceptance.

Accept this milestone only after native environment verification, a real
single-episode video, the 10–20-episode batch, review of several videos,
failure explanations, repeat comparisons and actual timing/storage evidence.
`complete` in the manifest means all attempts were accounted for. The separate
`visual_review` and `real_data_acceptance` fields remain pending for human review.
Current evidence and blockers are in [GIBSON_PILOT_RESULTS.md](GIBSON_PILOT_RESULTS.md).
