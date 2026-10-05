# Gibson pilot handoff — 5 October 2026

Implementation is ready for remote verification. **Real-data acceptance is pending.**
No official Gibson episode was executed here, and no simulation MP4 is claimed.

## Repository preservation

Initial branch: `codex/navigation-cleanup`. Its HEAD and the requested starting
branch `codex/gibson-pointnav` both pointed to `5c5f0ee`. The checkout already
contained substantial uncommitted deletions/restructuring, a `src/` package,
configuration files, scripts and a local `.venv-navigation` environment.

Created local `codex/gibson-pilot` from `codex/gibson-pointnav`, retaining that
working tree. No reset, stash, cleanup, deletion, commit, push or merge was
performed. Existing data/backups/environments were preserved. The large deletion
diff predates this task and must not be attributed to the pilot.

## Files added by this milestone

| File | Purpose |
| --- | --- |
| `src/simulator/lab_pointnav_env.py` | Official Lab task execution, start verification, reference metrics and Habitat maps |
| `src/data/gibson_pilot.py` | Small attempt ledger, provenance, validation, timing/storage and repeat comparison |
| `src/utils/pilot_cli.py` | Check/run/validate/export/compare commands |
| `scripts/gibson_pilot.py` | Simple script entry point |
| `scripts/check_pilot_environment.py` | Linux native imports, pinned versions and config verification |
| `scripts/check_pilot_configuration.py` | Actual upstream config check without native-runtime imports |
| `configs/gibson_pilot.yaml` | Train development paths and pilot navigation defaults |
| `environment-gibson-pilot.yml` | Separate pinned Linux runtime; Sim 0.3.3 + Lab 0.3.3 |
| `tests/test_gibson_pilot.py` | Focused contracts/failure/video/repeat tests and opt-in real integration test |
| `docs/GIBSON_PILOT.md` | Ordered RunPod setup, assets, execution and acceptance guide |
| `docs/MILESTONES.md` | Six-stage roadmap and current boundary |
| `docs/GIBSON_PILOT_RESULTS.md` | This evidence and handoff record |

## Existing working files modified narrowly

- `src/data/pointnav.py`: scene/source-ID filters before unrelated shard/asset access,
  optional deferred scene existence checks and preserved raw official definitions.
- `src/navigation/actions.py`: tested explicit Lab/saved ID conversions.
- `src/data/habitat_collector.py`: optional controller callback, XYZW/composite state
  recording and Lab success metrics; existing callers keep their behavior.
- `src/utils/video.py`: optional composite export; existing RGB exports remain supported.
- `pyproject.toml`: `jepa-gibson-pilot` console entry point.
- `README.md`: current pilot entry point and guide links; train split in development example.

## Actual local results

Host: Darwin arm64, Python 3.10.7. The existing local environment includes NumPy
1.26.4, Torch 2.3.1, ImageIO 2.34.2 and PyYAML 6.0. It uses system site packages;
it was **not upgraded** to the proposed remote pins. Neither `habitat_sim` nor
`habitat` is installed. No `.glb`, `.navmesh` or official `.json.gz` assets were
found in the thesis workspace. Configured `/workspace/datasets` is absent here.

| Check | Observed result |
| --- | --- |
| Existing suite before implementation | 28 discovered, 26 passed, 2 runtime tests skipped |
| Final `python -m unittest discover -s tests -v` | **41 discovered, 38 passed, 3 runtime tests skipped**, 2.830 s |
| `python -m compileall -q src scripts tests` | Passed |
| `git diff --check` | Passed |
| Pilot CLI `--help` | Passed; all five subcommands available |
| Pilot check against configured official train path | Exit 2: expected missing `train/train.json.gz`; no synthetic fallback |
| Actual upstream Lab 0.3.3 config composition in isolated temporary venv | Passed with NumPy 1.26.4 / Hydra 1.3.2 / OmegaConf 2.3.0 |
| `pip check` in isolated configuration environment | No broken requirements |
| Real Sim/Lab/Gibson integration | Not run; native runtime/assets unavailable |
| Real videos/visual review/repeatability/timing/storage | Pending Linux execution |

The three skipped tests are the real Habitat quaternion API, the existing real
direct-Sim Gibson episode, and the new real Lab episode/repeat test. Local video
tests encode and decode **fixture** frames and verify T+1 frames/dimensions;
they are not evidence of simulator rendering. The existing ImageIO version emits
resource warnings for FFmpeg subprocess streams; the encoding/decoding assertions
pass. This does not verify the remote pinned ImageIO build.

For the isolated configuration check, cloned the official Lab tag to a temporary
directory and verified commit `094d6be2f9d057e4781a68ae792132895fd4d3d0`.
The real upstream Python configuration modules were loaded without the package's
native simulator imports. No simulator/episode APIs were mocked for that check.
This exposed the integer `turn_angle` schema constraint, now handled explicitly.
It establishes config compatibility on Python 3.10, **not** full native package
compatibility on the pinned Linux/Python 3.9 environment.

Reproduce that limited check in a separate Python 3.9/3.10 venv with the three
configuration dependencies installed:

```bash
python scripts/check_pilot_configuration.py --lab-source /path/to/habitat-lab-v0.3.3
```

## Source verification and decisions

Read the official [Habitat-Sim repository](https://github.com/facebookresearch/habitat-sim),
[Habitat-Lab repository](https://github.com/facebookresearch/habitat-lab),
[task dataset table](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md),
[Gibson access instructions](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md),
and the requested [v0.3.3 shortest-path example](https://github.com/facebookresearch/habitat-lab/blob/v0.3.3/examples/shortest_path_follower_example.py).
Also inspected the pinned task config, episode loader, Env reset/step behavior,
follower, action enumeration, Success/SPL measures and visualization code.

The pilot uses Lab's strict STOP-plus-geodesic-distance success metric, an explicit
action conversion, official XYZW start rotations and existing Cartesian robot
goals. It records exact definitions, file/scene SHA256, effective configs,
simulator/package versions, actions, rotations, RGB and collision indicators.
Lab's automatic follower-error-to-STOP conversion is disabled. Each planned
source identity survives in the manifest whether it succeeds or fails.

Proposed first building: Allensville; later Beechwood/Benevolence, 15 episodes.
Those buildings are training scenes in the official Gibson metadata; actual
matching Habitat archive shards must still be supplied and checked. The guide
contains exact public episode archive/access links and destinations. The gated
scene archive filename cannot be verified without your access; the required
product is **Gibson Database for Habitat-sim**, not the full GibsonEnv database.

## Next action and remaining acceptance work

Follow [GIBSON_PILOT.md](GIBSON_PILOT.md) in order:

1. Complete the Gibson access form yourself and supply `Allensville.glb` plus
   official `train/train.json.gz` and `train/content/Allensville.json.gz` at the
   documented external destinations, checking existing pod assets first.
2. Transfer the current working files to an existing Linux NVIDIA GPU pod and
   create/verify the **new** `gibson-pilot` environment. Preserve the first native
   setup error if there is one; full environment/GPU compatibility is pending.
3. Execute and validate one explicitly selected official episode. Review its
   actual RGB/top-down MP4 before expanding.
4. Execute 10–20 episodes across 2–3 selected train buildings, inspect several
   videos, explain every failure, and record measured time/storage from manifests.
5. Repeat selected source IDs with identical settings and compare actions,
   poses and RGB. Keep the final evaluation scenes separate.

No model, training dataset, cloud resources or broad cleanup was added. The
milestone remains pending until this real-runtime evidence is available.
