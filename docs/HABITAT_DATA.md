# Habitat PointNav collection and RGB quality control

## External Gibson assets

Use matching Gibson meshes and official Habitat PointNav-v1 episode JSON/JSON.gz
files, acquired separately under their applicable dataset access terms. No scene
or episode content is embedded in this repository. The parser follows the
[Habitat-Lab 0.3.3 PointNav loader](https://github.com/facebookresearch/habitat-lab/blob/v0.3.3/habitat-lab/habitat/datasets/pointnav/pointnav_dataset.py)
for the standard scene prefix and content shard layout. Habitat-Lab need not be installed.

Example layout (names below are illustrative, not included assets):

```text
/workspace/datasets/scene_datasets/gibson/<scene>.glb
/workspace/datasets/pointnav/gibson/v1/val/val.json.gz
/workspace/datasets/pointnav/gibson/v1/val/content/<scene>.json.gz
```

Pass the **parent of `gibson/`** as `--scene-data-dir` when episode scene IDs are
`data/scene_datasets/gibson/<scene>.glb` or `gibson/<scene>.glb`.
The exact `data/scene_datasets/` prefix is removed; the remainder is joined to
that root. For bare `<scene>.glb` IDs, pass the folder containing the meshes.
No recursive basename search, absolute episode scene paths or root-escaping paths
are accepted. The requested scene must exist; missing selected scenes abort rather
than biasing the batch by silently dropping episodes.

`--episode-data` accepts a monolithic JSON/JSON.gz, a per-scene shard, or a split
index with `content_scenes_path` (default `{data_path}/content/{scene}.json.gz`).
Shards are sorted before reading. A seeded reservoir retains at most N candidate
episodes per scene, then deterministic scene round-robin selection exposes multiple
scenes in a small batch. Selection is not the original file order or a benchmark
sampling protocol. Source scene ID plus episode ID identifies an official episode.
Selected duplicates, malformed/non-unit quaternions, nonfinite poses, missing
assets, and unsupported multi-goal definitions cause errors; no replacement is made.
JSON parsing loads one shard at a time (large monolithic files still need RAM).

## Simulator assumptions and provenance

- Habitat-Sim **0.3.3**, Python/dependency pins in `environment-habitat.yml`;
  Linux x86_64/EGL is the supported runtime. Existing smoke environment unchanged.
- Official start XYZ, start quaternion **XYZW**, and the single goal XYZ are
  used directly. `quat_from_coeffs` converts XYZW to Habitat's quaternion type.
  Goal radius, if supplied, is used; otherwise success radius is 0.2m.
- The current smoke agent is retained: height 1.5m, radius 0.1m, RGB camera at
  `[0,1.5,0]`, square resolution 224 by default, Habitat's default 90-degree HFOV,
  forward 0.25m and turn 15 degrees. No actuation noise or frame skipping.
- Navmeshes are recomputed for that agent. A start/goal must be navigable with
  at most 0.1m vertical discrepancy and have a finite geodesic path. The actual
  supplied coordinates are never snapped or changed. Mismatched assets/navmesh
  settings may therefore fail clearly. No 2–10m restriction is applied to official episodes.
- This uses official episode definitions, **not the full Habitat-Lab benchmark
  task/evaluation protocol**. Do not compare its success rate to published Gibson
  benchmark scores without matching camera, agent, actions, success and step limits.
- One simulator is opened/closed per official episode for simple scene isolation.
  The old test-scene mode still reuses its simulator and retains its seed behavior.
- Trajectory metadata includes source dataset, file path/SHA256, source scene and
  episode IDs, exact source start/rotation/goal, source radius, simulator version,
  action parameters, camera settings and navmesh/agent settings. Local episode IDs
  remain zero-based stable filename indices. Manifest rows retain source IDs too.

Known expert failure/step limit rollouts are saved with failure metrics. Asset,
pose or unexpected simulator errors abort; the manifest remains incomplete.
Files are written atomically individually; interrupted writes can leave an orphan
trajectory, which manifest validation reports. No automatic resume or cleanup.

## RGB QC: data quality heuristics, not scientific metrics

```bash
python scripts/inspect_habitat_dataset.py --dataset-dir /workspace/gibson_test --qc
python scripts/inspect_habitat_dataset.py --dataset-dir /workspace/gibson_test --qc --export-flagged
python scripts/inspect_habitat_dataset.py --dataset-dir /workspace/gibson_test --episode 0 5 12
```

QC visits **every `.pt` file**, writes `qc_report.json` (replaced on a new QC run),
and never changes/deletes a trajectory or its success label. Corrupt/unreadable
files get report entries too. Structural/manifest errors still cause inspection
exit 2 after the QC report is written. RGB-only flags do not fail inspection.
Inspection uses structural validation independently of RGB quality, so selected
black episodes can also be exported without `--qc`. Video export requires structurally valid data; unreadable/NaN files cannot be
rendered as valid RGB. Existing videos are skipped rather than overwritten.
Selected IDs and flagged IDs are combined if both options are supplied.

All intensities are on the raw 0–255 scale. The report contains image dimensions,
NaN/Inf counts, per-frame mean intensity and variance, per-frame and aggregate
pixel ratios, consecutive-frame change, agent min/max world Y, flags and QC status.
The console prints a compact row per episode; JSON includes all per-frame details.

Initial thresholds in `data/rgb_qc.py`:

| Check | Definition / flag |
|---|---|
| Near-black pixel | All RGB channels ≤5; flag any frame with ≥80% such pixels |
| Saturated pixel | Any channel ≥250 (includes near-white); flag any frame with ≥80% |
| Low variance | Population variance across a frame's RGB values ≤4; flag any frame |
| Frozen pair | Mean absolute channel difference between consecutive frames ≤0.5 |
| Frozen episode | ≥80% frozen non-STOP transitions, with at least 3 non-STOP pairs |
| Invalid numeric data | Flag NaN/Inf, non-uint8 RGB, out-of-range values or invalid dimensions |

Raw frozen ratio includes STOP for transparency; `motion_frozen_ratio` excludes
STOP, whose repeated image is intentional. Collisions may legitimately freeze
frames. Dark rooms, bright walls and stationary views can create false positives;
clipped geometry can also escape these heuristics. Review flagged/random videos
and calibrate thresholds on real data before treating any episode as usable.
Y range is diagnostic only: multi-floor scenes are not rejected by an arbitrary Y limit.

Gibson collection preserves structurally valid black/constant RGB episodes for QC.
The original smoke/test-scene collector's existing constant-image safeguard remains.
The loader's existing success/constant-image checks remain conservative: success
filtering alone is **not** QC filtering. No automated training selection is added.

## Small RunPod validation (manual)

After activating the existing environment, run tests without asset variables first.
Then collect only 10–20 episodes to a fresh persistent output directory using the
README command. Review scene counts, success/failure reasons, geodesic distances,
RGB report and selected/flagged videos. Repeat the same seed/settings into another
fresh directory and compare source metadata and tensors; reproducibility is limited
to compatible assets, simulator and runtime versions. Do not scale automatically.

An optional one-episode real-asset test can be enabled explicitly:

```bash
HABITAT_GIBSON_SCENE_DIR=/workspace/datasets/scene_datasets \
HABITAT_GIBSON_EPISODES=/workspace/datasets/pointnav/gibson/v1/val/val.json.gz \
python -m unittest discover -s tests -p test_pointnav.py -v
```

That test verifies execution/schema/source coordinates, not rendering quality or a
required success rate. It otherwise skips clearly. Local unit tests use explicit
synthetic image/episode fixtures only; there is no production fallback data path.
