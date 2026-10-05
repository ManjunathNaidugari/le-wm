# Frozen V-JEPA 2 direct navigation baseline

Implementation branch: `codex/vjepa-direct-baseline`, created from the clean
`codex/gibson-pilot` checkout at `f6d260b`. No push or merge was performed.
**Runtime acceptance is pending.** No real pilot files, pretrained checkpoint,
Habitat installation or NVIDIA GPU were available on this Mac. Synthetic tests
and the official preprocessing check are local evidence only.

The user reports 15 successful expert episodes across Adrian, Albertville and
Anaheim, no runtime/video errors, and one exact repeat on Python 3.9.19 / Sim and
Lab 0.3.3. Comprehensive visual acceptance is still pending. Audit those existing
artifacts before training; a close-wall image is not automatically invalid.

## Model and data decisions

The model is the original **official V-JEPA 2 ViT-L/16**, not V-JEPA 2.1:

- Implementation: [Meta V-JEPA 2](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812),
  pinned revision `204698b45b3712590f06245fbfba32d3be539812`.
- Factory: `torch.hub.load(local_source, 'vjepa2_vit_large', source='local', pretrained=False)`.
  The official factory returns an encoder and predictor; the unused predictor is
  discarded. Only encoder `target_encoder` weights are loaded, all parameters are
  frozen, and every call uses evaluation mode and inference mode.
- Weights: [official original checkpoint](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/blob/b3c1679b7c34d3255ef3547f27c7b226aefab26f/original/model.pth),
  immutable revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`, SHA256
  `5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012`.
  This is about 5.13 GB of original training state, not just encoder weights.
  Hash/revision/package mismatches fail explicitly. Missing checkpoint keys fail;
  the upstream positional-embedding key is the only tolerated extra key.
- The pinned upstream hub currently points pretrained downloads at localhost.
  Explicitly loading the verified local original checkpoint avoids that URL and
  never falls back to random weights or another model.
- Preprocessing is the official `vjepa2_preprocessor(crop_size=256)`: deterministic
  bilinear short-side resize to 292, center crop 256, uint8 conversion to [0,1],
  ImageNet channel normalization. No navigation-specific augmentation is added.

For action `t`, only `observations[t]` and earlier **raw** uint8 RGB enter the
encoder. MP4s and `visualization_frames` are never training inputs. Default history
is 64 frames, stride one simulator observation, with index
`max(0, t - (63-i)*stride)` for slot `i`. Starts repeat observation zero; online
history resets per episode. Terminal observation T is never an action-T training
example. Changing any temporal or pooling setting requires a new cache.

The [official 64-frame, 256px training configuration](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/train/vitl16/cooldown-256px-64f.yaml)
uses video sampling at 4 FPS, patch size 16 and two-frame tubelets. Here a frame
means a discrete action-boundary observation. Turns and forward actions change the
camera differently; these 64 simulator observations are not a calibrated 16-second
video. Export FPS controls playback only. Shorter even histories are supported
for explicit experiments but deviate from the checkpoint's 64-frame setup.

Encoder tokens are time-major `[32,16,16,1024]`. Pool the temporal mean and the
latest tubelet independently into 4×4 grids, retaining their spatial order:
`features[t]` is `[32,1024]`. The latest tubelet itself covers only past/current
frames. FP32 is default; optional CUDA BF16 is recorded as a different cache and
checkpoint identity. One episode, one causal clip and one pooled-feature chunk
are held during extraction; training memory-maps at most four chunks. Full raw
trajectory/video tensors are still loaded one episode at a time.

The small policy applies a shared 1024→32 cell projection, a 3→32 goal embedding,
then a 128-unit hidden layer and four action logits (about 171,000 parameters).
Goals are robot-relative Cartesian **(forward, left, up) metres**, using the
existing quaternion convention. They contain no goal orientation. Goal mean/std
come only from training transitions; std has a 0.001m floor. Saved action IDs are
FORWARD=0, LEFT=1, RIGHT=2, STOP=3; Habitat-Lab IDs are 1,2,3,0 respectively.

Training is seeded behavior cloning with configurable AdamW, optional inverse
class-frequency weights, and unweighted mean cross-entropy for reported losses.
Reports contain class counts, confusion matrices, accuracy and per-action
precision/recall/F1 every epoch. `last.pt`, development-selected `best.pt`, and
`training.json` retain configs, encoder identity, splits and normalization. Tiny
mode selects a deterministic action-stratified subset, records exact timestep
identities, and validates on that same subset: **resubstitution/integration only**.
The goal-only ablation omits all visual features and uses the same interfaces.

## RunPod setup and first checks

Transfer this branch's working files to RunPod first; they have not been pushed.
Run commands from that repository root. Conda must be installed on local disk,
for example `/opt/conda`, not the FUSE-backed `/workspace`. Keep datasets, weights,
audit reports, caches, checkpoints, results and recovery archives on `/workspace`.
If either environment already exists, check it rather than recreating it. Preserve
the working Habitat environment; no torchvision/timm packages are added to it.

```bash
# Only create the Habitat environment if it was lost with the closed pod.
conda env create --prefix /opt/conda/envs/gibson-pilot -f environment-gibson-pilot.yml
conda env create --prefix /opt/conda/envs/vjepa-features -f environment-vjepa-features.yml
HABPY=/opt/conda/envs/gibson-pilot/bin/python
FEATPY=/opt/conda/envs/vjepa-features/bin/python
"$HABPY" -m pip install -e . --no-deps
"$FEATPY" -m pip install -e . --no-deps
"$HABPY" -m pip check
"$FEATPY" -m pip check
"$HABPY" scripts/check_pilot_environment.py
"$FEATPY" -c 'import torch, torchvision, timm, einops, PIL; print(torch.__version__, torchvision.__version__, timm.__version__, einops.__version__, PIL.__version__); assert torch.cuda.is_available()'
"$HABPY" scripts/direct_baseline.py audit \
  --run-dir /workspace/pilot-runs/three-scene-15 \
  --output /workspace/baseline/pilot-audit.json
```

Habitat keeps its earlier pins, with the required **Pillow==10.4.0** correction.
The separate worker environment pins Python3.9.19, Torch2.4.1, torchvision0.19.1,
timm1.0.9, einops0.8.0, NumPy1.26.4, Pillow10.4.0, opencv-python-headless4.10.0.84
and PyYAML6.0.2. ImageIO2.35.1/FFmpeg0.5.1 satisfy the repository's package metadata
in both environments; the worker does not import Habitat or ImageIO. Headless
OpenCV is required by upstream preprocessing imports. Upstream training
dependencies are not installed wholesale. This is a candidate reproducible Linux
environment, not an executed Linux compatibility claim. `pip check` is required.

The compact audit records each requested identity, structural/missing-file errors,
success, step/action counts, collisions, negligible-forward displacement (≤0.01m),
RGB flags and source frame indices. Collision rate divides by non-STOP actions,
including turns; negligible-forward rate divides by FORWARD actions. Zero
denominators yield zero and raw denominator counts are saved. Valid expert failures
with transitions and heuristically flagged frames remain included. Structural
failures, zero-transition recordings and learned/unknown-provenance labels are
ineligible for behavior cloning and remain in audit/failure records.
Optional `--exclusions exclusions.json` uses a mapping
like `{"Adrian/123": "researcher's documented review reason"}`; unknown IDs fail.
Audit outputs are new files and never delete source artifacts. Resolve exclusions
before caching; a changed audit requires a new cache directory.

Audit/cache schema 2 preserves controller and collection-purpose provenance.
Existing genuine pilots with `ShortestPathFollower(stop_on_error=False)` remain
compatible; rebuild older audits/caches in new locations. Unknown provenance needs
`--expert-resolutions resolutions.json`, mapping the audited `Building/episode_id`
to `{"kind":"expert","reason":"documented review","evidence":"original collection log reference"}`.
Known learned-policy recordings cannot be overridden as expert labels.

Pass `--splits` to `extract` to preflight all requested train/development inputs
before starting the worker. If labels are ineligible, repair collection or use
`resolve-training-inputs --splits original.json --audit audit.json --reason "review reason" --output derived.json`.
The derived manifest records explicit label exclusions and a new fingerprint while
preserving every evaluation request. Use it consistently for training/evaluation.
`integration-split` rejects ineligible rows rather than silently dropping identities.

Install the source and checkpoint **on RunPod**, then verify the real encoder:

```bash
mkdir -p /workspace/models /workspace/vendor
git clone https://github.com/facebookresearch/vjepa2.git /workspace/vendor/vjepa2
git -C /workspace/vendor/vjepa2 checkout 204698b45b3712590f06245fbfba32d3be539812
curl --fail --location --retry 3 \
  https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/resolve/b3c1679b7c34d3255ef3547f27c7b226aefab26f/original/model.pth \
  --output /workspace/models/vjepa2-vitl-original.pth
"$HABPY" scripts/direct_baseline.py check-encoder \
  --encoder-python "$FEATPY" --source-dir /workspace/vendor/vjepa2 \
  --checkpoint /workspace/models/vjepa2-vitl-original.pth
```

Reuse an existing verified clone/checkpoint instead of repeating the clone/download.
The check verifies hashes, dependencies, loading and a synthetic CUDA forward pass.
It is not a real-trajectory or navigation result. Capture its first error intact.
If GPU memory requires BF16, copy the feature config, set `precision: bfloat16`,
and use that same `--feature-config` for extraction and all rollouts. Do not change
the Habitat package versions to fix encoder import errors.

## Pilot integration: features, tiny training, learned rollout

```bash
"$HABPY" scripts/direct_baseline.py integration-split \
  --audit /workspace/baseline/pilot-audit.json \
  --output /workspace/baseline/pilot-integration.json
"$HABPY" scripts/direct_baseline.py extract \
  --audit /workspace/baseline/pilot-audit.json --cache-dir /workspace/baseline/pilot-cache \
  --encoder-python "$FEATPY" --source-dir /workspace/vendor/vjepa2 \
  --checkpoint /workspace/models/vjepa2-vitl-original.pth
"$HABPY" scripts/direct_baseline.py train \
  --cache-dir /workspace/baseline/pilot-cache --splits /workspace/baseline/pilot-integration.json \
  --tiny-steps 32 --epochs 200 --output-dir /workspace/baseline/tiny-direct
"$HABPY" scripts/direct_baseline.py rollout \
  --policy /workspace/baseline/tiny-direct/best.pt --splits /workspace/baseline/pilot-integration.json \
  --phase train --allow-integration --output-dir /workspace/baseline/tiny-rollouts \
  --encoder-python "$FEATPY" --source-dir /workspace/vendor/vjepa2 \
  --checkpoint /workspace/models/vjepa2-vitl-original.pth
```

Inspect tiny loss, each present action's recall and fitting accuracy before claiming
an overfit check passed. The commands do not guarantee a result in 200 epochs.
Tiny rollout executes the included pilot identities and exports videos; performance
on those training buildings is an integration result. Tiny/test checkpoints are
explicitly blocked from final evaluation.

The Habitat process keeps the policy head on CPU and launches a persistent
`FEATPY -m jepa_navigation.baseline.worker` subprocess on the GPU. A bounded binary
stdio protocol sends uint8 causal clips and receives FP32 pooled features; no
network server or pickle transport is used. **Offline and online call the same
worker preprocessing/pooling implementation.** The two-Python bridge was tested
locally with the explicitly identified test encoder; the pinned Linux worker and
real pretrained checkpoint remain to be tested together.

The online callback receives only RGB and the current relative goal. The recorder
can retain poses/maps/paths for evaluation and videos, but passes none to the
policy. Each episode starts fresh history. Latency includes history construction,
IPC, preprocessing, frozen forward/pooling and CPU action prediction, excluding
Habitat stepping, model startup and video encoding. Each trajectory and manifest
contains every action's latency, alongside mean/max latency. Results include
Lab success/SPL, final distance, steps, collisions and denominator counts. The
summary accounts for every requested identity, distinguishes navigation failure
from runtime/video errors, and reports success over all requested episodes.

For the goal-only ablation, rerun `train` with `--goal-only` and a new output
directory. Use the same split/tiny settings. `rollout` accepts the same arguments;
the goal-only checkpoint skips starting an encoder worker (the current CLI still
requires `--encoder-python`). Compare its results with the visual policy before
interpreting coordinate memorization as a contribution from vision.

## Prepare the larger experiment

First inspect **existing** official episode indices and compatible meshes:

```bash
"$HABPY" scripts/direct_baseline.py inventory \
  --episode-data /workspace/datasets/pointnav/gibson/v1/train/train.json.gz \
  --scene-data-dir /workspace/datasets/scene_datasets \
  --output /workspace/baseline/availability.json
```

If held-out official val/test indices already exist, include their paths in
`--episode-data` in that inventory command. Do not name unavailable files. Review
the inventory's errors, mesh presence and per-building episode counts. No scene
availability, final budget or scene names are assumed here. Prefer untouched
official val/test buildings for final evaluation. Using held-out training buildings
instead is a different split policy that must be stated in the thesis.

Choose 10 available train buildings, five different development buildings and
additional untouched final buildings. Set these arrays and an explicit final
budget from that review; placeholders below intentionally cannot run as-is:

```bash
TRAIN_SCENES=(REPLACE_WITH_TEN_AVAILABLE_BUILDINGS)
DEV_SCENES=(REPLACE_WITH_FIVE_DIFFERENT_AVAILABLE_BUILDINGS)
FINAL_SCENES=(REPLACE_WITH_UNTOUCHED_AVAILABLE_BUILDINGS)
FINAL_COUNT=REPLACE_WITH_CHOSEN_FINAL_BUDGET
"$HABPY" scripts/direct_baseline.py splits --inventory /workspace/baseline/availability.json \
  --train-scenes "${TRAIN_SCENES[@]}" --development-scenes "${DEV_SCENES[@]}" \
  --final-scenes "${FINAL_SCENES[@]}" --train-count 200 --development-count 50 \
  --final-count "$FINAL_COUNT" --seed 42 --output /workspace/baseline/experiment.json
"$HABPY" scripts/direct_baseline.py collect --splits /workspace/baseline/experiment.json \
  --phase train --output-dir /workspace/baseline/expert-train
"$HABPY" scripts/direct_baseline.py collect --splits /workspace/baseline/experiment.json \
  --phase development --output-dir /workspace/baseline/expert-development
"$HABPY" scripts/direct_baseline.py audit --run-dir /workspace/baseline/expert-train \
  --output /workspace/baseline/train-audit.json
"$HABPY" scripts/direct_baseline.py audit --run-dir /workspace/baseline/expert-development \
  --output /workspace/baseline/development-audit.json
"$HABPY" scripts/direct_baseline.py combine-audits \
  --inputs /workspace/baseline/train-audit.json /workspace/baseline/development-audit.json \
  --output /workspace/baseline/train-development-audit.json
"$HABPY" scripts/direct_baseline.py extract \
  --audit /workspace/baseline/train-development-audit.json --cache-dir /workspace/baseline/experiment-cache \
  --encoder-python "$FEATPY" --source-dir /workspace/vendor/vjepa2 \
  --checkpoint /workspace/models/vjepa2-vitl-original.pth
"$HABPY" scripts/direct_baseline.py train --cache-dir /workspace/baseline/experiment-cache \
  --splits /workspace/baseline/experiment.json --output-dir /workspace/baseline/direct-policy
"$HABPY" scripts/direct_baseline.py rollout --policy /workspace/baseline/direct-policy/best.pt \
  --splits /workspace/baseline/experiment.json --phase development \
  --output-dir /workspace/baseline/development-rollouts --encoder-python "$FEATPY" \
  --source-dir /workspace/vendor/vjepa2 --checkpoint /workspace/models/vjepa2-vitl-original.pth
```

One immutable split file contains explicit train/development/final sections, exact
official definitions, index/shard/mesh hashes and seed. It rejects building overlap,
duplicate identities and Adrian/Albertville/Anaheim in final evaluation. The CLI
supports other explicit budgets; 200/50 are provisional defaults, not a forced
scientific sample size. Missing/excluded selected trajectories block normal training;
repair collection or deliberately save a new split rather than silently shrinking it.

Finalize hyperparameters/checkpoint using development only. Then run the last
rollout command with `--phase final_evaluation` and a new output directory, once
the protocol is fixed. Final observations/features never participate in training
or checkpoint selection. Evaluate the goal-only checkpoint using those same saved
identities. Keep the expert workflow as the reference; `collect --videos` can
export its demonstrations without changing the pilot's 20-episode/3-building limits.

Collection/rollout `--resume` verifies the saved split/controller/settings and
committed trajectory hashes, skips completed navigation failures as well as
successes, and resumes unfinished identities. `--retry-errors` explicitly retries
runtime/video errors in a new attempt file. Every earlier attempt remains recorded;
no source episode is resampled. Feature extraction automatically verifies and
resumes committed chunks. Writers use `.writer.lock`; after a hard process kill,
first ensure the recorded process is gone and review partial artifacts, then
explicitly remove that stale lock before resuming. Changed/corrupt committed
artifacts fail instead of being overwritten. Training output directories must be
new; training itself does not currently resume optimizer state.

## Acceptance and interpretation

Required RunPod evidence: package/native checks; real pretrained encoder load and
CUDA forward; audit of the actual 15 trajectories; cache extraction and interrupted
resume; tiny real-feature overfit; learned causal rollouts with decoded videos;
review of failures and close-wall RGB; measured inference time/memory; then a
building-disjoint experiment and goal-only comparison. Preserve logs/results on
`/workspace`, and archive recovery material there while active Conda stays local.
This milestone remains pending until the relevant native/runtime checks pass.

The local evidence record is [DIRECT_BASELINE_RESULTS.md](DIRECT_BASELINE_RESULTS.md).
Behavior-cloning accuracy does not establish closed-loop success. A tiny pilot
cannot estimate generalization; 15 visually unaccepted episodes cannot justify
scaling without review. V-JEPA video-to-simulator timing differs, coarse pooling
can lose obstacle detail, 64-frame inference may be expensive, Cartesian goals
provide useful privileged task information by design, and class imbalance and
building diversity affect conclusions. No predictive dynamics, motion head or
planner is implemented or evaluated by this baseline.
