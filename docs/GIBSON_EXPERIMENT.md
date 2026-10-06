# Next experiment: unseen-building navigation

Prepared 6 October 2026. This uses the existing collection, audit, cache and policy
code. The predictor/planner is a later stage. No larger experiment has run yet.

## Protocol

The [configuration](../configs/gibson_experiment.json) requests 200 training
episodes across 10 official-training buildings, 50 development episodes across
three different official-training buildings, and reserves 100 final episodes
across five official-validation buildings. All three pilot buildings are excluded
from this new experiment. These are initial budgets, not performance guarantees.

Selection is deterministic (seed 42 and a stable name hash), independent of input
ordering and model performance. Only existing meshes with enough episodes qualify.
Ambiguous source inputs are ineligible. Insufficient assets fail explicitly;
the script never substitutes training buildings for final buildings. The saved
selection records the configuration and inventory hash; existing split machinery
records exact episode definitions and scene/source hashes. Output directories must
be new. Do not reselect buildings based on rollout performance.

Official validation is our reserved final set, not the official hidden test set.
It is not necessarily unseen to an external pretrained baseline: provenance must
be checked before any comparative final evaluation. Record known exposed buildings
in `final_excluded_buildings` before selection. This does not establish that an
empty list means there is no overlap. If protocol changes are necessary, version
the decision before inspecting final results. Do not collect, extract, tune or
run demonstrations on the reserved final episodes during development.

Keep the existing RGB sensor, oracle relative-goal input, 15-degree turns,
0.25-metre forward movement, 0.2-metre success radius and 500-action budget.
The scene-selection manifest does not itself pin simulator settings: use the
same checked-in simulator/navigation configuration for collection and preserve
the experiment commit and effective settings in run artifacts.

## Fresh RunPod setup

Run each block in order. Stop at an error and keep its output. Commands below
assume a Linux x86_64 Ubuntu/Debian pod with a root Bash terminal, and the same
persistent network volume mounted at `/workspace`. Keep Conda and active code on
local `/root` storage; keep datasets and results on the volume. The old
`le-wm-gibson-pilot.tar.gz` backup is not needed for this fresh Git checkout.

### 0. Push these new files from the Mac first

These preparation changes are local until committed and pushed. Run this block
on the **Mac**, not RunPod, if they have not already been pushed:

```bash
cd /Users/manjunathnaidugari/Documents/ChatGPT/thesis/le-wm
git branch --show-current
git add README.md docs/THESIS_PLAN.md docs/FULL_PILOT_RESULTS.md \
  docs/GIBSON_EXPERIMENT.md configs/gibson_experiment.json \
  scripts/plan_gibson_experiment.py tests/test_experiment_selection.py
git diff --cached --stat
git commit -m "Prepare building-disjoint Gibson experiment"
git push origin codex/vjepa-direct-baseline
```

The branch should be `codex/vjepa-direct-baseline`. If it is different, resolve
that before committing. If the changes were already pushed, skip this block.

### 1. Check the volume and GPU on RunPod

Use the RTX 4090 configuration that worked for the pilot if available. Inventory
and split selection themselves need no GPU, but subsequent collection needs
working NVIDIA graphics/EGL. Configure `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`
in the pod template before launch when supported; exporting it inside a running
container does not install missing driver mounts.

```bash
findmnt -T /workspace
df -h /workspace /
ls -lh /workspace/downloads/gibson_habitat.zip \
       /workspace/downloads/pointnav_gibson_v1.zip
ls /workspace/baseline/pilot-1986e39
nvidia-smi
```

Confirm `/workspace` is the network volume, not the pod's root overlay filesystem.
If archives are absent but the complete datasets are already extracted, skip
archive extraction in step 5. If both are absent, reconnect the correct volume
before downloading anything. Keep adequate free local space for Conda and PyTorch;
the previous Habitat installation occupied roughly 9 GB before adding the feature
environment. `nvidia-smi` alone does not prove CUDA or EGL initialization works.

### 2. Install Miniforge (Conda)

The installer is from the [official Miniforge repository](https://github.com/conda-forge/miniforge).
On a genuinely fresh pod:

```bash
apt-get update
apt-get install -y git curl ca-certificates build-essential unzip python3

python3 -c "import ctypes; c=ctypes.CDLL('libcuda.so.1'); print('cuInit:', c.cuInit(0))"
```

Expect `cuInit: 0`. Stop if it fails, as with the earlier broken pod.

```bash
curl -fL --retry 3 \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
  -o /tmp/miniforge.sh
bash /tmp/miniforge.sh -b -p /root/miniforge3
source /root/miniforge3/etc/profile.d/conda.sh
conda init bash
export PIP_CACHE_DIR=/root/.cache/pip
```

If `/root/miniforge3` already exists, source its `conda.sh` instead of reinstalling.

### 3. Clone the updated project

```bash
git clone --branch codex/vjepa-direct-baseline --single-branch \
  https://github.com/ManjunathNaidugari/le-wm.git /root/le-wm-gibson-pilot
cd /root/le-wm-gibson-pilot
git log -1 --oneline
ls configs/gibson_experiment.json scripts/plan_gibson_experiment.py
```

If those files are missing, the Mac changes have not reached the cloned branch.
Do not restore the old tarball over this checkout. If GitHub requests access,
authenticate using your existing GitHub credentials without placing tokens in logs.

### 4. Create and verify the Habitat environment

```bash
conda env create -f environment-gibson-pilot.yml
conda activate gibson-pilot
python -m pip install -e . --no-deps
python -m pip check
python scripts/check_pilot_environment.py
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python3 -m unittest discover -s tests -p test_experiment_selection.py -v
```

Expect no broken requirements, Python 3.9.19, Habitat-Sim/Lab 0.3.3, successful
configuration composition, CUDA available, and three passing selection tests.
Native rendering will be checked when collection starts. Do not install the
V-JEPA feature environment or download its weights for this inventory/collection
stage; those are needed later for extraction and learned rollouts.

### 5. Extract existing scenes and official episode splits

The pilot may have extracted only three scenes. Use the existing archives to
extract the remaining files; `-n` preserves files already present. Skip this block
only if the full scenes and train/val episode shards are already extracted.

```bash
unzip -tq /workspace/downloads/gibson_habitat.zip
unzip -tq /workspace/downloads/pointnav_gibson_v1.zip
mkdir -p /workspace/datasets/scene_datasets
mkdir -p /workspace/datasets/pointnav/gibson/v1

unzip -n /workspace/downloads/gibson_habitat.zip \
  -d /workspace/datasets/scene_datasets
unzip -n /workspace/downloads/pointnav_gibson_v1.zip \
  'train/*' 'val/*' -d /workspace/datasets/pointnav/gibson/v1

ls -lh /workspace/datasets/pointnav/gibson/v1/train/train.json.gz \
       /workspace/datasets/pointnav/gibson/v1/val/val.json.gz
```

Both archive integrity checks should pass. On this network filesystem, `unzip`
may warn that it cannot set permissions or timestamps, as it did in the pilot.
Those metadata warnings differ from CRC errors, truncated files or failed writes;
do not ignore the latter. The next inventory checks the mesh/episode intersection.

### 6. Inventory and materialize the split

No GPU inference is performed here. If too few assets are available, the planner
explains the shortage; share the inventory before changing the protocol. Do not
substitute training buildings into the final set to bypass an error.

```bash
cd /root/le-wm-gibson-pilot
conda activate gibson-pilot
export EXP=/workspace/baseline/gibson-generalization-v1
mkdir -p "$EXP"
set -o pipefail
git rev-parse HEAD > "$EXP/code-commit.txt"
git diff > "$EXP/source-changes.patch"
conda env export > "$EXP/habitat-environment.yml"
cp configs/gibson_experiment.json "$EXP/selection-config.json"
cp configs/simulator.yaml "$EXP/simulator.yaml"
cp configs/gibson_pilot.yaml "$EXP/navigation.yaml"

python scripts/direct_baseline.py inventory \
  --episode-data /workspace/datasets/pointnav/gibson/v1/train/train.json.gz \
                 /workspace/datasets/pointnav/gibson/v1/val/val.json.gz \
  --scene-data-dir /workspace/datasets/scene_datasets \
  --output "$EXP/availability.json"

python scripts/plan_gibson_experiment.py \
  --inventory "$EXP/availability.json" \
  --output-dir "$EXP/splits" --materialize
```

Share `availability.json`, `splits/selection.json` and `splits/experiment.json`.
They contain the actual asset-dependent building choices; none have been invented
or materialized on the Mac. Without `--materialize`, the planner saves only the
selection and prints the exact existing CLI command to materialize it later.

Stop here and share those three JSON files before collection. Existing inventory
or split outputs must not be overwritten to get a different selection. If this is
a resumed experiment, reuse them and its recorded code revision.

For a new terminal in the same pod, restore the shell state with:

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate gibson-pilot
cd /root/le-wm-gibson-pilot
export EXP=/workspace/baseline/gibson-generalization-v1
set -o pipefail
```

Before closing the pod, run `sync`, download the three planning JSON files, and
retain the network volume. Conda and source under `/root` are disposable and can
be restored on the next pod. The saved environment and commit identify this run.

## Collection and quality review

After confirming the selection, use the same manifest for both collections:

```bash
python scripts/direct_baseline.py collect \
  --splits "$EXP/splits/experiment.json" --phase train \
  --simulator-config configs/simulator.yaml --navigation-config configs/gibson_pilot.yaml \
  --output-dir "$EXP/expert-train" --videos
python scripts/direct_baseline.py collect \
  --splits "$EXP/splits/experiment.json" --phase development \
  --simulator-config configs/simulator.yaml --navigation-config configs/gibson_pilot.yaml \
  --output-dir "$EXP/expert-development" --videos
python scripts/direct_baseline.py audit \
  --run-dir "$EXP/expert-train" --output "$EXP/train-audit.json"
python scripts/direct_baseline.py audit \
  --run-dir "$EXP/expert-development" --output "$EXP/development-audit.json"
```

If collection is interrupted, repeat its command with `--resume`; do not resample.
Review flagged raw RGB sequences and a small fixed sample of unflagged sequences
across buildings. Darkness alone is not corruption, and a close wall is not by
itself a collision. Document reasons for exclusions; never use learned-policy
failure as an exclusion rule. Keep all development evaluation requests, including
episodes excluded as imitation labels. Existing `resolve-training-inputs` supports
this distinction. Keep failure denominators visible.

Stop here for review before a large encoder job. Benchmark extraction on an
explicit 10–20-episode training subset using the same BF16 feature configuration,
then estimate full extraction time/storage. Do not feed that partial cache into
full-manifest training. The existing direct-policy guide covers combined audits,
resolved manifests, resumable full extraction and training. For the real experiment
use `tiny_steps: 0`; checkpoint selection must use development buildings only.
Save success, SPL, final distance, action counts, collisions and inference latency
on the same 50 development requests. Three training seeds are desirable if budget
permits; select the protocol before final evaluation.

## External baseline decision: candidate, not yet accepted

The [Habitat-Lab v0.3.3 baseline documentation](https://github.com/facebookresearch/habitat-lab/blob/v0.3.3/habitat-baselines/README.md)
provides a pretrained archive with RGB, RGBD and depth PPO variants. Select only
the RGB variant. Baselines are a separate installation; do not alter the working
Habitat or feature environment to experiment with their dependencies.

The [v0.3.3 PPO configuration](https://github.com/facebookresearch/habitat-lab/blob/v0.3.3/habitat-baselines/habitat_baselines/config/pointnav/ppo_pointnav.yaml)
is a training recipe, not proof that the older archive loads into that policy.
Its 75-million-step budget also differs substantially from our imitation budget.
No archive weights have been downloaded, loaded or executed for this preparation.

Before accepting the baseline, record the checkpoint URL/hash and verify:

- RGB-only observations, camera geometry and preprocessing; no hidden depth input.
- Goal representation/sign convention and recurrent reset/previous-action behavior.
- Action ordering and physical movement sizes, STOP rule and episode limit.
- Training and checkpoint-selection buildings, particularly official-val exposure.
- A real checkpoint load and development rollout in an isolated environment.

Do not silently evaluate a policy trained for different turn sizes as a matched
baseline. If incompatibilities remain, reproduce a compatible established RGB PPO
policy with a documented feasible budget, or explicitly label a pretrained transfer
comparison and its limitations. Published leaderboard scores are context only.
Final comparative evaluation remains pending this decision; the train/development
pipeline can proceed while it is resolved. An internal goal-only/direct-policy
ablation supplements, and does not replace, the external comparison.
