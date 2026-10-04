
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Checkpoints &amp; Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

### Step-by-Step Training Guide

#### 1. Environment Setup

First, set up the Python environment:

```bash
# Create virtual environment with Python 3.10
uv venv --python=3.10
source .venv/bin/activate

# Install dependencies
uv pip install stable-worldmodel[train,env]
```

#### 2. Configure WandB (Optional)

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To disable WandB logging, set `wandb.enabled: false` in the same file.

#### 3. Download and Prepare Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

#### 4. Launch Training

To launch training on the Push-T dataset:
```bash
python train.py data=pusht
```

Available datasets:
- `data=pusht` - Push-T manipulation task
- `data=dmc` - DeepMind Control tasks
- `data=ogb` - Object Goal Navigation
- `data=tworoom` - Two-Room navigation task

#### 5. Training Configuration

Key hyperparameters in `config/train/lewm.yaml`:
- `trainer.max_epochs`: Number of training epochs (default: 100)
- `trainer.devices`: GPU devices to use (default: auto)
- `trainer.accelerator`: Set to `cpu` for CPU-only training
- `loader.batch_size`: Batch size (default: 128)
- `optimizer.lr`: Learning rate (default: 5e-5)

For CPU-only training (slower but works on laptops without GPU):
```bash
python train.py data=pusht trainer.accelerator=cpu trainer.precision=32
```

For different batch sizes (useful for limited VRAM):
```bash
python train.py data=pusht loader.batch_size=64
```

#### 6. Checkpoints

Checkpoints are saved to `$STABLEWM_HOME/checkpoints/` upon completion. The checkpoint structure will be:
```
$STABLEWM_HOME/checkpoints/<run_id>/
├── config.yaml          # Training configuration
├── lewm_weights.ckpt    # Weights-only checkpoint
└── lewm_object.ckpt     # Full model object (for evaluation)
```

### Training on a Laptop

For laptop training with limited resources:

1. **Reduce batch size** if you encounter OOM errors:
   ```bash
   python train.py data=pusht loader.batch_size=32
   ```

2. **Use CPU** if no GPU is available:
   ```bash
   python train.py data=pusht trainer.accelerator=cpu trainer.precision=32
   ```

3. **Reduce number of epochs** for quick experiments:
   ```bash
   python train.py data=pusht trainer.max_epochs=10
   ```

4. **Monitor resource usage**: The model has ~15M parameters and can train on a single GPU in a few hours. On CPU, expect significantly longer training times.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pretrained LeWM checkpoints for each environment are mirrored on the Hugging Face
Hub (model repos), alongside the datasets (dataset repos) in the same collection:

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube)
- [`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms)
- [`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)

The full baseline checkpoint suite (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop)
is available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e):

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

### From the Drive archive

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

Place the extracted files under `$STABLEWM_HOME/` and load via:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

`AutoCostModel` accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

### From the Hugging Face mirror

The HF model repos ship the LeWM checkpoint as a `weights.pt` (state dict) plus a
`config.json` describing the model. Convert once to produce the `_object.ckpt`
that `eval.py` expects:

```bash
# download weights.pt + config.json
hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht

# convert to object checkpoint under $STABLEWM_HOME/pusht/lewm_object.ckpt
python - <<'PY'
import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**cfg["predictor"]),
    action_encoder=Embedder(**cfg["action_encoder"]),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
PY
```

After conversion, load via `swm.policy.AutoCostModel('pusht/lewm')` as usual.

## Habitat indoor-navigation smoke test

This milestone runs one real RGB episode with **Habitat-Sim 0.3.3 only**.
There is no Habitat-Lab dependency, PointNav episode dataset, training, or MPC.
The former Habitat collector was a stub: it mixed continuous labels with discrete
execution, paired post-action images with pre-action poses, indexed a quaternion
incorrectly, steered directly through obstacles, and silently generated random data.
Those paths have been replaced. Old trajectories must be recollected.

### Setup and run

Use Linux x86_64 with an EGL-capable graphics driver (for example an NVIDIA GPU
with its driver installed), Conda, and network access for packages and test assets.
The headless environment below is not a macOS environment. The smoke test does
not need the upstream LeWM training dependencies or CUDA-enabled PyTorch.

```bash
conda env create -f environment-habitat.yml
conda activate lewm-habitat-smoke
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/habitat_data
python scripts/smoke_habitat.py
```

Run these commands from this repository's root. The freely downloadable
`skokloster-castle.glb` indoor test scene is the default; no HM3D credentials are
needed. Package installation and scene download are explicit, never automatic.
See the [official 0.3.3 installation and test-scene instructions](https://github.com/facebookresearch/habitat-sim/blob/v0.3.3/README.md).

The script creates an RGB camera, rebuilds a navmesh for a 1.5m tall / 0.1m radius
agent, samples a connected start and goal 2–10m apart, and uses Habitat's
`ShortestPath` and `GreedyGeodesicFollower` to follow obstacle-aware geodesics.
Sampling does not guarantee the selected route contains a bend. Forward motion
is 0.25m; turns are 15 degrees. Success requires STOP within 0.2m geodesic distance.
STOP is a terminal no-op with a fresh sensor observation. No frame skipping occurs.

```text
outputs/videos/episode_0001.mp4
outputs/trajectories/episode_0001.pt
```

The console reports success/failure, executed action count (including STOP),
travelled path length, final geodesic distance, and termination reason. Exit codes:
0 success, 1 failed rollout (including step limit or follower failure), 2 setup or
output error. Failed rollouts are saved too. Existing artifacts are not overwritten;
use `--output-dir outputs/another_run`. Options also include `--scene`, `--seed`,
`--max-steps`, `--resolution` (positive even pixels), and `--fps`.

### Trajectory contract

For T actions, observations and poses contain **T+1 states**. Transition t is
`observations[t], positions[t], headings[t], relative_goals[t], actions[t],
observations[t+1], positions[t+1], headings[t+1]`.

- `observations`: uint8 RGB `[T+1, 3, H, W]`, including initial and final frames.
- `actions`: int64 `[T]`: FORWARD=0, TURN_LEFT=1, TURN_RIGHT=2, STOP=3.
- `positions`: float32 `[T+1, 3]` in Habitat world XYZ, metres, Y up.
- `headings`: float32 `[T+1]`, radians; zero faces -Z, positive turns toward -X.
- `goal_position`: float32 `[3]`, fixed world XYZ goal.
- `relative_goals`: float32 `[T+1, 3]`, robot (forward, left, up), metres.
- `collisions`: bool `[T]`, corresponding to each executed action; STOP is false.
- `shortest_path`: initial navmesh path points. `metrics` stores outcome and distances;
  `metadata` stores scene, seed, version, and movement/camera playback settings.

Quaternion geometry uses Habitat's `quat_rotate_vector` and quaternion inverse.
`HabitatTrajectoryDataset` returns explicit current/next transition windows and
rejects legacy data. It does not claim compatibility with continuous-action LeWM
training. `configs/habitat_coord.yaml`, `configs/habitat_colab.yaml`, and the old
Colab notebook are legacy, unvalidated training drafts, not smoke-test inputs.

### Verification and troubleshooting

```bash
python -m unittest discover -s tests -v
```

Unit tests use explicit test doubles for alignment and termination; the demo never
uses these. The quaternion API test skips if Habitat is absent. Only a successful
real smoke command validates the renderer, navmesh, follower, and downloaded scene
together. Missing Habitat, wrong versions, absent scenes, and failed navmeshes are
errors; random frames are never substituted. EGL initialization errors require a
working graphics driver and compatible headless Habitat build.

---

## Collect a small Habitat expert dataset

The single-episode smoke test was verified by the project owner on Linux with
Habitat-Sim 0.3.3, RTX 3090/EGL and the real castle scene: 31 actions, 32 RGB
frames, 6.251m travelled, 6.290m initial geodesic distance, and 0.055m final
distance. All eight smoke tests passed there. The dataset extension needs its own
multi-episode validation on that environment.

From the repository root in the existing RunPod environment:

```bash
conda activate lewm-habitat-smoke
python -m unittest discover -s tests -v
python scripts/collect_habitat_dataset.py --num-episodes 100 --output-dir data/habitat_expert --seed 42
python scripts/inspect_habitat_dataset.py --dataset-dir data/habitat_expert
python scripts/inspect_habitat_dataset.py --dataset-dir data/habitat_expert --episode 0
```

No additional dependencies are needed. Collection accepts `--scene`,
`--max-steps` (500 by default), and `--resolution` (224 by default). It reuses the
verified collector, simulator and navmesh. Before every reset, simulator and
pathfinder are seeded with `(global_seed + episode_id) % 2**31`; the same seed
also controls the initial orientation. Episode zero therefore uses seed 42.
Reproducibility assumes the same scene assets, simulator build, settings and
hardware/software environment; it is not a promise of identical bytes across
platforms. Exact repeated start/goal/heading triples cause a clear error instead
of silently adding duplicate episodes.

Each attempted rollout produces `episode_000000.pt`, `episode_000001.pt`, etc.,
using the smoke test's T+1/T tensor schema. Failed expert rollouts are retained.
`manifest.json` records scene, episode ID/seed, success, termination, steps,
initial/final geodesic distances, travelled distance, collision count, and relative
trajectory path. An unreachable final distance is JSON `null` (the trajectory
retains infinity). Each file is written via a temporary file and rename; the
manifest is updated after each episode. `complete: false` identifies interrupted
collections. Setup, sampling, validation and unexpected simulator errors stop
collection; no dummy episode is generated. A crash between trajectory rename and
manifest update can leave an unlisted file, which inspection flags explicitly.
Automatic resume is not implemented. Choose a fresh output directory for each run;
existing directories are refused. Exit 0 means collection finished, not that all
episodes succeeded; exit 2 means a setup/collection error.

Inspection validates every trajectory and cross-checks the manifest before
printing totals, successes, failures, success rate, mean/median action count,
mean initial geodesic distance and mean collision count (all episodes included).
Validation checks action IDs, state/action lengths, tensor shapes/dtypes, finite
geometry, STOP placement, metrics and non-empty successes. Entirely constant
successful RGB sequences are rejected; this is a basic corruption check, not a
substitute for watching rendered frames. `--episode 0` also writes all T+1 RGB
frames to `data/habitat_expert/videos/episode_000000.mp4` (10 fps; override with
`--fps`). Existing videos are not overwritten. Summary-only inspection needs no
Habitat installation and exports no video.

For downstream dataset access, successful episodes are selected by default:

```python
from datasets.habitat_dataset import HabitatTrajectoryDataset
successful = HabitatTrajectoryDataset("data/habitat_expert", num_steps=4)
all_rollouts = HabitatTrajectoryDataset("data/habitat_expert", num_steps=4, success_only=False)
```

Windows never cross episode boundaries. Episodes shorter than `num_steps` supply
no windows; an all-failure directory gives an empty success-only dataset.
The loader validates trajectories; use the inspection command to additionally
validate manifest consistency. No training integration is added here.
Generated episode tensors and MP4s are ignored by Git; keep custom dataset roots
ignored too if storing manifests outside `data/habitat_expert*` or `outputs/`.
At 224x224, uncompressed RGB alone is about 15 MB per 100-state episode, or 75 MB
at the 500-action limit; reserve several GB for 100 episodes.

On RunPod, first collect a small batch in a fresh directory and inspect several
videos, then repeat the same seed/settings in a second directory. Compare action,
pose and RGB tensors and confirm starts/goals/headings vary between episode IDs.
Run the 100-episode command after that check; review failures and the success-only
window count before using this dataset for any later milestone.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
