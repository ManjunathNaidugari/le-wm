
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

## LeWM Indoor Navigation with Habitat

This section provides step-by-step instructions for training LeWorldModel on **Habitat indoor navigation tasks**. The implementation uses continuous velocity commands (linear, angular) and supports both coordinate-based and image-based goal specifications.

### Overview

The Habitat integration consists of three components:
1. **Data Collection** (`data/habitat_collector.py`) - Collects offline trajectories using shortest-path heuristic with Gaussian noise
2. **Dataset Loading** (`datasets/habitat_dataset.py`) - PyTorch dataset for loading trajectory files
3. **Environment Wrapper** (`envs/habitat_wrapper.py`) - Gym wrapper for evaluation with LeWM-compatible observations

### Step 1: Install Habitat Dependencies

In addition to the base LeWM dependencies, install Habitat:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Install habitat-lab and habitat-sim
# Note: habitat-sim requires compilation and may take 10-20 minutes
pip install habitat-lab habitat-sim

# Alternative: Use conda for easier habitat-sim installation
conda install -c conda-forge habitat-sim habitat-lab
```

**Troubleshooting:** If habitat-sim compilation fails, try:
```bash
# Ensure you have build tools installed
sudo apt-get update && sudo apt-get install -y build-essential cmake libglm-dev

# Or use pre-built binaries (if available for your system)
pip install habitat-sim --no-build-isolation
```

### Step 2: Prepare Habitat Scene Data

Download Habitat scene datasets (e.g., HM3D, Gibson, MatterPort3D):

```bash
# Example: Download HM3D train data
mkdir -p data/habitat_data
cd data/habitat_data

# Download from Habitat website or use existing scenes
# For testing, you can use the builtin tiny scenes
```

Create a Habitat configuration file (e.g., `configs/habitat_pointnav.yaml`):

```yaml
ENVIRONMENT:
  MAX_EPISODE_STEPS: 500
SIMULATOR:
  AGENT_0:
    SENSORS: ['RGB_SENSOR']
    ACTION_SPACE_CONFIG: "v0"
  HABITAT_SIM_V0:
    GPU_DEVICE_ID: 0
    ALLOW_SLIDING: True
  TURN_ANGLE: 15
  FORWARD_STEP_SIZE: 0.25
TASK:
  TYPE: Nav-v0
  POSSIBLE_ACTIONS: ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
  SENSORS: ['GPS', 'COMPASS']
  GOAL_SENSORS: ['POINTGOAL_WITH_GPS_COMPASS_SENSOR']
DATASET:
  TYPE: PointNav-v1
  SPLIT: train
  DATA_PATH: "data/habitat_data/{scene}/{scene}_{split}.json.gz"
  SCENES_DIR: "data/habitat_data/"
```

### Step 3: Collect Training Data

Collect offline trajectories using the provided script:

```bash
# Collect 10,000 episodes (adjust based on your needs)
python data/habitat_collector.py \
  --config configs/habitat_pointnav.yaml \
  --num_episodes 10000 \
  --save_dir data/habitat_trajectories \
  --seed 42
```

**What gets collected:**
- RGB frames at 224×224 resolution
- Continuous actions: `(linear_velocity, angular_velocity)` in range [-1, 1]
- Agent coordinates: `(x, z, yaw)` for each timestep
- Goal information: target coordinates and optional goal images

**Expected output:**
```
Collected 10000 episodes to data/habitat_trajectories
```

Each episode is saved as a `.pt` file containing:
```python
{
    "observations": (T, 3, 224, 224),   # RGB frames
    "actions": (T, 2),                   # (linear, angular) velocities
    "coordinates": (T, 3),               # (x, z, yaw)
    "goal_coord": (3,),                  # Target (x, z, yaw)
    "goal_image": (3, 224, 224)          # Optional goal image
}
```

### Step 4: Configure Training

Create a Habitat-specific training config. A reference config is provided at `configs/habitat_coord.yaml`:

```yaml
defaults:
  - _self_
  - launcher: local
  - model: lewm

output_model_name: lewm_habitat_coord
subdir: ${hydra:job.id}

num_workers: 6
train_split: 0.9
seed: 3072
img_size: 224

# Habitat-specific data configuration
data:
  dataset:
    num_steps: ${eval:'${wm.num_preds} + ${wm.history_size}'}
    frameskip: 5
    name: habitat_coord
    keys_to_load:
      - pixels
      - action
      - coordinates
    keys_to_cache:
      - action
      - coordinates
    trajectory_dir: data/habitat_trajectories

trainer:
  max_epochs: 50  # Adjust based on dataset size
  devices: auto
  accelerator: gpu
  precision: bf16
  gradient_clip_val: 1.0

loader:
  batch_size: 128  # Reduce if OOM
  num_workers: ${num_workers}
  persistent_workers: True
  prefetch_factor: 3
  pin_memory: True

optimizer:
  type: AdamW
  lr: 1e-4
  weight_decay: 1e-3

wm:
  type: lewm
  history_size: 3
  num_preds: 1
  embed_dim: 192

loss:
  sigreg:
    weight: 0.1
    kwargs:
      knots: 17
      num_proj: 1024

# Planning configuration for coordinate goals
planning:
  horizon: 5
  cem_samples: 300
  cem_iterations: 30
  cem_elites: 30
  mpc_steps: 5
  goal_type: coordinate
```

### Step 5: Launch Training

Train LeWM on your collected Habitat data:

```bash
# Using the provided config
python train.py --config-name=habitat_coord.yaml

# Or override specific parameters
python train.py --config-name=habitat_coord.yaml \
  trainer.max_epochs=100 \
  loader.batch_size=64 \
  optimizer.lr=5e-5
```

**Training on Laptop (Limited Resources):**

For laptops with limited VRAM or CPU-only:

```bash
# Reduced batch size for 4-6GB VRAM
python train.py --config-name=habitat_coord.yaml loader.batch_size=32

# CPU-only training (much slower but works without GPU)
python train.py --config-name=habitat_coord.yaml \
  trainer.accelerator=cpu \
  trainer.precision=32 \
  loader.batch_size=16 \
  num_workers=2

# Quick experiment with fewer epochs
python train.py --config-name=habitat_coord.yaml trainer.max_epochs=10
```

**Expected Training Time:**
- **GPU (RTX 3060+):** ~2-4 hours for 50 epochs with 10k episodes
- **CPU:** ~10-20 hours for 50 epochs (not recommended for full training)

### Step 6: Monitor Training

If WandB is enabled, monitor training progress at your WandB dashboard. Key metrics to watch:

- `loss/prediction` - Should decrease over time
- `loss/sigreg` - Regularization term (should remain stable)
- `total_loss` - Combined objective
- `reconstruction_error` - Quality of latent predictions

### Step 7: Evaluate the Trained Model

After training, evaluate your model on navigation tasks:

```bash
# Create evaluation config (adapt from config/eval/tworoom.yaml)
# Save as config/eval/habitat.yaml

python eval.py --config-name=habitat.yaml \
  policy=habitat_coord/lewm_habitat_coord
```

**Evaluation modes:**
1. **Coordinate-based navigation**: Navigate to specified (x, z) coordinates
2. **Image-based navigation**: Navigate to match a goal image (Phase 2 extension)

### Troubleshooting

**Issue: No trajectory files found**
```
ValueError: No trajectory files found in data/habitat_trajectories
```
**Solution:** Run the data collection script first (Step 3).

**Issue: CUDA out of memory**
```
RuntimeError: CUDA out of memory. Tried to allocate...
```
**Solution:** Reduce batch size:
```bash
python train.py --config-name=habitat_coord.yaml loader.batch_size=32
```

**Issue: habitat-sim import error**
```
ImportError: libGL.so.1: cannot open shared object file
```
**Solution:** Install OpenGL libraries:
```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

**Issue: Slow data loading**
**Solution:** Increase number of workers if you have CPU cores available:
```bash
python train.py --config-name=habitat_coord.yaml num_workers=8
```

### Customization Tips

1. **Different frame skip rates:** Adjust `frameskip` in the data config to change temporal resolution
2. **Longer prediction horizons:** Increase `wm.num_preds` for multi-step planning
3. **Larger context window:** Increase `wm.history_size` for more temporal context
4. **Alternative action spaces:** Modify `habitat_collector.py` to collect different action representations

### Expected Performance

With 10k episodes and 50 epochs of training, you should achieve:
- **Success Rate:** 60-80% on simple point-goal navigation
- **SPL (Success weighted by Path Length):** 0.5-0.7
- **Latent space quality:** Meaningful physical structure encoding (verifiable via probing)

---

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
