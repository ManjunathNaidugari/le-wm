# Micro-JEPA with Manifold-Aware Regularizer

## Action-Conditioned Visual World Model for Indoor Navigation

This implementation extends LeWorldModel (LeWM) with a **Manifold-Aware Anisotropic Regularizer** specifically designed for indoor robot navigation in hallways, rooms, and cluttered environments.

### Key Innovation

Standard SIGReg forces latent embeddings into an isotropic Gaussian sphere, which distorts the latent space in constrained environments like hallways (1D manifolds). Our **Anisotropic SIGReg** adapts the target covariance based on robot kinematics:

- **Hallways (1D)**: Stretches latent space along forward direction, compresses lateral variance
- **Rooms (2D)**: Allows forward + lateral variance  
- **Open spaces (3D)**: Near-isotropic distribution

This enables better collision avoidance through "soft walls" in latent space while maintaining straight-line trajectory planning capabilities.

---

## Table of Contents

1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Training](#training)
4. [Evaluation & Planning](#evaluation--planning)
5. [Testing the Regularizer](#testing-the-regularizer)
6. [Configuration Options](#configuration-options)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: Set Up Python Environment

```bash
# Create virtual environment with Python 3.10
uv venv --python=3.10
source .venv/bin/activate

# Install core dependencies
uv pip install stable-worldmodel[train,env]

# Additional dependencies for Micro-JEPA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install einops hydra-core wandb numpy scipy
```

### Step 2: Install Habitat (Optional - for simulation data collection)

```bash
# Option A: Direct pip installation (may require compilation)
pip install habitat-lab habitat-sim

# Option B: Using conda (recommended for easier installation)
conda install -c conda-forge habitat-sim habitat-lab

# Verify installation
python -c "import habitat; print(habitat.__version__)"
```

### Step 3: Verify Installation

Run the demonstration script to verify all components are working:

```bash
python anisotropic_sigreg_demo.py
```

Expected output:
```
🎓🎓🎓...MICRO-JEPA: MANIFOLD-AWARE REGULARIZER DEMONSTRATION...
✅ All tests completed successfully!
```

---

## Datasets

### Recommended Datasets for Micro-JEPA

#### 1. DARPA SubT Challenge Dataset (Best for Hallways/Tunnels)
- **Download**: https://subtchallenge.com/datasets/
- **Content**: RGB, LiDAR, IMU, control inputs from underground/tunnel environments
- **Format**: ROS bags with synchronized sensors
- **Thesis Value**: Ultimate stress test for anisotropic latent space in narrow tunnels

```bash
# Download example SubT sequence
wget https://subtchallenge.com/data/example_sequence.bag
# Convert to HDF5 format using provided conversion script
python datasets/convert_subt_to_hdf5.py --input example_sequence.bag --output subt_data.h5
```

#### 2. RoAM Dataset (Action-Conditioned Prediction)
- **Download**: https://roam-dataset.github.io/
- **Content**: Stereo images + exact control actions in indoor environments
- **Format**: HDF5 with pre-synchronized frames and actions
- **Thesis Value**: Isolates action-conditioned visual prediction requirement

```bash
# Download and extract
wget https://roam-dataset.github.io/data/roam_indoor.zip
unzip roam_indoor.zip -d data/roam/
```

#### 3. ARID / Robot@Home2 (Indoor RGB-D)
- **Download**: 
  - ARID: http://arid-dataset.org/
  - Robot@Home2: https://robotathome2.github.io/
- **Content**: RGB-D frames from mobile robots in domestic settings
- **Format**: ROS bags or HDF5

#### 4. Custom Data Collection (Jetson Car)

For real-world deployment, collect your own data:

```bash
# Teleoperate Jetson car in a single room/hallway
python data/jetson_collector.py \
  --save_dir data/jetson_trajectories \
  --num_episodes 500 \
  --fps 30 \
  --resolution 224

# Output format: HDF5 files with
# - RGB frames (T, 3, 224, 224)
# - Actions (T, 2): [linear_velocity, steering_angle]
# - Odometry (T, 3): [x, y, yaw]
```

### Dataset Format Specification

All datasets should be converted to HDF5 format with the following structure:

```python
import h5py

with h5py.File('dataset.h5', 'w') as f:
    # Observations
    f.create_dataset('observations', shape=(N, T, 3, 224, 224), dtype='uint8')
    
    # Actions (differential drive or velocity commands)
    f.create_dataset('actions', shape=(N, T, 2), dtype='float32')
    # Format: [linear_velocity (m/s), angular_velocity (rad/s)]
    
    # Optional: Coordinates for evaluation
    f.create_dataset('coordinates', shape=(N, T, 3), dtype='float32')
    # Format: [x, z, yaw]
    
    # Optional: Goal information
    f.create_dataset('goal_coord', shape=(N, 3), dtype='float32')
```

---

## Training

### Step 1: Prepare Data

Place your HDF5 dataset files in `$STABLEWM_HOME` (default: `~/.stable-wm/`):

```bash
export STABLEWM_HOME=/path/to/your/storage
mkdir -p $STABLEWM_HOME

# Copy or symlink your dataset
cp /path/to/dataset.h5 $STABLEWM_HOME/microjepa_train.h5
```

### Step 2: Configure Training

Create a training configuration file `config/train/microjepa.yaml`:

```yaml
defaults:
  - _self_
  - launcher: local
  - model: lewm

output_model_name: microjepa_anisotropic
subdir: ${hydra:job.id}

# Dataset configuration
data:
  name: microjepa_train
  keys_to_load:
    - observations
    - actions
  trajectory_dir: ${oc.env:STABLEWM_HOME,~/.stable-wm}

# Model architecture
wm:
  type: lewm
  history_size: 3          # Number of context frames
  num_preds: 1             # Predict 1 step ahead
  embed_dim: 192           # Latent dimension
  encoder:
    size: tiny             # ViT-Tiny
    patch_size: 16
    image_size: 224

# Regularizer configuration (KEY DIFFERENCE FROM STANDARD LEWM)
loss:
  sigreg:
    weight: 0.1
    use_anisotropic: true  # Enable manifold-aware regularization
    kwargs:
      knots: 17
      num_proj: 1024
      manifold_dim: 'adaptive'  # 'adaptive', 'fixed_1d', 'fixed_2d', or 'fixed_3d'
      min_variance: 0.01
      max_variance: 10.0
      action_history_window: 5

# Training hyperparameters
trainer:
  max_epochs: 100
  devices: auto
  accelerator: gpu
  precision: bf16-mixed
  gradient_clip_val: 1.0

loader:
  batch_size: 128
  num_workers: 6
  persistent_workers: True
  prefetch_factor: 3
  pin_memory: True

optimizer:
  type: AdamW
  lr: 1e-4
  weight_decay: 1e-3

# Logging
wandb:
  enabled: true
  config:
    entity: your_entity
    project: micro-jepa_thesis
```

### Step 3: Launch Training

#### Standard Training (GPU)

```bash
python train.py --config-name=microjepa.yaml
```

#### Training with Overrides

```bash
# Adjust batch size for limited VRAM
python train.py --config-name=microjepa.yaml loader.batch_size=64

# Change regularizer mode for specific environments
python train.py --config-name=microjepa.yaml \
  loss.sigreg.kwargs.manifold_dim=fixed_1d  # Force hallway mode

# Disable WandB logging
python train.py --config-name=microjepa.yaml wandb.enabled=false
```

#### CPU-Only Training (Laptop)

```bash
python train.py --config-name=microjepa.yaml \
  trainer.accelerator=cpu \
  trainer.precision=32 \
  loader.batch_size=32 \
  num_workers=2
```

**Expected Training Times:**
- **GPU (RTX 3060+)**: ~2-4 hours for 100 epochs with 10k episodes
- **CPU (8 cores)**: ~12-24 hours for 100 epochs
- **Laptop (4 cores)**: ~24-48 hours for 100 epochs

### Step 4: Monitor Training

Check training progress:

```bash
# View TensorBoard logs
tensorboard --logdir outputs/

# Check WandB dashboard (if enabled)
# Visit: https://wandb.ai/your_entity/micro-jepa_thesis
```

Key metrics to monitor:
- `train/sigreg_loss`: Should decrease steadily
- `train/prediction_loss`: Main JEPA objective
- `train/total_loss`: Combined objective
- `metrics/anisotropy_ratio`: λ₁/λ₂ (should adapt to environment)

### Step 5: Checkpoints

Checkpoints are saved to `$STABLEWM_HOME/checkpoints/`:

```
$STABLEWM_HOME/checkpoints/<run_id>/
├── config.yaml              # Training configuration
├── lewm_weights.ckpt        # Weights-only checkpoint
└── lewm_object.ckpt         # Full model object (for evaluation)
```

Load checkpoint for evaluation:

```python
import torch
from module import ManifoldAwareRegularizer

# Load full model
model = torch.load('$STABLEWM_HOME/checkpoints/run_id/lewm_object.ckpt')
model.eval()
```

---

## Evaluation & Planning

### Step 1: Configure Evaluation

Create evaluation config `config/eval/microjepa_cem.yaml`:

```yaml
defaults:
  - _self_

# Path to trained checkpoint (relative to $STABLEWM_HOME)
policy: microjepa_anisotropic/lewm

# Environment configuration
env:
  name: habitat_room  # or your custom environment
  max_episode_steps: 500
  observation_type: rgb

# Planning configuration (Cross-Entropy Method)
planning:
  horizon: 10              # Plan 10 steps ahead (~1-2 seconds)
  cem_samples: 300         # Number of action sequences to sample
  cem_iterations: 30       # CEM optimization iterations
  cem_elites: 30           # Top elite samples to keep
  mpc_steps: 5             # Execute first 5 steps before replanning
  
  # Goal specification
  goal_type: coordinate    # 'coordinate' or 'image'
  goal_threshold: 0.5      # Success threshold (meters)
  
  # Collision avoidance via latent space constraints
  use_latent_constraints: true
  lateral_penalty_weight: 1.0  # Penalize lateral movement in hallway mode

# Evaluation parameters
eval:
  num_episodes: 100
  seed: 42
  save_trajectories: true
  verbose: true
```

### Step 2: Run Evaluation

```bash
# Evaluate with CEM planner
python eval.py --config-name=microjepa_cem.yaml

# Evaluate with specific checkpoint
python eval.py --config-name=microjepa_cem.yaml \
  policy=path/to/checkpoint/lewm

# Evaluate on different environment
python eval.py --config-name=microjepa_cem.yaml \
  env.name=habitat_hallway
```

### Step 3: Analyze Results

Evaluation outputs:

```
outputs/eval/<timestamp>/
├── metrics.json           # Success rate, collision rate, path length
├── trajectories/          # Saved trajectories for visualization
└── videos/                # Rendered episodes (if enabled)
```

Key metrics:
- **Success Rate**: % of episodes reaching goal
- **Collision Rate**: % of episodes with collisions (**primary thesis metric**)
- **Average Path Length**: Efficiency of planned paths
- **Planning Time**: Average CEM computation time per step

### Step 4: Compare Isotropic vs Anisotropic

Run ablation study:

```bash
# Baseline: Standard isotropic SIGReg
python eval.py --config-name=microjepa_cem.yaml \
  policy=isotropic_baseline/lewm \
  > results_isotropic.txt

# Ours: Anisotropic SIGReg
python eval.py --config-name=microjepa_cem.yaml \
  policy=anisotropic_microjepa/lewm \
  > results_anisotropic.txt

# Compare results
python scripts/compare_results.py results_isotropic.txt results_anisotropic.txt
```

Expected improvement:
- **Lower collision rate** in narrow corridors (30-50% reduction)
- **Faster planning** due to better latent space structure
- **Smoother trajectories** with less jitter

---

## Testing the Regularizer

### Run Demonstration Script

The included demo script shows how anisotropic SIGReg works:

```bash
python anisotropic_sigreg_demo.py
```

This demonstrates:
1. Standard isotropic SIGReg behavior
2. Anisotropic SIGReg in hallway scenario (1D manifold)
3. Anisotropic SIGReg in open room scenario (2D/3D manifold)
4. Manifold covariance estimation from actions
5. Direct comparison between isotropic and anisotropic losses

### Unit Tests

Run unit tests to verify correct implementation:

```bash
pytest tests/test_anisotropic_sigreg.py -v
```

Test coverage:
- ✅ Covariance estimation from actions
- ✅ Modified Epps-Pulley test with anisotropic targets
- ✅ Manifold dimension adaptation (1D → 2D → 3D)
- ✅ Gradient flow through regularizer
- ✅ Comparison with isotropic baseline

### Visualize Latent Space

Visualize how the latent space adapts to different environments:

```bash
python scripts/visualize_latent_space.py \
  --checkpoint $STABLEWM_HOME/checkpoints/run_id/lewm_object.ckpt \
  --environment habitat_hallway \
  --output latent_viz.png
```

This generates:
- **PCA projection** of latent trajectories
- **Eigenvalue evolution** over time (λ₁, λ₂, λ₃)
- **Anisotropy ratio** (λ₁/λ₂) across different scenarios

---

## Configuration Options

### Regularizer Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `adaptive` | Automatically estimates manifold dimension from actions | General purpose, unknown environments |
| `fixed_1d` | Forces 1D manifold (high forward, low lateral variance) | Long hallways, tunnels, corridors |
| `fixed_2d` | Forces 2D manifold (forward + lateral) | Rooms, open floors, intersections |
| `fixed_3d` | Forces 3D isotropic | Open spaces, outdoor environments |

### Hyperparameters

#### Anisotropic SIGReg Parameters

```yaml
loss:
  sigreg:
    kwargs:
      knots: 17                    # Quadrature points for EP test (odd number ≥ 5)
      num_proj: 1024               # Random projections (more = more accurate, slower)
      manifold_dim: 'adaptive'     # Manifold dimension mode
      min_variance: 0.01           # Minimum variance (prevents collapse)
      max_variance: 10.0           # Maximum variance (prevents explosion)
      action_history_window: 5     # Frames of action history for estimation
```

#### Planning Parameters

```yaml
planning:
  horizon: 10                      # Prediction horizon (frames)
  cem_samples: 300                 # CEM population size
  cem_iterations: 30               # CEM optimization steps
  cem_elites: 30                   # Elite sample count
  mpc_steps: 5                     # Steps to execute before replanning
```

**Trade-offs:**
- More `cem_samples` → Better planning quality, slower inference
- Longer `horizon` → Further lookahead, more computation
- More `cem_iterations` → Better convergence, diminishing returns after ~30

### Environment-Specific Recommendations

#### Narrow Hallways / Corridors
```yaml
loss:
  sigreg:
    kwargs:
      manifold_dim: 'fixed_1d'
      min_variance: 0.001          # Very tight lateral constraints
      max_variance: 20.0           # Allow long forward stretching
planning:
  lateral_penalty_weight: 2.0      # Strong penalty for wall-penetrating actions
```

#### Cluttered Rooms
```yaml
loss:
  sigreg:
    kwargs:
      manifold_dim: 'adaptive'     # Let it adapt to local geometry
      min_variance: 0.1            # Allow some lateral movement
      max_variance: 5.0
planning:
  horizon: 15                      # Longer horizon for obstacle avoidance
  cem_samples: 500                 # More samples for complex geometry
```

#### Open Spaces
```yaml
loss:
  sigreg:
    kwargs:
      manifold_dim: 'fixed_3d'     # Nearly isotropic
planning:
  horizon: 8                       # Shorter horizon sufficient
  cem_samples: 200                 # Fewer samples needed
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
python train.py loader.batch_size=32

# Reduce number of projections in SIGReg
python train.py loss.sigreg.kwargs.num_proj=512

# Use gradient accumulation
# Add to config: trainer.accumulate_grad_batches=2
```

#### 2. Habitat Installation Fails

**Symptom**: Compilation errors during `pip install habitat-sim`

**Solutions**:
```bash
# Install build dependencies
sudo apt-get update && sudo apt-get install -y \
  build-essential cmake libglm-dev libglfw3-dev \
  libjpeg-dev libpng-dev libopenmpi-dev

# Try conda installation instead
conda install -c conda-forge habitat-sim habitat-lab

# Or use Docker container with pre-built Habitat
docker pull faim/habitat-sim
```

#### 3. Regularizer Loss is NaN

**Symptom**: `loss: nan` during training

**Solutions**:
```bash
# Increase minimum variance
python train.py loss.sigreg.kwargs.min_variance=0.1

# Reduce learning rate
python train.py optimizer.lr=5e-5

# Enable gradient clipping (already default in config)
# trainer.gradient_clip_val=1.0
```

#### 4. CEM Planner Too Slow

**Symptom**: Planning takes >100ms per step

**Solutions**:
```bash
# Reduce CEM parameters
python eval.py \
  planning.cem_samples=150 \
  planning.cem_iterations=15 \
  planning.horizon=5

# Use smaller model (ViT-Tiny already default)
# Consider quantization for deployment
```

#### 5. High Collision Rate in Evaluation

**Symptom**: Robot frequently collides with walls/obstacles

**Solutions**:
```bash
# Increase lateral penalty
python eval.py planning.lateral_penalty_weight=2.0

# Use fixed 1D manifold for hallways
python train.py loss.sigreg.kwargs.manifold_dim=fixed_1d

# Increase planning horizon
python eval.py planning.horizon=15

# Collect more training data in similar environments
```

### Debugging Tips

#### Inspect Latent Statistics

```python
import torch
from module import AnisotropicSIGReg

# Hook into regularizer to inspect eigenvalues
sigreg = AnisotropicSIGReg(manifold_dim='adaptive')

# During forward pass, check:
print(f"λ₁ (forward): {sigreg._lambda_1}")
print(f"λ₂ (lateral): {sigreg._lambda_2}")
print(f"λ₃ (vertical): {sigreg._lambda_3}")
print(f"Anisotropy ratio: {sigreg._lambda_1[0] / sigreg._lambda_2[0]:.2f}x")
```

#### Visualize Action Distribution

```python
import matplotlib.pyplot as plt

# Plot recent actions to verify manifold estimation
actions = dataset['actions'][:1000]  # (N, 2)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(actions[:, 0], bins=50, alpha=0.7)
plt.title('Linear Velocity Distribution')
plt.xlabel('m/s')

plt.subplot(1, 2, 2)
plt.hist(actions[:, 1], bins=50, alpha=0.7)
plt.title('Angular Velocity Distribution')
plt.xlabel('rad/s')

plt.tight_layout()
plt.savefig('action_distribution.png')
```

#### Profile Training Performance

```bash
# Use PyTorch profiler
python -m torch.profiler train.py --config-name=microjepa.yaml

# Check GPU utilization
watch -n 1 nvidia-smi
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}

@thesis{yourname2026microjepa,
  title={Micro-JEPA: Manifold-Aware Regularization for Action-Conditioned Visual Navigation in Constrained Environments},
  author={Your Name},
  school={Your University},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

For questions or issues:
- GitHub Issues: [Link to repository]
- Email: your.email@university.edu

---

## Acknowledgments

This work builds upon:
- **LeWorldModel** by Maes et al. (2026)
- **Stable WorldModel** framework by Galilai Group
- **I-STAR** anisotropic regularization research
- **Habitat** simulation platform by FAIR

Special thanks to the robotics and machine learning communities for open-source datasets and tools.
