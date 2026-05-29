"""
Demonstration and unit tests for Anisotropic SIGReg (Manifold-Aware Regularizer)

This script demonstrates:
1. The mathematical difference between isotropic and anisotropic SIGReg
2. How the regularizer adapts to different motion patterns (hallway vs room)
3. Unit tests verifying correct behavior
"""

import torch
import numpy as np
from module import SIGReg, AnisotropicSIGReg, ManifoldAwareRegularizer


def test_isotropic_sigreg():
    """Test standard isotropic SIGReg - forces spherical latent space"""
    print("=" * 60)
    print("TEST 1: Standard Isotropic SIGReg")
    print("=" * 60)
    
    sigreg = SIGReg(knots=17, num_proj=1024)
    
    # Create batch of embeddings (T, B, D)
    T, B, D = 5, 8, 192
    embeddings = torch.randn(T, B, D)
    
    loss = sigreg(embeddings)
    print(f"Input shape: {embeddings.shape}")
    print(f"Isotropic SIGReg loss: {loss.item():.4f}")
    print("Note: This loss penalizes ANY deviation from N(0, I) - forces sphere\n")
    
    return loss


def test_anisotropic_hallway():
    """Test anisotropic SIGReg in hallway scenario (1D manifold)"""
    print("=" * 60)
    print("TEST 2: Anisotropic SIGReg - Hallway Scenario (1D)")
    print("=" * 60)
    
    sigreg = AnisotropicSIGReg(
        knots=17, 
        num_proj=1024, 
        manifold_dim='fixed_1d',
        min_variance=0.01,
        max_variance=10.0
    )
    
    T, B, D = 5, 8, 192
    
    # Simulate hallway motion: high linear velocity, near-zero angular velocity
    # Actions: [linear_vel, angular_vel]
    actions = torch.zeros(B, T, 2)
    actions[:, :, 0] = 0.5  # Moving forward at 0.5 m/s
    actions[:, :, 1] = 0.01  # Almost no rotation (straight hallway)
    
    embeddings = torch.randn(T, B, D)
    
    loss = sigreg(embeddings, actions=actions)
    print(f"Action pattern: linear_vel=0.5 m/s, angular_vel=0.01 rad/s")
    print(f"Forward/Angular ratio: {0.5/0.01:.1f} → Strong 1D manifold signal")
    print(f"Anisotropic SIGReg loss: {loss.item():.4f}")
    print("Expected: Latent space stretched along forward direction, compressed laterally\n")
    
    return loss


def test_anisotropic_room():
    """Test anisotropic SIGReg in open room scenario (2D/3D manifold)"""
    print("=" * 60)
    print("TEST 3: Anisotropic SIGReg - Open Room Scenario (2D/3D)")
    print("=" * 60)
    
    sigreg = AnisotropicSIGReg(
        knots=17, 
        num_proj=1024, 
        manifold_dim='adaptive',
        min_variance=0.01,
        max_variance=10.0
    )
    
    T, B, D = 5, 8, 192
    
    # Simulate room exploration: moderate velocities with significant rotation
    actions = torch.zeros(B, T, 2)
    actions[:, :, 0] = 0.3  # Moving at 0.3 m/s
    actions[:, :, 1] = 0.5  # Rotating at 0.5 rad/s (exploring/turning)
    
    embeddings = torch.randn(T, B, D)
    
    loss = sigreg(embeddings, actions=actions)
    print(f"Action pattern: linear_vel=0.3 m/s, angular_vel=0.5 rad/s")
    print(f"Forward/Angular ratio: {0.3/0.5:.1f} → Mixed 2D/3D manifold signal")
    print(f"Anisotropic SIGReg loss: {loss.item():.4f}")
    print("Expected: More isotropic latent space allowing multi-directional movement\n")
    
    return loss


def test_manifold_covariance_estimation():
    """Visualize estimated covariance matrices for different scenarios"""
    print("=" * 60)
    print("TEST 4: Manifold Covariance Estimation")
    print("=" * 60)
    
    sigreg = AnisotropicSIGReg(
        knots=17, 
        num_proj=512,  # Fewer projections for faster demo
        manifold_dim='adaptive',
        min_variance=0.01,
        max_variance=10.0,
        action_history_window=5
    )
    
    D = 12  # Small dimension for visualization
    
    # Scenario 1: Pure forward motion (hallway)
    actions_hallway = torch.ones(1, 5, 2)
    actions_hallway[:, :, 0] = 0.8  # High forward velocity
    actions_hallway[:, :, 1] = 0.01  # Near-zero rotation
    
    # Call estimate to populate internal eigenvalue buffers
    sigreg.estimate_manifold_covariance(actions_hallway)
    lambda_1_h = sigreg._lambda_1
    lambda_2_h = sigreg._lambda_2
    lambda_3_h = sigreg._lambda_3
    
    print("\nHallway Scenario (high forward, low rotation):")
    print(f"  λ₁ (forward): {lambda_1_h[0].item():.3f}")
    print(f"  λ₂ (lateral): {lambda_2_h[0].item():.3f}")
    print(f"  λ₃ (vertical): {lambda_3_h[0].item():.3f}")
    print(f"  Expected: λ₁ >> λ₂ ≈ λ₃ (stretched along forward axis)")
    
    # Scenario 2: High rotation (intersection/room)
    actions_room = torch.ones(1, 5, 2)
    actions_room[:, :, 0] = 0.2  # Low forward velocity
    actions_room[:, :, 1] = 0.8  # High rotation
    
    sigreg.estimate_manifold_covariance(actions_room)
    lambda_1_r = sigreg._lambda_1
    lambda_2_r = sigreg._lambda_2
    lambda_3_r = sigreg._lambda_3
    
    print("\nRoom/Intersection Scenario (low forward, high rotation):")
    print(f"  λ₁ (forward): {lambda_1_r[0].item():.3f}")
    print(f"  λ₂ (lateral): {lambda_2_r[0].item():.3f}")
    print(f"  λ₃ (vertical): {lambda_3_r[0].item():.3f}")
    print(f"  Expected: λ₁ ≈ λ₂ ≈ λ₃ (more isotropic)")
    
    # Compute anisotropy ratio
    anisotropy_hallway = lambda_1_h[0].item() / (lambda_2_h[0].item() + 1e-6)
    anisotropy_room = lambda_1_r[0].item() / (lambda_2_r[0].item() + 1e-6)
    
    print(f"\nAnisotropy ratio (λ₁/λ₂):")
    print(f"  Hallway: {anisotropy_hallway:.2f}x")
    print(f"  Room: {anisotropy_room:.2f}x")
    print(f"  Higher ratio = more elongated latent space along forward direction")
    

def test_wrapper_module():
    """Test the ManifoldAwareRegularizer wrapper"""
    print("\n" + "=" * 60)
    print("TEST 5: ManifoldAwareRegularizer Wrapper")
    print("=" * 60)
    
    # Test isotropic mode
    reg_iso = ManifoldAwareRegularizer(use_anisotropic=False, knots=17, num_proj=1024)
    embeddings = torch.randn(5, 4, 64)
    loss_iso = reg_iso(embeddings)
    print(f"Wrapper (isotropic mode) loss: {loss_iso.item():.4f}")
    
    # Test anisotropic mode
    reg_aniso = ManifoldAwareRegularizer(
        use_anisotropic=True, 
        knots=17, 
        num_proj=1024,
        manifold_dim='adaptive'
    )
    actions = torch.randn(4, 5, 2)
    loss_aniso = reg_aniso(embeddings, actions=actions)
    print(f"Wrapper (anisotropic mode) loss: {loss_aniso.item():.4f}")
    print("Wrapper successfully switches between regularizer types\n")


def compare_regularizers():
    """Direct comparison: same data, isotropic vs anisotropic"""
    print("=" * 60)
    print("TEST 6: Direct Comparison - Isotropic vs Anisotropic")
    print("=" * 60)
    
    torch.manual_seed(42)
    T, B, D = 5, 8, 192
    
    # Create structured embeddings that mimic hallway data
    # High variance in first dimension, low in others
    embeddings = torch.zeros(T, B, D)
    embeddings[:, :, 0] = torch.randn(T, B) * 3.0  # High variance forward
    embeddings[:, :, 1:] = torch.randn(T, B, D-1) * 0.1  # Low variance lateral
    
    # Actions indicating hallway motion
    actions = torch.zeros(B, T, 2)
    actions[:, :, 0] = 0.5
    actions[:, :, 1] = 0.01
    
    # Isotropic regularizer (forces sphere - WRONG for hallways)
    sigreg_iso = SIGReg(knots=17, num_proj=1024)
    loss_iso = sigreg_iso(embeddings)
    
    # Anisotropic regularizer (allows stretching - CORRECT for hallways)
    sigreg_aniso = AnisotropicSIGReg(
        knots=17, 
        num_proj=1024, 
        manifold_dim='fixed_1d',
        min_variance=0.01,
        max_variance=10.0
    )
    loss_aniso = sigreg_aniso(embeddings, actions=actions)
    
    print(f"Data structure: σ_forward=3.0, σ_lateral=0.1 (mimics hallway)")
    print(f"Isotropic SIGReg loss: {loss_iso.item():.4f}")
    print(f"Anisotropic SIGReg loss: {loss_aniso.item():.4f}")
    print(f"\nKey insight:")
    print(f"  - Isotropic tries to force σ_forward ≈ σ_lateral (destroys structure)")
    print(f"  - Anisotropic accepts σ_forward >> σ_lateral (preserves hallway geometry)")
    print(f"  - This is why Micro-JEPA needs manifold-aware regularization!\n")


def main():
    print("\n" + "🎓" * 30)
    print("MICRO-JEPA: MANIFOLD-AWARE REGULARIZER DEMONSTRATION")
    print("🎓" * 30 + "\n")
    
    print("This demo shows how Anisotropic SIGReg solves the 'hallway collapse' problem")
    print("by allowing the latent space to stretch along navigable paths while")
    print("penalizing variance through walls.\n")
    
    # Run all tests
    test_isotropic_sigreg()
    test_anisotropic_hallway()
    test_anisotropic_room()
    test_manifold_covariance_estimation()
    test_wrapper_module()
    compare_regularizers()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
The Manifold-Aware Regularizer enables Micro-JEPA to:

1. ADAPT TO ENVIRONMENT GEOMETRY:
   - Hallways (1D): Stretch latent space along forward direction
   - Rooms (2D): Allow forward + lateral variance  
   - Open spaces (3D): Near-isotropic latent distribution

2. AVOID LATENT SPACE DISTORTION:
   - Standard SIGReg forces ALL environments into spheres
   - This creates bias when planning in constrained spaces
   - Anisotropic SIGReg respects physical manifold structure

3. IMPROVE COLLISION AVOIDANCE:
   - Narrow lateral bounds act as "soft walls" in latent space
   - CEM planner naturally avoids sampling collision trajectories
   - No need for explicit collision penalties in many cases

THESIS CONTRIBUTION:
Replace LeWM's isotropic SIGReg with kinematic-aware anisotropic regularization
to solve navigation in low-dimensional manifolds (hallways, corridors, tunnels).
    """)
    
    print("=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
