import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time


class AnisotropicSIGReg(torch.nn.Module):
    """
    Manifold-Aware Anisotropic SIGReg for Micro-JEPA.
    
    This regularizer replaces the standard isotropic Gaussian prior N(0, I) with
    a dynamic anisotropic Gaussian N(0, Σ_t) where Σ_t is estimated from the
    robot's recent kinematic history.
    
    Key insight: Hallways are 1D manifolds (high variance forward, zero through walls),
    rooms are 2D manifolds, open spaces are 3D. Standard SIGReg forces all into
    isotropic spheres, causing latent space distortion in constrained environments.
    
    Reference: LeJEPA authors note that "whenever the covariance matrix of the data
    is anisotropic, there will be downstream bias" if forced into isotropic prior.
    """

    def __init__(self, knots=17, num_proj=1024, manifold_dim='adaptive', 
                 min_variance=0.01, max_variance=10.0, action_history_window=5):
        """
        Args:
            knots: Number of quadrature points for Epps-Pulley test
            num_proj: Number of random projections for SIGReg
            manifold_dim: 'adaptive' (estimate from actions), 'fixed_1d', 'fixed_2d', or 'fixed_3d'
            min_variance: Minimum allowed variance in any direction (prevents collapse)
            max_variance: Maximum allowed variance (prevents explosion)
            action_history_window: Number of past action frames to use for manifold estimation
        """
        super().__init__()
        self.num_proj = num_proj
        self.manifold_dim = manifold_dim
        self.min_variance = min_variance
        self.max_variance = max_variance
        self.action_history_window = action_history_window
        
        # Quadrature setup for Epps-Pulley test
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        
        # Smoothing buffer for covariance estimation stability
        self.register_buffer("covariance_smooth", None)
        self.smoothing_alpha = 0.8  # EMA smoothing factor

    def estimate_manifold_covariance(self, actions, embeddings=None):
        """
        Estimate local manifold covariance matrix Σ_t from recent actions.
        
        For differential drive robots:
        - High linear velocity + low angular velocity → 1D manifold (hallway)
        - High angular velocity → 2D manifold (turning/intersection)
        - Mixed velocities → 3D manifold (open room exploration)
        
        Args:
            actions: (B, T, action_dim) tensor with [linear_vel, angular_vel] or similar
            embeddings: Optional, can use latent structure to refine estimate
            
        Returns:
            Sigma_t: (B, D, D) covariance matrices for each batch item
        """
        B, T, action_dim = actions.shape
        
        # Compute action statistics over recent history
        # Use last `action_history_window` timesteps
        window_size = min(self.action_history_window, T)
        recent_actions = actions[:, -window_size:]  # (B, window_size, action_dim)
        
        # Extract linear and angular velocities (assumes first 2 dims are v, ω)
        if action_dim >= 2:
            linear_vel = recent_actions[..., 0].abs().mean(dim=-1, keepdim=True)  # (B, 1)
            angular_vel = recent_actions[..., 1].abs().mean(dim=-1, keepdim=True)  # (B, 1)
        else:
            # Fallback: use magnitude of available actions
            linear_vel = recent_actions.abs().mean(dim=(1, 2), keepdim=True)
            angular_vel = linear_vel * 0.5  # Assume some rotation
        
        # Determine manifold type based on kinematic ratios
        # Threshold tuned for typical robot velocities (m/s, rad/s)
        vel_threshold = 0.1  # m/s
        rot_threshold = 0.2  # rad/s
        
        # Compute anisotropy ratio: how much more we move forward vs rotate
        forward_dominance = linear_vel / (angular_vel + 1e-6)  # (B, 1)
        
        # Build target covariance eigenvalues based on motion pattern
        # Lambda_1 (forward/tangent direction), Lambda_2, Lambda_3 (lateral/normal)
        if self.manifold_dim == 'fixed_1d':
            # Strict hallway: only forward motion allowed
            lambda_1 = torch.ones_like(forward_dominance) * self.max_variance
            lambda_2 = torch.ones_like(forward_dominance) * self.min_variance
            lambda_3 = torch.ones_like(forward_dominance) * self.min_variance
        elif self.manifold_dim == 'fixed_2d':
            # Room/plane: forward + lateral motion
            lambda_1 = torch.ones_like(forward_dominance) * self.max_variance
            lambda_2 = torch.ones_like(forward_dominance) * self.max_variance * 0.5
            lambda_3 = torch.ones_like(forward_dominance) * self.min_variance
        elif self.manifold_dim == 'fixed_3d':
            # Open space: isotropic
            lambda_1 = torch.ones_like(forward_dominance) * 1.0
            lambda_2 = torch.ones_like(forward_dominance) * 1.0
            lambda_3 = torch.ones_like(forward_dominance) * 1.0
        else:  # 'adaptive'
            # Adaptive: blend between 1D, 2D, 3D based on actual motion
            # Smooth transition using sigmoid-like blending
            blend_1d_2d = torch.sigmoid((forward_dominance - 2.0) * 2.0)  # High ratio → 1D
            blend_2d_3d = torch.sigmoid((angular_vel - rot_threshold) * 10.0)  # High rotation → 3D
            
            # Base variances (B, 3)
            var_1d = torch.stack([
                torch.ones_like(forward_dominance) * self.max_variance,
                torch.ones_like(forward_dominance) * self.min_variance,
                torch.ones_like(forward_dominance) * self.min_variance,
            ], dim=-1)  # (B, 3)
            
            var_2d = torch.stack([
                torch.ones_like(forward_dominance) * self.max_variance,
                torch.ones_like(forward_dominance) * self.max_variance * 0.5,
                torch.ones_like(forward_dominance) * self.min_variance,
            ], dim=-1)  # (B, 3)
            
            var_3d = torch.stack([
                torch.ones_like(forward_dominance) * 1.0,
                torch.ones_like(forward_dominance) * 1.0,
                torch.ones_like(forward_dominance) * 1.0,
            ], dim=-1)  # (B, 3)
            
            # Blend: start with 1D, blend to 2D, then to 3D
            eigenvalues = var_1d * (1 - blend_1d_2d).unsqueeze(-1) + var_2d * blend_1d_2d.unsqueeze(-1)
            eigenvalues = eigenvalues * (1 - blend_2d_3d).unsqueeze(-1) + var_3d * blend_2d_3d.unsqueeze(-1)
            
            lambda_1 = eigenvalues[:, 0, 0] if eigenvalues.dim() > 2 else eigenvalues[:, 0]  # (B,)
            lambda_2 = eigenvalues[:, 0, 1] if eigenvalues.dim() > 2 else eigenvalues[:, 1]  # (B,)
            lambda_3 = eigenvalues[:, 0, 2] if eigenvalues.dim() > 2 else eigenvalues[:, 2]  # (B,)
        
        # Clamp to valid range
        lambda_1 = torch.clamp(lambda_1, self.min_variance, self.max_variance)
        lambda_2 = torch.clamp(lambda_2, self.min_variance, self.max_variance)
        lambda_3 = torch.clamp(lambda_3, self.min_variance, self.max_variance)
        
        # Store eigenvalues for use in forward pass - ensure shape is (B,)
        # Select first column since all columns are identical after clamping
        self._lambda_1 = lambda_1[:, 0] if lambda_1.dim() > 1 else lambda_1
        self._lambda_2 = lambda_2[:, 0] if lambda_2.dim() > 1 else lambda_2
        self._lambda_3 = lambda_3[:, 0] if lambda_3.dim() > 1 else lambda_3
        
        # Return a placeholder - actual Sigma_t constructed in forward() 
        # when we know D
        return None

    def modified_epps_pulley_test(self, h, target_var):
        """
        Modified Epps-Pulley test against N(0, target_var) instead of N(0, 1).
        
        Standard EP test checks if projection matches N(0, 1).
        We scale the characteristic function evaluation by target variance.
        
        Args:
            h: (T, B) projected values along random direction
            target_var: (B,) target variance for each batch item
            
        Returns:
            EP statistic measuring deviation from N(0, target_var)
        """
        # Reshape target_var for broadcasting: (B,) -> (1, B, 1)
        target_var = target_var.unsqueeze(0).unsqueeze(-1)  # (1, B, 1)
        
        # Scale the test statistic by target variance
        # For N(0, σ²), the characteristic function is exp(-σ²t²/2)
        # We adjust the quadrature accordingly
        scaled_t = self.t * torch.sqrt(target_var + 1e-8)  # (1, B, knots)
        scaled_phi = torch.exp(-scaled_t.square() / 2.0)  # (1, B, knots)
        scaled_weights = self.weights * torch.exp(-scaled_t.square() / 2.0)  # (1, B, knots)
        
        # Compute EP statistic with scaled parameters
        # h: (T, B) -> (T, B, 1) for broadcasting with (1, B, knots)
        x_t = h.unsqueeze(-1) * scaled_t  # (T, B, knots)
        
        # Mean over time dimension (dim=-3 = T)
        cos_mean = x_t.cos().mean(dim=-3, keepdim=True)  # (1, B, knots)
        sin_mean = x_t.sin().mean(dim=-3, keepdim=True)  # (1, B, knots)
        
        err = (cos_mean - scaled_phi).square() + sin_mean.square()  # (1, B, knots)
        statistic = (err * scaled_weights).sum(dim=-1) * h.size(-2)  # (1, B)
        
        return statistic.squeeze(0)  # (B,)

    def forward(self, proj, actions=None, embeddings=None):
        """
        Compute anisotropic SIGReg loss.
        
        Args:
            proj: (T, B, D) latent projections
            actions: (B, T, action_dim) recent actions for manifold estimation
            embeddings: Optional full embeddings for refined manifold estimation
            
        Returns:
            Anisotropic regularization loss (scalar)
        """
        T, B, D = proj.shape
        
        # Step 1: Estimate target covariance from actions to get eigenvalues
        if actions is not None:
            self.estimate_manifold_covariance(actions, embeddings)
            
            # Build diagonal covariance matrix from stored eigenvalues
            # Eigenvalues are shape (B, 1), need to distribute across D dimensions
            # Distribute eigenvalues across latent dimensions
            lambda_1_flat = self._lambda_1.squeeze(-1)  # (B,)
            lambda_2_flat = self._lambda_2.squeeze(-1)  # (B,)
            lambda_3_flat = self._lambda_3.squeeze(-1)  # (B,)
            
            if D <= 3:
                parts = [lambda_1_flat]  # (B,)
                if D > 1:
                    parts.append(lambda_2_flat)
                if D > 2:
                    parts.append(lambda_3_flat)
                diag_values = torch.stack(parts, dim=-1)  # (B, D)
            else:
                # Distribute: first third gets lambda_1, second third gets lambda_2, rest gets lambda_3
                third = D // 3
                diag_values = torch.cat([
                    lambda_1_flat.unsqueeze(-1).expand(-1, third),
                    lambda_2_flat.unsqueeze(-1).expand(-1, third),
                    lambda_3_flat.unsqueeze(-1).expand(-1, D - 2*third),
                ], dim=-1)  # (B, D)
            
            # Apply EMA smoothing for temporal consistency
            if self.covariance_smooth is None:
                self.covariance_smooth = diag_values
            else:
                self.covariance_smooth = self.smoothing_alpha * self.covariance_smooth + \
                                         (1 - self.smoothing_alpha) * diag_values
            
            # Construct Sigma_t as diagonal matrix (B, D, D)
            Sigma_t = torch.diag_embed(self.covariance_smooth)
        else:
            # Fallback to isotropic if no actions provided
            Sigma_t = torch.eye(D, device=proj.device).unsqueeze(0).expand(B, -1, -1)
        
        # Step 2: Sample random projections
        A = torch.randn(D, self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))  # (D, num_proj)
        
        # Step 3: Project embeddings
        h = proj @ A  # (T, B, num_proj)
        
        # Step 4: Compute target variance for each projection direction
        # var_m = u_m^T Σ_t u_m for each projection u_m
        # (B, D, D) @ (D, num_proj) -> (B, D, num_proj)
        Sigma_A = Sigma_t @ A  # (B, D, num_proj)
        # u^T Σ u for each column of A
        target_vars = (A.unsqueeze(0) * Sigma_A).sum(dim=1)  # (B, num_proj)
        
        # Step 5: Compute anisotropic EP test for each projection
        losses = []
        for m in range(self.num_proj):
            h_m = h[:, :, m]  # (T, B)
            target_var_m = target_vars[:, m]  # (B,)
            loss_m = self.modified_epps_pulley_test(h_m, target_var_m)
            losses.append(loss_m)
        
        # Average over all projections
        total_loss = torch.stack(losses).mean()
        
        return total_loss


class ManifoldAwareRegularizer(nn.Module):
    """
    Wrapper that chooses between isotropic and anisotropic SIGReg based on config.
    
    Usage in training:
        regularizer = ManifoldAwareRegularizer(use_anisotropic=True, ...)
        loss = regularizer(embeddings, actions=recent_actions)
    """
    def __init__(self, use_anisotropic=False, **kwargs):
        super().__init__()
        self.use_anisotropic = use_anisotropic
        
        if use_anisotropic:
            self.regularizer = AnisotropicSIGReg(**kwargs)
        else:
            self.regularizer = SIGReg(**{k: v for k, v in kwargs.items() 
                                          if k in ['knots', 'num_proj']})
    
    def forward(self, proj, actions=None, embeddings=None):
        if self.use_anisotropic:
            return self.regularizer(proj, actions, embeddings)
        else:
            return self.regularizer(proj)
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
