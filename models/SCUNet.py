"""
DAU-Net: Scale-Adaptive U-Net
for Time Series Forecasting

Architecture:
  Input [B, seq_len, N]
    → RevIN (norm)
    → predict_linear (粗糙预测):  seq_len → seq_len + pred_len
    → StemConv:        1 → d_model features, stride=2 time compression
    → Encoder (L×):    MSCD → SC^2R → Fuse → DownSample(×2)
    → Bottleneck
    → Decoder (L×):    Upsample(×2) → CGF
    → AdaptiveSpectralGate:  learnable frequency-domain gating
    → Head:            Flatten → Linear → pred_len
    → RevIN (denorm)
  Output [B, pred_len, N]

Key hyperparameters (from configs):
  --mscd_K         : K_comp, number of component types (default 3)
  --sc2r_M_avg     : avg edges per variable for SC^2R routing (default 2)
  --sc2r_K_edges   : absolute edge count override (default None, uses M_avg*N)
  --gate_reduction : reduction ratio for spectral gate MLP (default 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from layers.Embed import PositionalEmbedding
from layers.RevIN import RevIN


# ==============================================================================
# AdaptiveSpectralGate — learnable frequency-domain gating
#   Learns per-frequency-bin gating weights from a small MLP.
# ==============================================================================

class AdaptiveSpectralGate(nn.Module):
    """
    Learnable per-frequency-bin gating for adaptive spectral routing.

    Instead of a single FFT cross-correlation scalar alpha per channel,
    uses a small MLP to predict per-frequency-bin gating weights conditioned
    on both the predicted and reference magnitude spectra.  A learnable
    residual scale (initialised small) provides identity-like behaviour at
    the start of training.

    Input:  x     [B*N, T, D]  — decoder features
            x_ref [B*N, T, D]  — stem reference features
    Output: x_out [B*N, T, D]  — frequency-refined features
    """

    def __init__(self, D, reduction=4):
        super().__init__()
        self.D = D
        hidden = max(D // reduction, 8)

        # MLP: concat magnitudes [2D] → hidden → D (per-frequency gating weights)
        self.mlp = nn.Sequential(
            nn.Linear(2 * D, hidden),
            nn.GELU(),
            nn.Linear(hidden, D),
            nn.Sigmoid(),
        )

        # Learnable residual scale (starts near zero for identity-like init)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # Separate norms for predicted and reference features
        self.norm = nn.LayerNorm(D)
        self.norm_ref = nn.LayerNorm(D)

    def forward(self, x, x_ref):
        """
        x:     [B*N, T, D]
        x_ref: [B*N, T, D]
        Returns: [B*N, T, D]
        """
        # LayerNorm
        x = self.norm(x)
        x_ref = self.norm_ref(x_ref)

        # FFT to frequency domain
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')          # [B*N, freq, D] complex
        x_ref_fft = torch.fft.rfft(x_ref, dim=1, norm='ortho')  # [B*N, freq, D] complex

        # Magnitudes
        x_mag = torch.abs(x_fft)       # [B*N, freq, D]
        ref_mag = torch.abs(x_ref_fft) # [B*N, freq, D]

        # Concat magnitudes → predict per-frequency gating weights
        joint = torch.cat([x_mag, ref_mag], dim=-1)  # [B*N, freq, 2D]
        gate = self.mlp(joint)                         # [B*N, freq, D], values in (0,1)

        # Apply gate to complex spectrum (residual connection in frequency domain)
        x_fft_gated = x_fft * gate                     # complex * real → complex
        x_fft_out = x_fft + (x_fft_gated - x_fft) * self.residual_scale

        # IFFT back to time domain
        T = x.shape[1]
        x_out = torch.fft.irfft(x_fft_out, n=T, dim=1)  # [B*N, T, D]

        return x_out


# ==============================================================================
# Component Profiles — defines MSCD branches for each K_comp
# ==============================================================================

COMPONENT_PROFILES = {
    # K_comp = 3 (default): trend, periodic, mutation
    #   Cross-variable mixing is handled by SC²R, not by a separate component.
    3: OrderedDict([
        ('trend', {
            'kernel': 7, 'dilation': 1,
            'tags': ['smooth', 'long_term'],
        }),
        ('periodic', {
            'kernel': 3, 'dilations': [1, 2, 6, 12],
            'tags': ['rhythmic', 'multi_scale'],
        }),
        ('mutation', {
            'kernel': 3, 'residual': True,
            'tags': ['transient', 'local'],
        }),
    ]),
    # K_comp = 2: smooth + transient only
    2: OrderedDict([
        ('smooth', {
            'kernel': 7, 'dilation': 1,
            'tags': ['smooth', 'long_term'],
        }),
        ('transient', {
            'kernel': 3, 'residual': True,
            'tags': ['transient', 'local'],
        }),
    ]),
    # K_comp = 4: trend, periodic, mutation, spillover (legacy)
    4: OrderedDict([
        ('trend', {
            'kernel': 7, 'dilation': 1,
            'tags': ['smooth', 'long_term'],
        }),
        ('periodic', {
            'kernel': 3, 'dilations': [1, 2, 6, 12],
            'tags': ['rhythmic', 'multi_scale'],
        }),
        ('mutation', {
            'kernel': 3, 'residual': True,
            'tags': ['transient', 'local'],
        }),
        ('spillover', {
            'kernel': 3, 'cross_var_mix': True,
            'tags': ['cross', 'coarse'],
        }),
    ]),
    # K_comp = 6: trend, periodic, mutation, spillover, event, noise
    6: OrderedDict([
        ('trend', {
            'kernel': 7, 'dilation': 1,
            'tags': ['smooth', 'long_term'],
        }),
        ('periodic', {
            'kernel': 3, 'dilations': [1, 2, 6, 12],
            'tags': ['rhythmic', 'multi_scale'],
        }),
        ('mutation', {
            'kernel': 3, 'residual': True,
            'tags': ['transient', 'local'],
        }),
        ('spillover', {
            'kernel': 3, 'cross_var_mix': True,
            'tags': ['cross', 'coarse'],
        }),
        ('event', {
            'kernel': 3, 'calendar_embed': True,
            'tags': ['event', 'external'],
        }),
        ('noise', {
            'kernel': 1, 'identity_init': True,
            'tags': ['noise', 'residual'],
        }),
    ]),
}


def get_component_profile(K_comp):
    """Get the component profile for a given K_comp, or raise if unsupported."""
    if K_comp not in COMPONENT_PROFILES:
        supported = list(COMPONENT_PROFILES.keys())
        raise ValueError(
            f"K_comp={K_comp} has no preset profile. "
            f"Supported values: {supported}. "
            f"Use a custom profile dict for other K_comp values."
        )
    return COMPONENT_PROFILES[K_comp]


def build_exclusive_pairs(profile):
    """
    Auto-generate mutually exclusive component pairs from tags.
    Rule: 'smooth' components should be orthogonal to 'transient' components.
    """
    smooth_names = [k for k, v in profile.items() if 'smooth' in v.get('tags', [])]
    transient_names = [k for k, v in profile.items() if 'transient' in v.get('tags', [])]
    pairs = []
    for s in smooth_names:
        for t in transient_names:
            pairs.append((s, t))
    return pairs


# ==============================================================================
# MSCD: Multi-Scale Component Decomposition
# ==============================================================================

class MSCD(nn.Module):
    """
    Decomposes features into K_comp temporal components, each with its own
    convolutional configuration matching a target temporal pattern.

    Input:  F [B, N, T, D]
    Output: List of [B, N, T, d]   where d = D // K_comp, len = K_comp
    """

    def __init__(self, D, K_comp=4, profile=None):
        super().__init__()
        self.D = D
        self.K_comp = K_comp
        self.d = D // K_comp
        assert D % K_comp == 0, f"D ({D}) must be divisible by K_comp ({K_comp})"

        # Load profile
        if profile is None:
            profile = get_component_profile(K_comp)
        self.profile = profile
        self.component_names = list(profile.keys())
        self.exclusive_pairs = build_exclusive_pairs(profile)

        # Build one branch per component
        self.branches = nn.ModuleDict()
        for name, cfg in profile.items():
            self.branches[name] = self._build_branch(cfg)
            if cfg.get('cross_var_mix', False):
                setattr(self, f'_spillover_proj_{name}',
                        nn.Conv1d(2 * D, D, kernel_size=1))

    def _build_branch(self, cfg):
        """Build a single component branch based on its config dict."""
        layers = []
        in_ch = self.D

        # --- Special: identity branch (noise component) ---
        if cfg.get('identity_init', False):
            # Just a 1x1 conv from D → d (no temporal filtering)
            return nn.Conv1d(in_ch, self.d, kernel_size=1)

        # --- Special: calendar embed branch (event component) ---
        if cfg.get('calendar_embed', False):
            # Learnable embedding + time conv, simplified as Conv1d with
            # larger effective receptive field
            return nn.Sequential(
                nn.Conv1d(in_ch, self.d, kernel_size=cfg.get('kernel', 3),
                         padding=cfg.get('kernel', 3) // 2,
                         dilation=cfg.get('dilation', 1)),
                nn.GELU(),
                nn.Conv1d(self.d, self.d, kernel_size=3, padding=1),
            )

        # --- Multi-dilation branch (periodic component) ---
        dilations = cfg.get('dilations', None)
        if dilations is not None:
            # One conv per dilation, outputs summed
            branches = nn.ModuleList([
                nn.Conv1d(in_ch, self.d, kernel_size=cfg.get('kernel', 3),
                         padding=d * (cfg.get('kernel', 3) // 2),
                         dilation=d)
                for d in dilations
            ])
            # Wrap in a container that sums outputs
            return _MultiDilationBranch(branches)

        # --- Cross-variable mix branch (spillover component) ---
        if cfg.get('cross_var_mix', False):
            # Will be handled specially in forward — variable mix happens there
            return nn.Conv1d(in_ch, self.d, kernel_size=cfg.get('kernel', 3),
                           padding=cfg.get('kernel', 3) // 2,
                           dilation=cfg.get('dilation', 1))

        # --- Residual branch (mutation component) ---
        if cfg.get('residual', False):
            return _ResidualConvBranch(in_ch, self.d, cfg)

        # --- Standard branch (trend, etc.) ---
        return nn.Sequential(
            nn.Conv1d(in_ch, self.d, kernel_size=cfg.get('kernel', 7),
                     padding=cfg.get('kernel', 7) // 2,
                     dilation=cfg.get('dilation', 1)),
            nn.GELU(),
        )

    def forward(self, F):
        """
        F: [B, N, T, D]
        Returns: list of [B, N, T, d], one per component
        """
        B, N, T, D = F.shape
        components = []

        for name in self.component_names:
            cfg = self.profile[name]

            # Reshape to [B*N, D, T] for per-variable time convolution
            x = F.permute(0, 1, 3, 2).reshape(B * N, D, T)

            # --- Cross-var mix for spillover ---
            if cfg.get('cross_var_mix', False):
                # Mix across variables at each time step before time conv
                # x is [B*N, D, T], reshape to [B, N, D, T]
                x_4d = x.reshape(B, N, D, T)
                # Average over variables (coarse mix) and broadcast
                x_mixed = x_4d.mean(dim=1, keepdim=True)  # [B, 1, D, T]
                # Concatenate mixed info back
                x = torch.cat([x_4d, x_mixed.expand(-1, N, -1, -1)], dim=2)
                x = x.reshape(B * N, 2 * D, T)
                # Project back to D channels before conv (built in __init__)
                proj = getattr(self, f'_spillover_proj_{name}')
                x = proj(x)
                # Now apply time conv
                x = self.branches[name](x)  # [B*N, d, T]

            elif isinstance(self.branches[name], _ResidualConvBranch):
                x = self.branches[name](x)

            elif isinstance(self.branches[name], _MultiDilationBranch):
                x = self.branches[name](x)

            else:
                # Standard conv branch
                x = self.branches[name](x)  # [B*N, d, T]

            # Reshape back: [B*N, d, T] → [B, N, T, d]
            x = x.reshape(B, N, self.d, T).permute(0, 1, 3, 2)
            components.append(x)

        return components


class _MultiDilationBranch(nn.Module):
    """Sum of multiple Conv1d with different dilations."""
    def __init__(self, branches):
        super().__init__()
        self.branches = branches

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        return sum(outs)


class _ResidualConvBranch(nn.Module):
    """Conv with residual skip connection."""
    def __init__(self, in_ch, out_ch, cfg):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=cfg.get('kernel', 3),
            padding=cfg.get('kernel', 3) // 2,
            dilation=cfg.get('dilation', 1),
        )
        self.act = nn.GELU()
        # Skip: if in_ch != out_ch, need projection
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x)) + self.skip(x)


# ==============================================================================
# SC^2R: Sparse Cross-Variable Component Routing
# ==============================================================================

class SC2R(nn.Module):
    """
    Sparse cross-variable routing within one component type.
    Uses global TopK to select the most informative inter-variable edges.

    Input:  c [B, N, T, d]  — one component's features
    Output: c_new [B, N, T, d], routing_matrix [B, N, N]
    """

    def __init__(self, d, K_edges=None, M_avg=2, d_h=64):
        super().__init__()
        self.d = d
        self.K_edges = K_edges  # absolute count (overrides M_avg if set)
        self.M_avg = M_avg      # average edges per variable

        # Routing score network
        self.W = nn.Linear(2 * d, d_h)
        self.u = nn.Linear(d_h, 1)

        # Value projection
        self.V = nn.Linear(d, d)

        self.layer_norm = nn.LayerNorm(d)

    def forward(self, c):
        """
        c: [B, N, T, d]
        Returns: c_new [B, N, T, d], R [B, N, N]
        """
        B, N, T, d = c.shape

        # --- Step 1: Component summary (mean pool over time) ---
        s = c.mean(dim=2)  # [B, N, d]

        # --- Step 2: Pairwise routing scores ---
        # s_i: [B, N, 1, d] → expand to [B, N, N, d]
        s_i = s.unsqueeze(2).expand(-1, -1, N, -1)
        s_j = s.unsqueeze(1).expand(-1, N, -1, -1)
        pair = torch.cat([s_i, s_j], dim=-1)  # [B, N, N, 2d]

        # Score: u^T tanh(W · pair)
        scores = self.u(torch.tanh(self.W(pair))).squeeze(-1)  # [B, N, N]

        # Exclude self-loops
        diag_mask = torch.eye(N, device=c.device, dtype=torch.bool).unsqueeze(0)
        scores = scores.masked_fill(diag_mask, -float('inf'))

        # --- Step 3: Global TopK selection (batched, no per-sample loop) ---
        K_edges = self.K_edges if self.K_edges is not None else max(1, self.M_avg * N)
        R = self._sparse_routing_batched(scores, N, K_edges)  # [B, N, N]

        # --- Step 4: Message passing ---
        V_c = self.V(c)  # [B, N, T, d]

        # Weighted aggregation: for each target i, sum over source j
        # R: [B, N, N], V_c: [B, N, T, d]
        # m[b,i,t,:] = sum_j R[b,i,j] * V_c[b,j,t,:]
        m = torch.einsum('bij,bjtd->bitd', R, V_c)  # [B, N, T, d]

        # --- Step 5: Residual update ---
        c_new = self.layer_norm(c + m)
        return c_new, R

    def _sparse_routing_batched(self, scores, N, K_edges):
        """
        scores: [B, N, N] pairwise scores (diagonal = -inf)
        Returns: [B, N, N] row-normalized sparse routing matrix

        Fully vectorized over batch — no per-sample Python loop.
        Equivalent to calling the old per-sample _sparse_routing B times,
        but with a single topk / scatter / row-normalize pass.
        """
        B = scores.shape[0]

        # Flatten each [N, N] → [N*N]
        flat = scores.reshape(B, N * N)                    # [B, N*N]
        K_actual = min(K_edges, N * (N - 1))

        # Batch topk — all B samples in one kernel
        top_values, top_indices = torch.topk(flat, K_actual, dim=-1)  # [B, K], [B, K]

        # Build sparse routing matrix via scatter (single kernel)
        R = torch.zeros(B, N * N, device=scores.device, dtype=scores.dtype)
        top_weights = torch.softmax(top_values, dim=-1)    # [B, K]
        R.scatter_(1, top_indices, top_weights)            # [B, N*N]
        R = R.reshape(B, N, N)                             # [B, N, N]

        # Ensure min in-degree of 1 (no isolated target nodes)
        in_degrees = (R > 0).float().sum(dim=1)              # [B, N] — sum over sources
        isolated_mask = (in_degrees == 0)                     # [B, N]

        if isolated_mask.any():
            # For each isolated target, pick the best incoming edge
            # scores[:, :, j] → argmax over dim=1 gives best source per target
            best_sources = torch.argmax(scores, dim=1)        # [B, N]
            batch_idx, node_idx = torch.where(isolated_mask)  # flat indices
            best_i = best_sources[batch_idx, node_idx]
            R[batch_idx, best_i, node_idx] = 1.0

        # Row-normalize: each source's outgoing weights sum to 1
        row_sum = R.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, N, 1]
        R = R / row_sum

        return R


# ==============================================================================
# CGF: Conditional Gated Fusion
# ==============================================================================

class CGF(nn.Module):
    """
    Conditionally fuses encoder skip features with decoder upsampled features.

    For initial implementation, uses a per-layer learnable condition vector.
    Can be upgraded to time-step-level conditioning per §3.3.

    Input:  E [B, N, T, D]  — encoder skip
            D [B, N, T, D]  — decoder upsample
    Output: F [B, N, T, D]
    """

    def __init__(self, D, d_z=64):
        super().__init__()
        # Feature alignment for skip
        self.align = nn.Conv1d(D, D, kernel_size=1)

        # Gate generation MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_z, d_z),
            nn.ReLU(),
            nn.Linear(d_z, 2 * D),
            nn.Sigmoid(),
        )

        # Learnable per-layer condition (upgrade to time-step-level later)
        self.z_embed = nn.Parameter(torch.randn(1, d_z) * 0.02)

        # Post-fusion refinement
        self.refine_conv = nn.Conv1d(D, D, kernel_size=3, padding=1)
        self.refine_act = nn.GELU()
        self.refine_norm = nn.LayerNorm(D)

    def forward(self, E, D):
        """
        E: [B, N, T, D]  encoder skip connection
        D: [B, N, T, D]  decoder upsampled features
        """
        B, N, T, D_dim = D.shape

        # Align skip features to decoder channels
        # E: [B, N, T, D] → [B*N, D, T] → conv → [B*N, D, T] → [B, N, T, D]
        E_flat = E.permute(0, 1, 3, 2).reshape(B * N, D_dim, T)
        E_aligned = self.align(E_flat).reshape(B, N, D_dim, T)
        E_aligned = E_aligned.permute(0, 1, 3, 2)  # [B, N, T, D]

        # Gate generation from condition
        z = self.z_embed.expand(B, -1)  # [B, d_z]
        gates = self.mlp(z)  # [B, 2*D]
        alpha = gates[:, :D_dim].unsqueeze(1).unsqueeze(2)   # [B, 1, 1, D]
        beta = gates[:, D_dim:].unsqueeze(1).unsqueeze(2)    # [B, 1, 1, D]

        # Weighted fusion
        F = alpha * E_aligned + beta * D  # [B, N, T, D]

        # Refinement
        F_flat = F.permute(0, 1, 3, 2).reshape(B * N, D_dim, T)
        F_refined = self.refine_act(self.refine_conv(F_flat))  # [B*N, D, T]
        # LayerNorm expects last dim = D, so permute [B*N, D, T] → [B*N, T, D]
        F_refined = self.refine_norm(F_refined.permute(0, 2, 1))  # [B*N, T, D]
        F_refined = F_refined.permute(0, 2, 1)  # [B*N, D, T]
        F_refined = F_refined.reshape(B, N, D_dim, T)
        F_refined = F_refined.permute(0, 1, 3, 2)  # [B, N, T, D]

        return F_refined


# ==============================================================================
# DAU-Net Main Model
# ==============================================================================

class Model(nn.Module):
    """
    DAU-Net: Scale-Adaptive U-Net

    Configs used by DAU-Net:
        mscd_K        : K_comp, number of MSCD component types (default 3)
        sc2r_M_avg    : avg edges per variable for SC^2R (default 2)
        sc2r_K_edges  : absolute K_edges override (default None)
        sc2r_d_h      : hidden dim for SC^2R score network (default 64)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in       # N = number of variables
        self.d_model = configs.d_model     # D
        self.e_layers = configs.e_layers   # L
        self.dropout = configs.dropout

        # --- MSCD hyperparams ---
        self.K_comp = getattr(configs, 'mscd_K', 3)
        self.component_profile = get_component_profile(self.K_comp)
        self.component_names = list(self.component_profile.keys())
        self.exclusive_pairs = build_exclusive_pairs(self.component_profile)
        self.d_per_comp = self.d_model // self.K_comp  # d

        # --- SC^2R hyperparams ---
        self.M_avg = getattr(configs, 'sc2r_M_avg', 2)
        self.K_edges = getattr(configs, 'sc2r_K_edges', None)
        self.d_h = getattr(configs, 'sc2r_d_h', 64)

        # ========================
        # Layers
        # ========================

        # RevIN normalization
        self.revin = RevIN(self.enc_in)

        # Coarse predictor (粗糙预测): seq_len → seq_len + pred_len
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)

        # Stem: feature projection + initial time compression
        self.stem_conv = nn.Conv1d(1, self.d_model, kernel_size=7, stride=2, padding=3)
        self.pos_embed = PositionalEmbedding(self.d_model)

        # Stem dropout
        self.stem_dropout = nn.Dropout(self.dropout)

        # Encoder: L layers of (MSCD → SC^2R → Fuse → DownSample)
        self.encoder_mscd = nn.ModuleList([
            MSCD(self.d_model, self.K_comp, profile=self.component_profile)
            for _ in range(self.e_layers)
        ])
        self.encoder_sc2r = nn.ModuleList([
            nn.ModuleList([
                SC2R(self.d_per_comp,
                     K_edges=self.K_edges,
                     M_avg=self.M_avg,
                     d_h=self.d_h)
                for _ in range(self.K_comp)
            ])
            for _ in range(self.e_layers)
        ])
        self.encoder_fuse = nn.ModuleList([
            nn.Conv2d(self.d_model, self.d_model, kernel_size=1)
            for _ in range(self.e_layers)
        ])
        self.encoder_down = nn.ModuleList([
            nn.Sequential(
                # Downsample time dimension by ×2
                nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            )
            for _ in range(self.e_layers)
        ])
        self.encoder_norm = nn.ModuleList([
            nn.LayerNorm(self.d_model)
            for _ in range(self.e_layers)
        ])

        # Decoder: L layers of (Upsample → CGF)
        # Upsample convs (upsampling itself done via F.interpolate with target size)
        self.decoder_up_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
                nn.GELU(),
            )
            for _ in range(self.e_layers)
        ])
        self.decoder_cgf = nn.ModuleList([
            CGF(self.d_model)
            for _ in range(self.e_layers)
        ])
        self.decoder_norm = nn.ModuleList([
            nn.LayerNorm(self.d_model)
            for _ in range(self.e_layers)
        ])

        # Conv1d with stride 2 produces ceil(T_total / 2) samples.
        T_stem = (self.seq_len + self.pred_len + 1) // 2

        # AdaptiveSpectralGate: learnable frequency-domain gating
        # Dynamic spectral routing branch.
        gate_reduction = getattr(configs, 'gate_reduction', 4)
        self.spectral_gate = AdaptiveSpectralGate(self.d_model, reduction=gate_reduction)

        # Head: flatten features → predict pred_len
        self.head = nn.Linear(self.d_model * T_stem, self.pred_len)

    # ========================================================================
    # Forecast
    # ========================================================================

    def forecast(self, x_input, x_mark_input):
        """
        x_input: [B, seq_len, N]
        Returns: [B, pred_len, N]
        """
        B = x_input.shape[0]
        N = self.enc_in
        D = self.d_model
        L = self.e_layers

        # ---- 1. RevIN Normalize ----
        x_ori = x_input.contiguous()
        x = self.revin(x_input, 'norm')  # [B, seq_len, N]

        # ---- 2. Coarse predictor (粗糙预测) ----
        x = x.permute(0, 2, 1)                        # [B, N, seq_len]
        x = self.predict_linear(x)                     # [B, N, seq_len + pred_len]
        T_total = self.seq_len + self.pred_len
        x = x.permute(0, 2, 1)                         # [B, T_total, N]

        # ---- 3. Stem: feature projection + initial compression ----
        x = x.permute(0, 2, 1)                         # [B, N, T_total]
        x = x.reshape(B * N, 1, T_total)               # [B*N, 1, T_total]
        x = self.stem_conv(x)                           # [B*N, D, T_stem]
        T_stem = x.shape[-1]
        x = x.reshape(B, N, D, T_stem)                 # [B, N, D, T_stem]
        x = x.permute(0, 1, 3, 2)                      # [B, N, T_stem, D]

        # Position embedding
        pos = self.pos_embed(x[:, 0, :, :])            # [1, T_stem, D]
        x = x + pos.unsqueeze(1)                        # broadcast over N
        x = self.stem_dropout(x)

        # Save stem reference for AdaptiveSpectralGate
        x_old = x.clone()  # reference signal for frequency-domain gating

        # ---- 4. Encoder ----
        skips = []
        for i in range(L):
            B_cur, N_cur, T_cur, D_cur = x.shape

            # MSCD: decompose into K_comp components
            components = self.encoder_mscd[i](x)
            # components: list of [B, N, T, d], len = K_comp

            # SC^2R: route within each component
            new_components = []
            for k in range(self.K_comp):
                c_new, _ = self.encoder_sc2r[i][k](components[k])
                new_components.append(c_new)
            # new_components: list of [B, N, T, d]

            # Fuse: concat along channel dim → [B, N, T, D]
            x_cat = torch.cat(new_components, dim=-1)  # [B, N, T, D]

            # 1×1 Conv2d for learned fusion
            x_fused = self.encoder_fuse[i](
                x_cat.permute(0, 3, 1, 2)               # [B, D, N, T]
            ).permute(0, 2, 3, 1)                        # [B, N, T, D]

            x_fused = self.encoder_norm[i](x_fused)

            # Store skip connection (before downsampling)
            skips.append(x_fused)

            # Downsample time dimension
            x_fused_flat = x_fused.permute(0, 1, 3, 2).reshape(B * N, D, T_cur)
            x_down = self.encoder_down[i](x_fused_flat)  # [B*N, D, T_cur/2]
            T_next = x_down.shape[-1]
            x = x_down.reshape(B, N, D, T_next).permute(0, 1, 3, 2)  # [B, N, T_next, D]

        # ---- 5. Decoder ----
        for i in range(L - 1, -1, -1):
            B_cur, N_cur, T_cur, D_cur = x.shape
            skip = skips[i]
            T_target = skip.shape[2]  # target time dim from skip connection

            # Upsample to match skip connection's time dimension
            x_flat = x.permute(0, 1, 3, 2).reshape(B * N, D, T_cur)
            x_up = F.interpolate(x_flat, size=T_target, mode='linear',
                                 align_corners=True)      # [B*N, D, T_target]
            x_up = self.decoder_up_conv[i](x_up)          # [B*N, D, T_target]
            T_up = x_up.shape[-1]
            x = x_up.reshape(B, N, D, T_up).permute(0, 1, 3, 2)  # [B, N, T_up, D]

            # CGF: fuse with skip connection
            x = self.decoder_cgf[i](skip, x)               # [B, N, T_up, D]
            x = self.decoder_norm[i](x)

        # ---- 6. Adaptive Spectral Gate ----
        # Learn per-frequency-bin gating. Complements MSCD's time-domain
        # component decomposition with frequency-domain spectral refinement.
        #
        # Flatten [B, N, T_stem, D] → [B*N, T_stem, D]
        x_old_flat = x_old.permute(0, 1, 3, 2).reshape(B * N, D, T_stem)
        x_old_flat = x_old_flat.permute(0, 2, 1)  # [B*N, T_stem, D]

        x_flat = x.permute(0, 1, 3, 2).reshape(B * N, D, T_stem)
        x_flat = x_flat.permute(0, 2, 1)  # [B*N, T_stem, D]

        x_flat = self.spectral_gate(x_flat, x_old_flat)  # [B*N, T_stem, D]

        # ---- 7. Head: flatten + project to pred_len ----
        # x_flat: [B*N, T_stem, D] → reshape → [B*N, D*T_stem] → Linear → [B*N, pred_len]
        x_flat = x_flat.permute(0, 2, 1)            # [B*N, D, T_stem]
        x_flat = x_flat.reshape(B * N, D * T_stem)  # [B*N, D*T_stem]
        x_out = self.head(x_flat)                    # [B*N, pred_len]
        x_out = x_out.reshape(B, N, self.pred_len)   # [B, N, pred_len]
        x_out = x_out.permute(0, 2, 1)               # [B, pred_len, N]

        # ---- 8. RevIN Denormalize ----
        x_out = self.revin(x_out, 'denorm')

        return x_out

    # ========================================================================
    # Forward — standard forecasting interface
    # ========================================================================

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Standard forecasting model interface.
        x_enc:       [B, seq_len, N]
        x_mark_enc:  [B, seq_len, time_feat_dim]  (unused by DAU-Net)
        x_dec:       [B, label_len+pred_len, N]   (unused)
        x_mark_dec:  [B, label_len+pred_len, ...] (unused)
        Returns:     [B, pred_len, N]
        """
        out = self.forecast(x_enc, x_mark_enc)
        return out

    # ========================================================================
    # Loss helpers (for use in training script)
    # ========================================================================

    def compute_orthogonality_loss(self, encoder_outputs):
        """
        Compute L_orth between mutually exclusive component pairs.
        encoder_outputs: list of lists —
            encoder_outputs[layer][k] = [B, N, T, d] for component k at layer l
        Returns: scalar loss
        """
        if not self.exclusive_pairs:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        total_loss = 0.0
        count = 0
        for layer_comps in encoder_outputs:
            for k1_name, k2_name in self.exclusive_pairs:
                k1_idx = self.component_names.index(k1_name)
                k2_idx = self.component_names.index(k2_name)
                c1 = layer_comps[k1_idx]  # [B, N, T, d]
                c2 = layer_comps[k2_idx]  # [B, N, T, d]

                # Cosine similarity squared
                c1_norm = F.normalize(c1, dim=-1)
                c2_norm = F.normalize(c2, dim=-1)
                cos_sim = (c1_norm * c2_norm).sum(dim=-1)  # [B, N, T]
                total_loss += (cos_sim ** 2).mean()
                count += 1

        return total_loss / max(count, 1)

    def compute_consistency_loss(self, routing_matrices):
        """
        Compute L_cons (Jaccard edge-structure consistency) across adjacent layers.
        routing_matrices: list of [B, N, N] per encoder layer
        Returns: scalar loss
        """
        if len(routing_matrices) < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        total_loss = 0.0
        for ell in range(len(routing_matrices) - 1):
            # Average routing matrix over components for each layer
            R_ell = routing_matrices[ell]     # [B, N, N]
            R_next = routing_matrices[ell + 1]  # [B, N, N]

            # Binarize
            A_ell = (R_ell > 0).float()
            A_next = (R_next > 0).float()

            # Jaccard distance
            intersection = (A_ell * A_next).sum(dim=(1, 2))
            union = (A_ell + A_next).clamp(0, 1).sum(dim=(1, 2))
            jaccard = 1.0 - intersection / (union + 1e-8)

            # Coarse-only penalty
            coarse_only = ((A_next * (1 - A_ell)).sum(dim=(1, 2)) /
                          (A_next.sum(dim=(1, 2)) + 1e-8))

            total_loss += (jaccard + 0.5 * coarse_only).mean()

        return total_loss / (len(routing_matrices) - 1)
