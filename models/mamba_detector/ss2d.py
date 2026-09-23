"""
2D Selective Scan (SS2D) Module in Pure PyTorch
=================================================
Core State Space Model (SSM) layer for 2D visual data, adapted from VMamba
(Liu et al., 2024: Visual State Space Model).

Features:
- Pure PyTorch tensor operations: zero CUDA compilation (works on Windows, Linux, CPU, and GPU).
- 4-way cross-scan mechanism (CSM): scans image horizontally, vertically, and in reverse.
- Input-dependent selectivity: computes dynamic delta, B, and C matrices.
- Linear O(H * W) computational complexity.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan_pure_pytorch(
    u: torch.Tensor,       # (B, D, L)
    delta: torch.Tensor,   # (B, D, L)
    A: torch.Tensor,       # (D, N)
    B: torch.Tensor,       # (B, N, L)
    C: torch.Tensor,       # (B, N, L)
    D: Optional[torch.Tensor] = None, # (D,)
) -> torch.Tensor:
    """
    Pure PyTorch implementation of the selective scan recurrence.

    h_t = exp(delta_t * A) * h_{t-1} + (delta_t * B_t) * u_t
    y_t = C_t * h_t + D * u_t
    """
    b, d, l = u.shape
    n = A.shape[1]

    # Compute discretized A: exp(delta * A) -> (B, D, L, N)
    # delta: (B, D, L, 1), A: (1, D, 1, N)
    delta_A = torch.exp(delta.unsqueeze(-1) * A.view(1, d, 1, n)) # (B, D, L, N)

    # Compute discretized B * u: (delta * u) * B
    # delta_u: (B, D, L, 1), B: (B, 1, L, N) -> (B, D, L, N)
    delta_u = (delta * u).unsqueeze(-1)
    B_expanded = B.transpose(1, 2).unsqueeze(1) # (B, 1, L, N)
    delta_B_u = delta_u * B_expanded # (B, D, L, N)

    # Recurrence along sequence length L
    # We maintain hidden state h of shape (B, D, N)
    h = torch.zeros(b, d, n, dtype=u.dtype, device=u.device)
    ys = []

    C_transposed = C.transpose(1, 2) # (B, L, N)

    for t in range(l):
        # h_t = delta_A_t * h_{t-1} + delta_B_u_t
        h = delta_A[:, :, t, :] * h + delta_B_u[:, :, t, :]
        # y_t = sum_n (h_{t, n} * C_{t, n}) -> (B, D)
        # h: (B, D, N), C_t: (B, 1, N)
        C_t = C_transposed[:, t:t+1, :] # (B, 1, N)
        y_t = torch.sum(h * C_t, dim=-1) # (B, D)
        ys.append(y_t)

    y = torch.stack(ys, dim=-1) # (B, D, L)

    if D is not None:
        y = y + u * D.view(1, d, 1)

    return y


class SS2D(nn.Module):
    """
    2D Selective Scan (Cross-Scan Module)
    Traverses 2D feature maps in 4 directions to build non-causal 2D visual context.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: Optional[int] = None,
        act_layer=nn.SiLU,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank is None else dt_rank

        # In-project: splits into input x and gate z
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # 2D Depthwise convolution to capture immediate local inductive biases
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=3,
            padding=1,
        )
        self.act = act_layer()

        # SSM parameters for the 4 scanning directions
        # dt_proj maps from dt_rank to d_inner
        self.dt_projs = nn.Parameter(torch.empty(4, self.d_inner, self.dt_rank))
        self.dt_bias = nn.Parameter(torch.empty(4, self.d_inner))

        # Input-dependent B and C projection: maps from d_inner to (dt_rank + 2 * d_state)
        self.x_proj = nn.Linear(self.d_inner, 4 * (self.dt_rank + 2 * self.d_state), bias=False)

        # S4 parameter A: log-space initialization for numerical stability
        # A is negative: A = -exp(A_log)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_logs = nn.Parameter(torch.log(A).repeat(4, 1, 1)) # (4, d_inner, d_state)

        # Skip connection D parameter
        self.D = nn.Parameter(torch.ones(4, self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # Initialize dt_projs
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_projs, -dt_init_std, dt_init_std)

        # Initialize dt_bias so softplus(dt_bias) ~ dt_init
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_bias.copy_(inv_dt.repeat(4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) or (B, H, W, C)
        Returns:
            y: (B, C, H, W)
        """
        # Ensure input format is (B, H, W, C) for linear projections
        if x.dim() == 4 and x.shape[1] == self.d_model:
            B, C, H, W = x.shape
            x_in = x.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)
        else:
            B, H, W, C = x.shape
            x_in = x

        # 1. In-projection -> x_proj, z_proj
        xz = self.in_proj(x_in) # (B, H, W, 2 * d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1) # (B, H, W, d_inner) each

        # 2. Local 2D depthwise convolution on x_branch
        x_conv = x_branch.permute(0, 3, 1, 2).contiguous() # (B, d_inner, H, W)
        x_conv = self.act(self.conv2d(x_conv)) # (B, d_inner, H, W)

        # 3. Four-way Cross-Scan
        L = H * W
        # Direction 1: Row-wise forward
        scan1 = x_conv.view(B, self.d_inner, L) # (B, d_inner, L)
        # Direction 2: Row-wise backward
        scan2 = torch.flip(scan1, dims=[-1])
        # Direction 3: Column-wise forward
        scan3 = x_conv.permute(0, 1, 3, 2).contiguous().view(B, self.d_inner, L)
        # Direction 4: Column-wise backward
        scan4 = torch.flip(scan3, dims=[-1])

        # Stack scans: (4, B, d_inner, L)
        xs = torch.stack([scan1, scan2, scan3, scan4], dim=0)

        # 4. Compute input-dependent projections (delta, B, C)
        # x_branch: (B, H, W, d_inner) -> (B, L, d_inner)
        x_flat = x_conv.permute(0, 2, 3, 1).contiguous().view(B, L, self.d_inner)
        x_dbl = self.x_proj(x_flat) # (B, L, 4 * (dt_rank + 2 * d_state))
        x_dbl = x_dbl.view(B, L, 4, -1).permute(2, 0, 3, 1).contiguous() # (4, B, dt_rank + 2*d_state, L)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        # dts: (4, B, dt_rank, L), Bs: (4, B, d_state, L), Cs: (4, B, d_state, L)

        # Apply dt projection
        # dt_projs: (4, d_inner, dt_rank), dts: (4, B, dt_rank, L)
        # output: (4, B, d_inner, L)
        dts_proj = torch.einsum("k d r, k b r l -> k b d l", self.dt_projs, dts)
        dts_proj = F.softplus(dts_proj + self.dt_bias.unsqueeze(1).unsqueeze(-1))

        # 5. Execute Selective Scan across all 4 directions
        ys = []
        for k in range(4):
            A_k = -torch.exp(self.A_logs[k].float()) # (d_inner, d_state)
            y_k = selective_scan_pure_pytorch(
                u=xs[k],
                delta=dts_proj[k],
                A=A_k,
                B=Bs[k],
                C=Cs[k],
                D=self.D[k],
            )
            ys.append(y_k)

        # 6. Reverse and merge 4 scans back into 2D canvas
        out1 = ys[0].view(B, self.d_inner, H, W)
        out2 = torch.flip(ys[1], dims=[-1]).view(B, self.d_inner, H, W)
        out3 = ys[2].view(B, self.d_inner, W, H).permute(0, 1, 3, 2).contiguous()
        out4 = torch.flip(ys[3], dims=[-1]).view(B, self.d_inner, W, H).permute(0, 1, 3, 2).contiguous()

        y = out1 + out2 + out3 + out4 # (B, d_inner, H, W)
        y = y.permute(0, 2, 3, 1).contiguous() # (B, H, W, d_inner)

        # 7. Multiplicative Gating + LayerNorm + Out Projection
        y = self.out_norm(y)
        y = y * F.silu(z_branch)
        out = self.out_proj(y) # (B, H, W, d_model)
        out = self.dropout(out)

        # Return in (B, C, H, W) format
        return out.permute(0, 3, 1, 2).contiguous()


class VSSBlock(nn.Module):
    """
    Visual State Space Block (VSSBlock)
    Equivalent to a Transformer block, but replacing multi-head self-attention with SS2D.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim) # GroupNorm(1) == LayerNorm over channels
        self.ss2d = SS2D(
            d_model=dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        residual = x
        x_norm = self.norm(x)
        out = self.ss2d(x_norm)
        return residual + out
