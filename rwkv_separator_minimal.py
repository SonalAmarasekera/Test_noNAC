
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- small utilities -------------------------

class Snake(nn.Module):
    """
    Snake activation (simplified): x + (1/a) * sin^2(a * x).
    Works well as a smooth periodic nonlinearity for audio features.
    """
    def __init__(self, a: float = 1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a)))

    def forward(self, x):
        a = self.a.abs() + 1e-6
        return x + (torch.sin(a * x) ** 2) / a

class PreNorm(nn.Module):
    def __init__(self, dim, mod):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mod = mod

    def forward(self, x, *args, **kwargs):
        return self.mod(self.norm(x), *args, **kwargs)

# ------------------------- RWKV-like blocks -------------------------

class TimeMix(nn.Module):
    """
    A lightweight RWKV-style time-mixing cell:
      - token shift (x_t, x_{t-1})
      - gated mixing with learned decay using a causal depthwise 1D conv
    Note: This is a pragmatic stand-in compatible with [B,T,C] tensors.
    """
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.token_shift = nn.ZeroPad2d((0,0,1,0))  # shift along T by 1 (prepend zero row)
        # depthwise conv mixes along time causally
        padding = kernel_size - 1
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim, bias=False)
        # make it causal by slicing
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.constant_(self.dw.weight, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C]
        B, T, C = x.shape
        # causal dw-conv
        x1 = x.transpose(1, 2)                                    # [B,C,T]
        y = self.dw(x1)[:, :, :T]                                  # [B,C,T] causal slice
        y = y.transpose(1, 2)                                      # [B,T,C]
        g = torch.sigmoid(self.gate(x))
        out = self.proj(g * (x + y))
        return out

class ChannelMix(nn.Module):
    """Two-layer MLP with Snake nonlinearity."""
    def __init__(self, dim: int, mult: int = 2):
        super().__init__()
        hidden = dim * mult
        self.fc1 = nn.Linear(dim, hidden)
        self.act = Snake()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class UniRWKVBlock(nn.Module):
    """Pre-LN -> TimeMix -> dropout -> residual -> Pre-LN -> ChannelMix -> dropout -> residual"""
    def __init__(self, dim: int, dropout: float = 0.1, kernel_size:int=3, mlp_mult:int=2):
        super().__init__()
        self.time = PreNorm(dim, TimeMix(dim, kernel_size))
        self.chan = PreNorm(dim, ChannelMix(dim, mlp_mult))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.time(x))
        x = x + self.drop(self.chan(x))
        return x

class BiRWKVStack(nn.Module):
    """
    Bidirectional stack with Direction Dropout:
      - forward path uses order t=0..T-1
      - backward path uses flipped order and flips back
      - during training, randomly drop one direction with prob=dir_drop_p
    Fusion: simple average (can be upgraded later).
    """
    def __init__(self, dim: int, layers: int = 6, dropout: float = 0.1,
                 dir_drop_p: float = 0.0, kernel_size:int=3, mlp_mult:int=2):
        super().__init__()
        self.fwd = nn.ModuleList([UniRWKVBlock(dim, dropout, kernel_size, mlp_mult) for _ in range(layers)])
        self.bwd = nn.ModuleList([UniRWKVBlock(dim, dropout, kernel_size, mlp_mult) for _ in range(layers)])
        self.dir_drop_p = dir_drop_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Forward path
        xf = x
        for blk in self.fwd:
            xf = blk(xf)

        # Backward path
        xb = torch.flip(x, dims=[1])
        for blk in self.bwd:
            xb = blk(xb)
        xb = torch.flip(xb, dims=[1])

        if self.training and self.dir_drop_p > 0.0:
            # Drop one direction stochastically (per batch) to support uni inference
            if torch.rand(()) < self.dir_drop_p:
                return xf
            if torch.rand(()) < self.dir_drop_p:
                return xb

        # Fuse
        return 0.5 * (xf + xb)

# ------------------------- Separator (minimal) -------------------------

@dataclass
class RWKVSeparatorConfig:
    in_dim: int              # input latent channels C
    hidden_dim: Optional[int] = None  # if None, use in_dim//2 after downproj
    layers: int = 6
    dropout: float = 0.1
    dir_drop_p: float = 0.0
    kernel_size: int = 3
    mlp_mult: int = 2
    heads_use_mask: bool = True   # if True, predict mask+resid; else pure residual

class RWKVSeparatorMinimal(nn.Module):
    """
    Conv1x1 down (C -> C/2 or hidden_dim) -> Bi-RWKV stack -> Conv1x1 up (-> C) -> two heads.
    Each head predicts:
      - mask m_s via sigmoid (optional)
      - residual r_s
      Predicted latent: y_s = m_s * z_mix + r_s   (if heads_use_mask) else y_s = z_mix + r_s
    """
    def __init__(self, cfg: RWKVSeparatorConfig):
        super().__init__()
        C = cfg.in_dim
        H = cfg.hidden_dim or (C // 2)
        assert H > 0, "hidden_dim must be > 0"
        self.cfg = cfg

        self.down = nn.Linear(C, H)    # 1x1 conv across channels (implemented as Linear)
        self.core = BiRWKVStack(H, layers=cfg.layers, dropout=cfg.dropout,
                                dir_drop_p=cfg.dir_drop_p, kernel_size=cfg.kernel_size, mlp_mult=cfg.mlp_mult)
        self.up   = nn.Linear(H, C)

        # Heads: small MLPs
        head_hidden = max(64, C)
        def make_head():
            return nn.Sequential(
                nn.LayerNorm(C),
                nn.Linear(C, head_hidden),
                Snake(),
                nn.Linear(head_hidden, C),
            )
        self.resid1 = make_head()
        self.resid2 = make_head()
        if cfg.heads_use_mask:
            self.mask1 = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, C))
            self.mask2 = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, C))
        else:
            self.mask1 = self.mask2 = None

    def forward(self, z_mix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        z_mix: [B,T,C]
        Returns dict with:
          pred1, pred2: [B,T,C]
          (optionally) mask1, mask2, resid1, resid2
        """
        B, T, C = z_mix.shape
        h = self.down(z_mix)
        h = self.core(h)
        h = self.up(h)

        r1 = self.resid1(h)
        r2 = self.resid2(h)
        if self.mask1 is not None:
            m1 = torch.sigmoid(self.mask1(h))
            m2 = torch.sigmoid(self.mask2(h))
            y1 = m1 * z_mix + r1
            y2 = m2 * z_mix + r2
            return {"pred1": y1, "pred2": y2, "mask1": m1, "mask2": m2, "resid1": r1, "resid2": r2}
        else:
            y1 = z_mix + r1
            y2 = z_mix + r2
            return {"pred1": y1, "pred2": y2, "resid1": r1, "resid2": r2}


# ------------------------- quick smoke test -------------------------

if __name__ == "__main__":
    B, T, C = 2, 400, 128
    cfg = RWKVSeparatorConfig(in_dim=C, layers=4, dir_drop_p=0.5, dropout=0.1)
    model = RWKVSeparatorMinimal(cfg)
    x = torch.randn(B, T, C)
    out = model(x)
    print({k: tuple(v.shape) for k, v in out.items()})
