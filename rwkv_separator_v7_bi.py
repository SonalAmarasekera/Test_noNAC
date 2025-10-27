# rwkv_separator_v7_bi.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn

# Adjust this import to your actual RWKV v7 kernel path if different.
from RWKV.RWKV_v7.train_temp.src.model import RWKV_Tmix_x070, RWKV_CMix_x070

# ----------------------- Configs -----------------------

@dataclass
class V7Args:
    """Minimal arg set required by RWKV_Tmix_x070 / RWKV_CMix_x070."""
    n_embd: int                 # internal channel dim
    n_layer: int                # number of stacked layers
    dim_att: int                # timemix inner dim (use n_embd)
    head_size_a: int            # head size must divide n_embd
    my_testing: str = "x070"    # ensure x070 logic in the kernels
    head_size_divisor: int = 64
    pre_ffn: int = 0
    my_pos_emb: int = 0

@dataclass
class SeparatorV7Config:
    in_dim: int                 # input feature dim (e.g., 513 for STFT mag)
    layers: int = 6
    head_size_a: int = 64
    hidden_dim: Optional[int] = None   # auto -> round to multiple of head_size_a
    dir_drop_p: float = 0.0            # direction dropout prob
    use_mask: bool = True              # mask heads enabled
    enforce_bf16: bool = True          # activations in bf16 at fused ops

# ----------------------- Layers -----------------------
class V7Layer(nn.Module):
    """One x070 layer: PreLN -> TimeMix -> +res -> PreLN -> ChannelMix -> +res, with v_first plumbing."""
    def __init__(self, args: V7Args, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        # Keep LN weights in fp32 for numerical stability
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.tmix = RWKV_Tmix_x070(args, layer_id)
        self.cmix = RWKV_CMix_x070(args, layer_id)

    @staticmethod
    def _to_bf16_contig(t: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is bf16 and contiguous (required for fused CUDA kernels)."""
        if t.is_cuda and t.dtype != torch.bfloat16:
            t = t.to(torch.bfloat16)
        return t.contiguous()

    def forward(self, x: torch.Tensor, v_first: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        use_cuda = x.is_cuda

        # --- TimeMix ---
        x1 = self.ln1(x.float())              # LN in fp32
        x1 = self._to_bf16_contig(x1)         # kernel wants bf16 + contiguous
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            h, v_first = self.tmix(x1, v_first)
        x = x + h                              # residual

        # --- ChannelMix ---
        x2 = self.ln2(x.float())
        x2 = self._to_bf16_contig(x2)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            x = x + self.cmix(x2)

        return x, v_first

class V7Core(nn.Module):
    """Uni-directional stack of V7Layer."""
    def __init__(self, args: V7Args):
        super().__init__()
        self.layers = nn.ModuleList([V7Layer(args, i) for i in range(args.n_layer)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_first: Optional[torch.Tensor] = None
        for lyr in self.layers:
            x, v_first = lyr(x, v_first)
        return x

class BiV7Core(nn.Module):
    """Bidirectional wrapper with optional Direction Dropout."""
    def __init__(self, args: V7Args, dir_drop_p: float = 0.0):
        super().__init__()
        self.dir_drop_p = float(dir_drop_p)
        self.fwd = V7Core(args)
        self.bwd = V7Core(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.dir_drop_p > 0.0:
            u = torch.rand((), device=x.device)
            if u < 0.5 * self.dir_drop_p:
                return self.fwd(x)
            if u < self.dir_drop_p:
                xb = torch.flip(x, dims=[1]).contiguous()
                xb = self.bwd(xb)
                return torch.flip(xb, dims=[1]).contiguous()

        xf = self.fwd(x)
        xb = torch.flip(x, dims=[1]).contiguous()
        xb = self.bwd(xb)
        xb = torch.flip(xb, dims=[1]).contiguous()
        return 0.5 * (xf + xb)

# ---------------------- Helper --------------------------
def pad_to_chunk(x: torch.Tensor, chunk_len: int = 16) -> Tuple[torch.Tensor, int]:
    """Pad time dimension to a multiple of chunk_len. Returns (x_pad, T_orig)."""
    B, T, C = x.shape
    pad = (chunk_len - (T % chunk_len)) % chunk_len
    if pad > 0:
        pad_tensor = torch.zeros(B, pad, C, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad_tensor], dim=1)
    return x, T

# ----------------------- Separator -----------------------
class RWKVv7Separator(nn.Module):
    """
    Linear down (C -> H) -> Bi V7 core (L layers) -> Linear up (H -> C) -> mask head.
    H is adjusted so that H % head_size_a == 0 (required by v7 kernels).
    For the STFT-mask smoke test, we return only {"mask1","mask2"} in [B,T,C].
    """
    def __init__(self, cfg: SeparatorV7Config):
        super().__init__()

        # Keep the kernel env in sync with the configuration (compile-once safety).
        os.environ.setdefault("RWKV_MY_TESTING", "x070")
        os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
        os.environ["RWKV_HEAD_SIZE_A"] = str(cfg.head_size_a)

        C = int(cfg.in_dim)
        if cfg.hidden_dim is None:
            # pick >= C//2 and round UP to multiple of head_size_a
            Happrox = max(C // 2, cfg.head_size_a)
            H = (Happrox + cfg.head_size_a - 1) // cfg.head_size_a * cfg.head_size_a
        else:
            H = int(cfg.hidden_dim)
            if H % cfg.head_size_a != 0:
                H = (H + cfg.head_size_a - 1) // cfg.head_size_a * cfg.head_size_a  # round up safely

        assert H % cfg.head_size_a == 0, f"hidden_dim {H} must be divisible by head_size_a={cfg.head_size_a}"

        self.cfg = cfg
        self.in_dim = C
        self.hid_dim = H

        # Channel 1x1 convs as Linear
        self.down = nn.Linear(C, H)

        # RWKV-v7 Bi core
        v7args = V7Args(
            n_embd=H,
            n_layer=cfg.layers,
            dim_att=H,
            head_size_a=cfg.head_size_a,
            my_testing="x070",
        )
        self.core = BiV7Core(v7args, dir_drop_p=cfg.dir_drop_p)

        self.up = nn.Linear(H, C)

        # Mask head: produce 2*C logits per frame, then softmax over the 2 sources
        self.use_mask = bool(cfg.use_mask)
        if self.use_mask:
            self.head_m = nn.Sequential(
                nn.LayerNorm(C),   # LN in fp32, autocast will cast activations as needed
                nn.Linear(C, 2 * C)
            )
        else:
            self.head_m = None  # (not used in the smoke test)

    def _pre_core_cast(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.enforce_bf16 and x.is_cuda and x.dtype is not torch.bfloat16:
            x = x.to(torch.bfloat16)
        return x.contiguous()

    def forward(self, z_mix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        z_mix: [B,T,C] float{32|16|bf16}
        Returns:
            {"mask1": [B,T,C], "mask2": [B,T,C]}  (for STFT-mask pipeline)
        """
        assert z_mix.dim() == 3, f"Expected [B,T,C], got {tuple(z_mix.shape)}"
        B, T_orig, C = z_mix.shape
        assert C == self.in_dim, f"Input C={C} != configured C={self.in_dim}. Re-cache or reconfigure."

        # Down-project to hidden (H). Keep kernel contract: H % head_size_a == 0
        x = self.down(z_mix)                     # [B, T_orig, H]
        x = self._pre_core_cast(x)               # cast activations for kernel

        # Pad T up to CHUNK_LEN=16 for the fused kernel, then run the core
        x, T0 = pad_to_chunk(x, 16)              # T0 = T_orig
        if self.cfg.enforce_bf16 and x.is_cuda:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h = self.core(x)                 # [B, T_pad, H]
        else:
            h = self.core(x)

        # Up-project back to feature dim C and trim to original time
        h = self.up(h)                           # [B, T_pad, C]
        h = h[:, :T0, :].contiguous()            # [B, T0, C]

        if not self.use_mask:
            raise RuntimeError("use_mask=False is not supported in the STFT-mask smoke test.")

        # Produce 2*C logits -> [B,T0,C,2] -> softmax over source axis
        logits = self.head_m(h).contiguous()     # [B, T0, 2*C]
        logits = logits.view(B, T0, C, 2)        # use T0 (pre-pad length), not T_pad
        m = torch.softmax(logits, dim=-1)        # [B, T0, C, 2]
        m1, m2 = m[..., 0], m[..., 1]            # each [B, T0, C]

        return {"mask1": m1, "mask2": m2}

# ----------------------- Smoke test -----------------------
if __name__ == "__main__":
    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")

    B, T, C = 2, 320, 513
    cfg = SeparatorV7Config(in_dim=C, layers=4, head_size_a=64, dir_drop_p=0.0, use_mask=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RWKVv7Separator(cfg).to(device)
    x = torch.randn(B, T, C, device=device)
    out = model(x)
    for k, v in out.items():
        print(k, tuple(v.shape), v.dtype)
