# pit_losses.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, Callable
import torch
import torch.nn.functional as F

# ---------------------------
# Utilities
# ---------------------------

def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    x: [B, T, ...]
    mask: [B, T] or None
    returns mean over T (masked if provided), keeping batch & remaining dims
    """
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(dtype=x.dtype).unsqueeze(-1)  # [B,T,1]
    num = (x * m).sum(dim=1)
    den = m.sum(dim=1).clamp_min(1.0)
    return num / den

def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.float().norm(dim=-1, keepdim=True).clamp_min(eps))

def _cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a,b: [B, E] (assumed normalized)
    returns mean(1 - cosine_sim)
    """
    return (1.0 - (a * b).sum(dim=-1)).mean()

# ---------------------------
# Core losses
# ---------------------------

def _masked_mse(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Framewise MSE on [B,T,C] with optional [B,T] mask.
    Numerically stable in fp32, averaged over valid frames * channels.
    """
    diff = (a - b).to(torch.float32)  # [B,T,C]
    if mask is not None:
        m = mask.to(dtype=diff.dtype).unsqueeze(-1)  # [B,T,1]
        diff = diff * m
        denom = (m.sum() * diff.shape[-1]).clamp_min(1.0)
        return diff.square().sum() / denom
    else:
        return diff.square().mean()

def pit_latent_mse_2spk(
    preds: Dict[str, torch.Tensor],
    tgt1: torch.Tensor,
    tgt2: torch.Tensor,
    pad_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Tuple[int, int], Dict[str, torch.Tensor]]:
    """PIT over two sources in latent space."""
    y1, y2 = preds["pred1"], preds["pred2"]
    L_12 = _masked_mse(y1, tgt1, pad_mask) + _masked_mse(y2, tgt2, pad_mask)
    L_21 = _masked_mse(y1, tgt2, pad_mask) + _masked_mse(y2, tgt1, pad_mask)
    perm = (0, 1) if L_12 <= L_21 else (1, 0)
    pit = torch.minimum(L_12, L_21)
    extras = {"L_1122": L_12.detach(), "L_1221": L_21.detach()}
    return pit, perm, extras

def _reorder_two(y1: torch.Tensor, y2: torch.Tensor, perm: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    return (y1, y2) if perm == (0, 1) else (y2, y1)

# ---------------------------
# Embedding Space Loss (EL)
# ---------------------------

@torch.no_grad()
def _maybe_decode_wav(decode_fn: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor) -> torch.Tensor:
    """
    decode_fn: callable mapping latents [B,T,C] -> waveform [B,L] (or [B,1,L])
    """
    wav = decode_fn(z)
    if wav.dim() == 3 and wav.size(1) == 1:  # [B,1,L] -> [B,L]
        wav = wav[:, 0, :]
    return wav

def _compute_embeddings(
    el_mode: str,
    y: torch.Tensor,          # [B,T,C] predicted (after PIT reordering)
    z: torch.Tensor,          # [B,T,C] target    (after PIT reordering)
    pad_mask: Optional[torch.Tensor],
    *,
    # latent-teacher
    latent_teacher: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    # decoder-teacher
    decode_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    embed_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (e_pred, e_tgt) embeddings with shape [B,E] (pooled over time).
    - el_mode == 'latent': latent_teacher maps [B,T,C] -> [B,T,E] or [B,E]
    - el_mode == 'decoder': decode_fn maps [B,T,C] -> [B,L], embed_fn maps [B,L] -> [B,E] or [B,T',E]
    """
    if el_mode == "latent":
        if latent_teacher is None:
            # Fallback: identity projection on latents (then pool). It's weaker than a trained teacher but still valid.
            e_pred = _masked_mean(y, pad_mask)  # [B,C]
            e_tgt  = _masked_mean(z, pad_mask)  # [B,C]
            return e_pred, e_tgt
        p = latent_teacher(y)
        t = latent_teacher(z)
        # Accept either [B,E] or [B,T,E]-like; pool over time if needed
        if p.dim() == 3:
            p = _masked_mean(p, pad_mask)  # [B,E]
        if t.dim() == 3:
            t = _masked_mean(t, pad_mask)
        return p, t

    elif el_mode == "decoder":
        assert decode_fn is not None and embed_fn is not None, \
            "decoder mode requires both decode_fn (latents->wav) and embed_fn (wav->emb)"
        with torch.no_grad():
            wav_p = _maybe_decode_wav(decode_fn, y)
            wav_t = _maybe_decode_wav(decode_fn, z)
        p = embed_fn(wav_p)  # [B,E] or [B,T',E]
        t = embed_fn(wav_t)
        if p.dim() == 3:
            # No direct pad_mask for wav frames; use simple mean
            p = p.mean(dim=1)
        if t.dim() == 3:
            t = t.mean(dim=1)
        return p, t

    else:
        # 'none' -> no embeddings
        return y.new_zeros(y.size(0), 1), z.new_zeros(z.size(0), 1)

def _embedding_space_loss(
    e_pred: torch.Tensor, e_tgt: torch.Tensor, cosine: bool = True
) -> torch.Tensor:
    """
    Compute EL on [B,E]; use cosine by default, L2 otherwise.
    """
    e_pred = e_pred.float()
    e_tgt  = e_tgt.float()
    if cosine:
        e_pred = _l2_normalize(e_pred)
        e_tgt  = _l2_normalize(e_tgt)
        return _cosine_loss(e_pred, e_tgt)
    else:
        return F.mse_loss(e_pred, e_tgt)

# ---------------------------
# Public API
# ---------------------------

def total_separator_loss(
    preds: Dict[str, torch.Tensor],
    z_s1: torch.Tensor,
    z_s2: torch.Tensor,
    pad_mask: Optional[torch.Tensor],
    *,
    lambda_residual_l2: float = 1e-4,
    lambda_mask_entropy: float = 0.0,
    # ---- Embedding Loss (CodecFormer-EL) ----
    el_mode: str = "none",                 # 'none' | 'latent' | 'decoder'
    lambda_el: float = 0.0,                # weight for embedding loss
    latent_teacher: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    decode_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    embed_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    el_cosine: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Tuple[int, int]]:
    """
    Combined loss = PIT latent MSE (+ optional L2 on residuals, mask entropy)
                    + lambda_el * Embedding-Loss (latent- or decoder-teacher).
    """
    # --- PIT in latent space ---
    pit_loss, perm, extras = pit_latent_mse_2spk(preds, z_s1, z_s2, pad_mask)

    # Reorder predictions & targets by PIT for any auxiliary losses
    y1, y2 = preds["pred1"], preds["pred2"]
    y1, y2 = _reorder_two(y1, y2, perm)
    t1, t2 = _reorder_two(z_s1, z_s2, perm)

    # --- Regularizers ---
    reg_loss = 0.0
    for key in ("resid1", "resid2"):
        r = preds.get(key, None)
        if r is not None:
            r = r.float()
            if pad_mask is not None:
                m = pad_mask.to(dtype=r.dtype).unsqueeze(-1)
                reg_loss = reg_loss + (r.square() * m).sum() / m.sum().clamp_min(1.0)
            else:
                reg_loss = reg_loss + r.square().mean()
    reg_loss = reg_loss * float(lambda_residual_l2)

    ent_loss = 0.0
    if lambda_mask_entropy > 0.0 and ("mask1" in preds and "mask2" in preds):
        for key in ("mask1", "mask2"):
            m = preds[key].clamp(1e-6, 1 - 1e-6).float()
            H = -(m * torch.log(m) + (1 - m) * torch.log(1 - m))  # [B,T,C]
            if pad_mask is not None:
                pm = pad_mask.to(dtype=H.dtype).unsqueeze(-1)
                ent_loss = ent_loss + (H * pm).sum() / pm.sum().clamp_min(1.0)
            else:
                ent_loss = ent_loss + H.mean()
        ent_loss = ent_loss * float(lambda_mask_entropy)

    # --- Embedding Loss (CodecFormer-EL style) ---
    el_loss = torch.tensor(0.0, device=y1.device)
    if el_mode != "none" and lambda_el > 0.0:
        # Compute embeddings for each source, then average losses
        e_p1, e_t1 = _compute_embeddings(
            el_mode, y1, t1, pad_mask,
            latent_teacher=latent_teacher,
            decode_fn=decode_fn, embed_fn=embed_fn,
        )
        e_p2, e_t2 = _compute_embeddings(
            el_mode, y2, t2, pad_mask,
            latent_teacher=latent_teacher,
            decode_fn=decode_fn, embed_fn=embed_fn,
        )
        el1 = _embedding_space_loss(e_p1, e_t1, cosine=el_cosine)
        el2 = _embedding_space_loss(e_p2, e_t2, cosine=el_cosine)
        el_loss = 0.5 * (el1 + el2)

    total = pit_loss + reg_loss + ent_loss + (lambda_el * el_loss)

    logs = {
        "loss/pit_latent": pit_loss.detach(),
        "loss/reg_residual": torch.as_tensor(reg_loss).detach(),
        "loss/mask_entropy": torch.as_tensor(ent_loss).detach(),
        "loss/EL": torch.as_tensor(el_loss).detach(),
        "loss/total": total.detach(),
        **extras,
    }
    return total, logs, perm
