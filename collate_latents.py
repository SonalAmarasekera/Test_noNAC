# collate_latents.py
from typing import List, Dict, Optional
import torch

def collate_rwkv_latents(
    batch: List[Dict],
    pad_value: float = 0.0,
    chunk_len: Optional[int] = None,
) -> Dict[str, torch.Tensor]:

    # Ensure consistent channel dim after dataset normalization
    Cs = [ex["z_mix"].shape[1] for ex in batch]
    if len(set(Cs)) != 1:
        raise RuntimeError(
            f"Inconsistent channel dims in batch: {Cs}. "
            f"Likely some latents were cached with a different DAC/config. Re-cache to a single C."
        )

    B = len(batch)
    C = Cs[0]
    T_max = max(x["z_mix"].shape[0] for x in batch)

    if chunk_len is not None and chunk_len > 0 and (T_max % chunk_len != 0):
        T_max = ((T_max + chunk_len - 1) // chunk_len) * chunk_len

    out_dtype = batch[0]["z_mix"].dtype
    device = batch[0]["z_mix"].device

    z_mix = torch.full((B, T_max, C), pad_value, dtype=out_dtype, device=device)
    mask  = torch.zeros((B, T_max), dtype=torch.bool, device=device)

    has_targets = "z_s1" in batch[0]
    if has_targets:
        z_s1 = torch.full((B, T_max, C), pad_value, dtype=out_dtype, device=device)
        z_s2 = torch.full((B, T_max, C), pad_value, dtype=out_dtype, device=device)

    for i, ex in enumerate(batch):
        zi = ex["z_mix"]           # [Ti, C] guaranteed by dataset
        Ti = zi.shape[0]
        z_mix[i, :Ti] = zi
        mask[i, :Ti]  = True
        if has_targets:
            z_s1[i, :Ti] = ex["z_s1"]
            z_s2[i, :Ti] = ex["z_s2"]

    out = {"z_mix": z_mix, "mask": mask}
    if has_targets:
        out.update({"z_s1": z_s1, "z_s2": z_s2})
    return out
