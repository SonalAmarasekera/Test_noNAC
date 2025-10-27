# dataset_latents.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset

HEAD_SIZE_A = 64  # same as your v7 head_size_a

def _to_tc(z, expected_C: int | None = None) -> torch.Tensor:
    t = z["z"] if isinstance(z, dict) else z
    t = torch.as_tensor(t)

    if t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)        # [C,T] or [T,C]
    if t.ndim != 2:
        raise ValueError(f"latent must be 2D [T,C] (or [1,T,C] / [C,T]), got {tuple(t.shape)}")

    d0, d1 = int(t.shape[0]), int(t.shape[1])

    # If caller tells us the true C, honor it.
    if expected_C is not None:
        if d0 == expected_C:        # [C,T] -> [T,C]
            t = t.transpose(0, 1)
        elif d1 == expected_C:      # [T,C] -> ok
            pass
        else:
            # If ambiguous (both multiples of 64), choose the axis equal to expected_C
            # Otherwise, fall back to "smaller=channels" but warn for re-caching.
            if d0 % 64 == 0 and d1 % 64 == 0:
                # neither equals expected_C -> likely a wrongly cached file
                # keep as-is and let the loader drop it later
                pass
            else:
                if d0 <= d1:
                    t = t.transpose(0, 1)
    else: # When expected_C = None
        if d1 % 64 == 0 and d1 <= 2048:
            pass # Assume [T, C]
        elif d0 % 64 and d0 <= 2048:
            t = t.transpose(0, 1) # Assume [C, T] -> [T, C]
        else:
            if d0 <= d1:
                t = t.transpose(0, 1)
    return t.contiguous().to(torch.float32)
    
class RWKVLatentDataset(Dataset):
    def __init__(self, root, require_targets=True, extensions=(".pt",), expected_C: int | None = None):
        self.root = Path(root)
        self.mix_dir = self.root / "mix_clean"
        self.s1_dir  = self.root / "s1"
        self.s2_dir  = self.root / "s2"
        self.require_targets = require_targets
        self.extensions = extensions
        self.expected_C = expected_C

        assert self.mix_dir.is_dir(), f"Missing {self.mix_dir}"
        if require_targets:
            assert self.s1_dir.is_dir() and self.s2_dir.is_dir(), "Missing s1/ or s2/"

        # Build list of utterances by intersecting file stems
        mix_files = [p for p in self.mix_dir.rglob("*") if p.suffix in extensions]
        if require_targets:
            s1_set = {p.relative_to(self.s1_dir).with_suffix("").as_posix() for p in self.s1_dir.rglob("*") if p.suffix in extensions}
            s2_set = {p.relative_to(self.s2_dir).with_suffix("").as_posix() for p in self.s2_dir.rglob("*") if p.suffix in extensions}

        items = []
        for m in mix_files:
            rel = m.relative_to(self.mix_dir).with_suffix("")  # preserve subdirs
            if require_targets:
                key = rel.as_posix()
                if key not in s1_set or key not in s2_set:
                    continue
            items.append(rel)

        assert len(items) > 0, f"No items found under {self.root}"
        self.items = sorted(items)

        # After: self.items = sorted(items)
        # Enforce a single latent channel count across the dataset when expected_C is given.
        if self.expected_C is not None:
            kept = []
            dropped = 0
            for rel in self.items:
                mix_p = self.mix_dir / (str(rel) + ".pt")
                obj = torch.load(mix_p, map_location="cpu")
                z = _to_tc(obj, expected_C=self.expected_C)  # don’t force transpose yet, just read shape
                C_here = z.shape[1]
                if C_here == self.expected_C:
                    kept.append(rel)
                else:
                    dropped += 1
            if dropped > 0:
                print(f"[dataset] filtered out {dropped} files with C != {self.expected_C}")
            self.items = kept
        assert len(self.items) > 0, f"No items with C={self.expected_C} under {self.root}"

    def __len__(self) -> int:
        return len(self.items)

    def _load_payload(self, p: Path) -> Dict:
        obj = torch.load(p, map_location="cpu")
        # Required keys
        assert "z" in obj, f"Missing 'z' in {p}"
        return obj

    def __getitem__(self, idx: int) -> Dict:
        rel = self.items[idx]
        mix_p = self.mix_dir / (str(rel) + ".pt")
        mix_obj = self._load_payload(mix_p)
        zmix = _to_tc(mix_obj, expected_C=self.expected_C)

        ex: Dict = {
            "utt_id": str(rel),
            "z_mix": zmix,                         # [T,C]
            "fps": int(mix_obj.get("fps", 320)),
            "sr":  int(mix_obj.get("sr", 16000)),
        }

        if self.require_targets:
            s1_p = self.s1_dir / (str(rel) + ".pt")
            s2_p = self.s2_dir / (str(rel) + ".pt")
            s1_obj = self._load_payload(s1_p)
            s2_obj = self._load_payload(s2_p)
            ex["z_s1"] = _to_tc(s1_obj, expected_C=self.expected_C)
            ex["z_s2"] = _to_tc(s2_obj, expected_C=self.expected_C)
        return ex
