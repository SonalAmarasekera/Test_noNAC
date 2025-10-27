#!/usr/bin/env python
"""
Encode WSJ0 source WAVs with the 16-kHz Descript Audio Codec
and save latent tensors (.pt) for later training.

Usage:
    python cache_latents.py --csv data/train.csv --out_dir data/latents/train
"""

import argparse
import csv
import math
import os
import pathlib
import sys
import multiprocessing as mp
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import soundfile as sf
import torch
import tqdm

import dac


# ---------------------------- Audio discovery (mirror mode) ----------------------------

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def find_audio_paths(root: pathlib.Path) -> List[pathlib.Path]:
    if root.is_file():
        return [root]
    files = []
    for p in root.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS and p.is_file():
            files.append(p)
    return files


# ---------------------------- Windowed encoding utility ----------------------------

def chunk_indices(num_samples: int, sr: int, win_seconds: float) -> List[Tuple[int, int]]:
    if win_seconds <= 0:
        return [(0, num_samples)]
    hop = int(sr * win_seconds)
    if hop <= 0:
        return [(0, num_samples)]
    idx = []
    start = 0
    while start < num_samples:
        end = min(num_samples, start + hop)
        idx.append((start, end))
        start = end
    return idx


# ---------------------------- Encoding core ----------------------------

def _encode_tensor(
    wav_f32: np.ndarray,
    sr: int,
    device: str,
    win_seconds: float,
    codec_kwargs: Dict
) -> torch.Tensor:
    """Return latent tensor z: [1, C, T_latent]."""
    assert sr in (16000, 24000, 44100), f"Unsupported sample rate {sr}; choose model_type to match."
    wav = torch.from_numpy(wav_f32).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,N]
    with torch.no_grad():
        if win_seconds <= 0:
            z, *_ = _CODEC.encode(wav, **codec_kwargs)
            return z
        # windowed: concatenate latents along time dimension
        N = wav.shape[-1]
        pieces = []
        for s, e in chunk_indices(N, sr, win_seconds):
            z_part, *_ = _CODEC.encode(wav[..., s:e], **codec_kwargs)
            pieces.append(z_part)
        z = torch.cat(pieces, dim=-1)
        return z

def _write_pt(save_path: pathlib.Path, z: torch.Tensor):
    """
    Always save latents as [T, C] float32 with reliable metadata.
    """
    # z is [1, C, T] from _CODEC.encode
    z = z.detach().cpu()
    assert z.ndim == 3 and z.shape[0] == 1, f"unexpected latent shape {tuple(z.shape)}"
    z = z.squeeze(0)             # [C, T]
    C, T = int(z.shape[0]), int(z.shape[1])
    z = z.transpose(0, 1).contiguous()  # → [T, C]

    meta = {
        "z": z,                         # [T, C] float32
        "C": C,
        "T": T,
        "sr": getattr(_CODEC, "sample_rate", 16000),
        "fps": 50,                      # set to your known latent fps
        "model_type": os.environ.get("DAC_MODEL_TYPE", "16khz"),
        "bitrate": None,                # fill from args if you use it
        "n_quantizers": None            # fill from args if you use it
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(meta, save_path)

# ---------------------------- Worker wrapper ----------------------------

def _worker_encode(args):
    """Worker-safe wrapper: (wav_path, save_path, device, expect_sr, win_seconds, codec_kwargs)."""
    wav_path, save_path, device, expect_sr, win_seconds, codec_kwargs = args
    if save_path.exists():
        return  # idempotent
    # read
    wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if wav.ndim == 2:  # mixdown if stereo
        wav = wav.mean(axis=1)
    if expect_sr is not None:
        assert sr == expect_sr, f"Expected {expect_sr}Hz, got {sr} @ {wav_path}"
    # encode
    z = _encode_tensor(wav, sr, device, win_seconds, codec_kwargs)
    _write_pt(save_path, z)


# ---------------------------- Job builders ----------------------------

def build_jobs_from_csv(
    csv_path: pathlib.Path,
    out_root: pathlib.Path,
    mirror_from: Optional[pathlib.Path],
    expect_sr: Optional[int],
    device: str,
    win_seconds: float,
    codec_kwargs: Dict
) -> List[Tuple[pathlib.Path, pathlib.Path, str, Optional[int], float, Dict]]:
    rows = list(csv.DictReader(open(csv_path)))
    # column → subdir mapping
    col2sub = {"mix_path": "mix_clean", "s1_path": "s1", "s2_path": "s2"}
    jobs = []
    for row in rows:
        for col, sub in col2sub.items():
            p = row.get(col, "")
            if not p:
                continue
            wav_path = pathlib.Path(p)
            # mirrored relative path (if requested)
            if mirror_from is not None:
                rel = pathlib.Path(os.path.relpath(str(wav_path), str(mirror_from)))
                save_path = out_root / sub / rel.with_suffix(".pt")
            else:
                save_path = out_root / sub / (wav_path.stem + ".pt")
            jobs.append((wav_path, save_path, device, expect_sr, win_seconds, codec_kwargs))
    return jobs


def build_jobs_from_mirror(
    in_path: pathlib.Path,
    out_root: pathlib.Path,
    expect_sr: Optional[int],
    device: str,
    win_seconds: float,
    codec_kwargs: Dict
) -> List[Tuple[pathlib.Path, pathlib.Path, str, Optional[int], float, Dict]]:
    in_path = in_path.resolve()
    files = find_audio_paths(in_path)
    jobs = []
    for f in files:
        rel = f.relative_to(in_path) if in_path.is_dir() else pathlib.Path(f.name)
        save_path = (out_root / rel).with_suffix(".pt")
        jobs.append((f, save_path, device, expect_sr, win_seconds, codec_kwargs))
    return jobs


# ---------------------------- CLI and main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    # Input selection
    g_in = ap.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--csv", type=str, help="CSV with mix_path,s1_path,s2_path")
    g_in.add_argument("--in_path", type=str, help="File or directory (mirror mode)")
    ap.add_argument("--out_dir", required=True, type=str, help="Root folder to write latents")

    # Mirroring options (CSV mode only)
    ap.add_argument("--mirror_from", type=str, default=None,
                    help="Root folder to preserve substructure relative to this path (CSV mode)")

    # Performance mode
    ap.add_argument("--fast_mp", action="store_true", help="Use multiprocessing (faster, higher VRAM)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))

    # Codec/model options (like encode.py)
    ap.add_argument("--model_type", type=str, default="16khz", choices=["16khz", "24khz", "44khz"])
    ap.add_argument("--bitrate", type=str, default=None, choices=[None, "8kbps", "16kbps"])
    ap.add_argument("--n_quantizers", type=int, default=None, help="Override number of residual quantizers if supported")
    ap.add_argument("--weights_path", type=str, default=None, help="Path to custom model weights")
    ap.add_argument("--model_tag", type=str, default=None, help="Alternative tag identifying model weights")

    # Device & memory
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'. Default: auto")
    ap.add_argument("--win_seconds", type=float, default=0.0,
                    help="Encode in windows of N seconds (0 = full file). Helps on low VRAM.")

    args = ap.parse_args()

    out_root = pathlib.Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Expected SR per model_type
    expect_sr = {"16khz": 16000, "24khz": 24000, "44khz": 44100}[args.model_type]

    # Build codec kwargs for encode()
    codec_kwargs: Dict = {}
    if args.n_quantizers is not None:
        # Some DAC builds accept n_quantizers in encode(); we pass through if supported.
        codec_kwargs["n_quantizers"] = int(args.n_quantizers)
    if args.bitrate is not None:
        # Some builds accept bitrate string; forward if supported.
        codec_kwargs["bitrate"] = args.bitrate

    # Jobs
    if args.csv:
        mirror_from = pathlib.Path(args.mirror_from).resolve() if args.mirror_from else None
        jobs = build_jobs_from_csv(
            pathlib.Path(args.csv),
            out_root,
            mirror_from,
            expect_sr,
            device,
            args.win_seconds,
            codec_kwargs,
        )
    else:
        jobs = build_jobs_from_mirror(
            pathlib.Path(args.in_path),
            out_root,
            expect_sr,
            device,
            args.win_seconds,
            codec_kwargs,
        )

    print(f"[cache_latents] model_type={args.model_type} device={device} "
          f"win_seconds={args.win_seconds} fast_mp={args.fast_mp}")
    print(f"[cache_latents] jobs: {len(jobs)}  →  out_dir={out_root}")

    if not jobs:
        return

    # Run
    if args.fast_mp:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            list(tqdm.tqdm(pool.imap(_worker_encode, jobs), total=len(jobs)))
    else:
        for job in tqdm.tqdm(jobs, total=len(jobs)):
            _worker_encode(job)


# ---------------------------- Global codec load ----------------------------

def _load_codec(model_type: str, device: str, weights_path: Optional[str], model_tag: Optional[str]):
    """
    Load DAC model. We prefer built-in downloads; fall back to tag/weights if provided.
    """
    # Default path: official downloads for 16/24/44 kHz
    try:
        model = dac.DAC.load(dac.utils.download(model_type))
    except Exception:
        # Optional alternate pathways
        if weights_path is not None:
            model = dac.DAC.load(weights_path)
        elif model_tag is not None:
            model = dac.DAC.load(dac.utils.download(model_tag))
        else:
            raise
    return model.eval().to(device)


# Load once at module import so workers reuse it
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_TYPE_ENV = os.environ.get("DAC_MODEL_TYPE", "16khz")
_CODEC = _load_codec(_MODEL_TYPE_ENV, _DEVICE, weights_path=None, model_tag=None)
torch.set_grad_enabled(False)


# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow overriding global preload using CLI by reloading after parse.
    # We re-enter main which will use the preloaded _CODEC; to honor CLI model_type,
    # reload here before dispatch if it differs.
    mp.set_start_method("spawn", force=True)
    # Peek CLI to decide if we need to reload codec with another model_type/device.
    # We do a light parse to avoid duplicating argparse logic.
    try:
        idx = sys.argv.index("--model_type")
        requested_type = sys.argv[idx + 1]
    except ValueError:
        requested_type = _MODEL_TYPE_ENV
    try:
        didx = sys.argv.index("--device")
        requested_device = sys.argv[didx + 1]
    except ValueError:
        requested_device = _DEVICE
    if (requested_type != _MODEL_TYPE_ENV) or (requested_device != _DEVICE):
        # reload to match user request
        _CODEC = _load_codec(requested_type, requested_device, None, None)
    main()
