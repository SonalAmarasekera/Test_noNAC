# train_separator.py (STFT-mask smoke test)
from pathlib import Path
import os, math, json, random
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# --- RWKV v7 CUDA settings (must be set before importing RWKV model) ---
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")
# -----------------------------------------------------------------------

from rwkv_separator_v7_bi import RWKVv7Separator, SeparatorV7Config

# ----------------- Hyperparameters -----------------
@dataclass
class HParams:
    # data (expects WSJ0-2mix/WHAM!-style layout with subdirs: mix|mix_both|mix_clean, s1, s2)
    train_root: str = "/content/2speakers/wav16k/min/tr"
    val_root:   str = "/content/2speakers/wav16k/min/cv"
    sample_rate: int = 16000
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    # segmenting
    seg_seconds: float = 6.0
    # model
    layers: int = 6
    head_size_a: int = 64
    hidden_dim: int | None = None
    dir_drop_p: float = 0.0      # disable for smoke test
    use_mask: bool = True        # mask-only path in the separator
    enforce_bf16: bool = True
    # optimization
    epochs: int = 40
    lr: float = 8e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 5.0
    warmup_steps: int = 2000
    cosine_min_lr_ratio: float = 0.1
    # misc
    seed: int = 123
    ckpt_dir: str = "checkpoints"
    ema_decay: float = 0.999

hp = HParams()

# ----------------- Utility helpers -----------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def store(self, model): self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def copy_to(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow: v.copy_(self.shadow[k])

def cosine_lr(step, total, base_lr, min_ratio=0.1):
    if step < hp.warmup_steps:
        return base_lr * (step + 1) / hp.warmup_steps
    t = (step - hp.warmup_steps) / max(1, total - hp.warmup_steps)
    return base_lr * (min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * (1 - t))))

# ----------------- STFT helpers (fixed) ------------------
STFT_CFG = dict(n_fft=1024, hop_length=256, win_length=1024)  # -> F = 513

def _hann(device, win_length=1024):
    return torch.hann_window(win_length, periodic=True, device=device)

def stft_encode(wav, n_fft=1024, hop_length=256, win_length=1024):
    """
    wav: [B, T_wav] float32
    returns:
      mag_btF: [B, T_frames, F]
      X_bFT:   [B, F, T_frames] complex (for reconstruction)
    """
    window = _hann(wav.device, win_length)
    X = torch.stft(
        wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, return_complex=True, center=True, pad_mode="reflect"
    )  # [B, F, T] complex
    mag = X.abs()                   # FIX: call .abs()
    mag_btF = mag.transpose(1, 2)   # -> [B, T, F] for RWKV
    return mag_btF.contiguous(), X

def istft_decode(Y_bFT, n_fft=1024, hop_length=256, win_length=1024, length=None):
    window = _hann(Y_bFT.device, win_length)  # FIX: use Y_bFT.device
    wav_hat = torch.istft(
        Y_bFT, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, length=length
    )
    return wav_hat

# ----------- Loss Function (Smoke Test) ----------
def sisdr(x, s, eps=1e-8):
    # x: estimate [B, T], s: target [B, T]
    dot = torch.sum(x * s, dim=1, keepdim=True)
    s_energy = torch.sum(s * s, dim=1, keepdim=True) + eps
    s_target = (dot / s_energy) * s
    e_noise = x - s_target
    num = torch.sum(s_target ** 2, dim=1) + eps
    den = torch.sum(e_noise ** 2, dim=1) + eps
    return 10 * torch.log10(num / den)

def pit_sisdr(yhat1, yhat2, y1, y2):
    sdr_12 = sisdr(yhat1, y1) + sisdr(yhat2, y2)
    sdr_21 = sisdr(yhat1, y2) + sisdr(yhat2, y1)
    sdr = torch.stack([sdr_12, sdr_21], dim=1)  # [B,2]
    best, idx = torch.max(sdr, dim=1)           # maximize SI-SDR
    loss = -best.mean()
    return loss, idx

# ----------------- Simple waveform dataset -----------------
# Expects:
#   <root>/
#     mix|mix_both|mix_clean/*.wav
#     s1/*.wav
#     s2/*.wav
# File basenames must match across dirs.
try:
    import torchaudio
    from torchaudio.functional import resample as ta_resample
    HAVE_TORCHAUDIO = True
except Exception:
    HAVE_TORCHAUDIO = False
    import soundfile as sf

class WaveMixDataset(Dataset):
    def __init__(self, root: str, seg_seconds: float, sample_rate: int = 16000):
        super().__init__()
        self.root = Path(root)
        self.sr = sample_rate
        self.seg_samples = int(seg_seconds * sample_rate)

        mix_dir = None
        for d in ["mix", "mix_both", "mix_clean"]:
            cand = self.root / d
            if cand.exists():
                mix_dir = cand
                break
        if mix_dir is None:
            raise FileNotFoundError(f"No mix dir found under {root} (tried mix|mix_both|mix_clean)")

        s1_dir = self.root / "s1"
        s2_dir = self.root / "s2"
        if not s1_dir.exists() or not s2_dir.exists():
            raise FileNotFoundError(f"Expected s1/s2 under {root}")

        mix_files = sorted(mix_dir.rglob("*.wav"))
        pairs = []
        for m in mix_files:
            stem = m.name
            s1 = s1_dir / stem
            s2 = s2_dir / stem
            if s1.exists() and s2.exists():
                pairs.append((m, s1, s2))
        if not pairs:
            raise RuntimeError(f"No paired files found under {root}")
        self.items = pairs

    def __len__(self): return len(self.items)

    def _load_wav(self, path: Path):
        if HAVE_TORCHAUDIO:
            wav, sr = torchaudio.load(str(path))  # [C, T]
            if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
            if sr != self.sr:
                wav = ta_resample(wav, sr, self.sr)
            wav = wav.squeeze(0)
            return wav
        else:
            wav, sr = sf.read(str(path), dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != self.sr:
                # light-weight resample via linear interpolation
                import numpy as np
                t_old = np.linspace(0, len(wav)/sr, num=len(wav), endpoint=False)
                t_new = np.linspace(0, len(wav)/sr, num=int(len(wav)*self.sr/sr), endpoint=False)
                wav = np.interp(t_new, t_old, wav).astype("float32")
            return torch.from_numpy(wav.copy())

    def __getitem__(self, idx: int):
        mix_p, s1_p, s2_p = self.items[idx]
        mix = self._load_wav(mix_p)  # [Tw]
        s1  = self._load_wav(s1_p)
        s2  = self._load_wav(s2_p)

        # Align & random crop to seg_seconds
        Tw = min(len(mix), len(s1), len(s2))
        mix, s1, s2 = mix[:Tw], s1[:Tw], s2[:Tw]
        if Tw > self.seg_samples:
            start = random.randint(0, Tw - self.seg_samples)
            end = start + self.seg_samples
            mix, s1, s2 = mix[start:end], s1[start:end], s2[start:end]
        elif Tw < self.seg_samples:
            pad = self.seg_samples - Tw
            mix = F.pad(mix, (0, pad))
            s1  = F.pad(s1,  (0, pad))
            s2  = F.pad(s2,  (0, pad))

        return {"wav_mix": mix, "wav_s1": s1, "wav_s2": s2}

def collate_waves(batch):
    # tensors are already cropped/padded to same length by dataset
    wav_mix = torch.stack([b["wav_mix"] for b in batch], dim=0)  # [B, Tw]
    wav_s1  = torch.stack([b["wav_s1"]  for b in batch], dim=0)
    wav_s2  = torch.stack([b["wav_s2"]  for b in batch], dim=0)
    return {"wav_mix": wav_mix, "wav_s1": wav_s1, "wav_s2": wav_s2}

# ----------------- Main training (SMOKE TEST) -----------------
def main():
    set_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # datasets: load waveforms, not latents
    train_ds = WaveMixDataset(hp.train_root, seg_seconds=hp.seg_seconds, sample_rate=hp.sample_rate)
    val_ds   = WaveMixDataset(hp.val_root,   seg_seconds=hp.seg_seconds, sample_rate=hp.sample_rate)

    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True,
                              num_workers=hp.num_workers, pin_memory=hp.pin_memory,
                              collate_fn=collate_waves, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=hp.batch_size, shuffle=False,
                              num_workers=hp.num_workers, pin_memory=hp.pin_memory,
                              collate_fn=collate_waves, drop_last=False)

    # model: STFT mag has 513 bins (n_fft=1024)
    cfg = SeparatorV7Config(
        in_dim=STFT_CFG["n_fft"] // 2 + 1,  # 513
        layers=hp.layers,
        head_size_a=hp.head_size_a,
        hidden_dim=hp.hidden_dim,
        dir_drop_p=hp.dir_drop_p,           # keep 0.0 for the smoke test
        use_mask=hp.use_mask,               # mask-only head
        enforce_bf16=hp.enforce_bf16
    )
    model = RWKVv7Separator(cfg).to(device)

    os.makedirs(hp.ckpt_dir, exist_ok=True)
    with open(Path(hp.ckpt_dir) / "hparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, betas=hp.betas, weight_decay=hp.weight_decay)
    ema = EMA(model, hp.ema_decay)

    # Training
    global_step = 0
    steps_per_epoch = len(train_loader)
    total_steps = hp.epochs * steps_per_epoch

    for epoch in range(hp.epochs):
        model.train()
        run_loss = 0.0
        prog = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{hp.epochs}", leave=True)

        for step_in_epoch, batch in enumerate(prog):
            wav_mix = batch["wav_mix"].to(device)
            wav_s1  = batch["wav_s1"].to(device)
            wav_s2  = batch["wav_s2"].to(device)

            # Front-end STFT
            mag_mix_btF, X_bFT = stft_encode(wav_mix, **STFT_CFG)    # [B,T,F], [B,F,T]

            # RWKV forward on magnitudes
            if device.type == "cuda" and hp.enforce_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(mag_mix_btF)  # expects {"mask1","mask2"} in [B,T,F]
            else:
                out = model(mag_mix_btF)

            # Masks -> complex STFT -> iSTFT
            m1_btF = out["mask1"]  # [B,T,F]
            m2_btF = out["mask2"]  # [B,T,F]
            m1_bFT = m1_btF.transpose(1, 2).contiguous()
            m2_bFT = m2_btF.transpose(1, 2).contiguous()

            Y1_bFT = m1_bFT * X_bFT
            Y2_bFT = m2_bFT * X_bFT

            yhat1 = istft_decode(Y1_bFT, **STFT_CFG, length=wav_mix.shape[-1])  # [B, Tw]
            yhat2 = istft_decode(Y2_bFT, **STFT_CFG, length=wav_mix.shape[-1])

            # uPIT SI-SDR loss
            loss, _ = pit_sisdr(yhat1, yhat2, wav_s1, wav_s2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            opt.step()
            ema.update(model)

            # LR schedule (cosine by global step)
            lr_now = cosine_lr(global_step, total_steps, hp.lr, hp.cosine_min_lr_ratio)
            for pg in opt.param_groups: pg["lr"] = lr_now

            run_loss += float(loss.item())
            avg_loss = run_loss / (step_in_epoch + 1)
            prog.set_postfix(lr=f"{lr_now:.2e}", sisdr_loss=f"{avg_loss:.4f}")

            global_step += 1

        # ----- end of epoch: print summary, run full validation, save ckpt -----
        epoch_train_loss = run_loss / max(1, steps_per_epoch)
        print(f"[epoch {epoch+1}] train_sisdr_loss {epoch_train_loss:.4f}")

        # Standard validation (EMA & non-EMA)
        val_loss = evaluate(model, val_loader, device)

        ema.store(model); ema.copy_to(model)
        val_loss_ema = evaluate(model, val_loader, device)
        model.load_state_dict(ema.backup)

        ckpt_path = Path(hp.ckpt_dir) / f"epoch{epoch+1:03d}_val{val_loss:.4f}_valEMA{val_loss_ema:.4f}.pt"
        torch.save({"epoch": epoch + 1, "step": global_step, "model": model.state_dict(),
                    "ema": ema.shadow, "hp": asdict(hp)}, ckpt_path)
        print(f"[ckpt] saved {ckpt_path}")

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        wav_mix = batch["wav_mix"].to(device)
        wav_s1  = batch["wav_s1"].to(device)
        wav_s2  = batch["wav_s2"].to(device)

        mag_mix_btF, X_bFT = stft_encode(wav_mix, **STFT_CFG)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type=="cuda")):
            out = model(mag_mix_btF)
            m1_btF = out["mask1"]; m2_btF = out["mask2"]
            m1_bFT = m1_btF.transpose(1, 2).contiguous()
            m2_bFT = m2_btF.transpose(1, 2).contiguous()

            Y1_bFT = m1_bFT * X_bFT
            Y2_bFT = m2_bFT * X_bFT

            yhat1 = istft_decode(Y1_bFT, **STFT_CFG, length=wav_mix.shape[-1])
            yhat2 = istft_decode(Y2_bFT, **STFT_CFG, length=wav_mix.shape[-1])

            loss, _ = pit_sisdr(yhat1, yhat2, wav_s1, wav_s2)
        losses.append(float(loss.item()))
    return sum(losses) / max(1, len(losses))

if __name__ == "__main__":
    main()
