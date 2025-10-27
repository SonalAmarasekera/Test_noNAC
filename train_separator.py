# train_separator.py
from pathlib import Path
import os, math, json, random, time
from dataclasses import dataclass, asdict
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# --- RWKV v7 CUDA settings (must be set before importing RWKV model) ---
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")
# -----------------------------------------------------------------------

from dataset_latents import RWKVLatentDataset
from collate_latents import collate_rwkv_latents
from rwkv_separator_v7_bi import RWKVv7Separator, SeparatorV7Config
from pit_losses import total_separator_loss

# ----------------- Hyperparameters -----------------
@dataclass
class HParams:
    # data
    train_root: str = "/workspace/latents/min/train"
    val_root:   str = "/workspace/latents/min/dev"
    batch_size: int = 8
    num_workers: int = 2
    pin_memory: bool = True
    # segmenting
    seg_seconds: float = 6.0
    latent_fps: float = 50.0
    # model
    layers: int = 6
    head_size_a: int = 64
    hidden_dim: int | None = None
    dir_drop_p: float = 0.5
    use_mask: bool = True
    enforce_bf16: bool = True
    # optimization
    epochs: int = 40
    lr: float = 8e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 5.0
    warmup_steps: int = 2000
    cosine_min_lr_ratio: float = 0.1
    # regularizers
    lambda_residual_l2: float = 1e-4
    lambda_mask_entropy: float = 0.0
    # embedding-loss (EL) knobs
    el_mode: str = "none"        # options: "none", "latent", or "decoder"
    lambda_el: float = 0.0       # "0.1-0.3" for <latent and decoder>, "0.0" for <none>
    el_cosine: bool = True       # cosine vs L2
    el_emb_dim: int = 128        
    # misc
    seed: int = 123
    log_every: int = 100
    val_every_steps: int = 1000
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

def crop_batch_to_seconds(batch, seg_seconds, fps, chunk_len=None):
    """Crop batch to fixed seconds using latent fps; preserves shapes and mask."""
    z_mix, z_s1, z_s2, mask = batch["z_mix"], batch.get("z_s1"), batch.get("z_s2"), batch["mask"]
    B, T, C = z_mix.shape
    frames = int(seg_seconds * float(fps))
    if chunk_len and frames % chunk_len != 0:
        frames = ((frames + chunk_len - 1) // chunk_len) * chunk_len
    if T <= frames:
        return batch
    starts = torch.randint(low=0, high=T - frames + 1, size=(B,), device=z_mix.device)
    idx = torch.arange(frames, device=z_mix.device)[None, :] + starts[:, None]
    idx = idx.clamp(0, T - 1)
    for k in ("z_mix", "z_s1", "z_s2"):
        if k in batch and batch[k] is not None:
            batch[k] = torch.gather(batch[k], 1, idx[..., None].expand(B, frames, C))
    batch["mask"] = torch.ones(B, frames, dtype=mask.dtype, device=mask.device)
    return batch

# ----------------- Teacher setup for EL -----------------
class SimpleLatentTeacher(nn.Module):
    """Light MLP teacher for latent-teacher mode."""
    def __init__(self, in_dim: int, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, z):  # z: [B,T,C]
        return self.net(z)

def setup_teachers(hp, cfg, device):
    """Return teacher handles depending on el_mode."""
    latent_teacher = None
    decode_fn = None
    embed_fn = None

    if hp.el_mode == "latent":
        latent_teacher = SimpleLatentTeacher(cfg.in_dim, hp.el_emb_dim).to(device)
        latent_teacher.eval()
        for p in latent_teacher.parameters():
            p.requires_grad = False

    elif hp.el_mode == "decoder":
        # These should point to your actual DAC decode/embed functions
        from descript_audio_codec import DAC  # example placeholder
        dac = DAC.load("descript-audio-codec-16khz").to(device)
        decode_fn = lambda z: dac.decode(z)
        embed_fn = lambda wav: dac.encode(wav)["z"]
        # freeze DAC
        for p in dac.parameters():
            p.requires_grad = False

    return latent_teacher, decode_fn, embed_fn

# ----------------- Main training -----------------
def main():
    set_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # probe channel dimension
    probe_ds = RWKVLatentDataset(hp.train_root, require_targets=True)
    probe = torch.load(next(p for p in probe_ds.mix_dir.rglob("*.pt")))
    C_meta = int(probe.get("C", 1024))

    # datasets
    train_ds = RWKVLatentDataset(hp.train_root, require_targets=True, expected_C=C_meta)
    val_ds = RWKVLatentDataset(hp.val_root, require_targets=True, expected_C=C_meta)
    collate_fn = partial(collate_rwkv_latents, chunk_len=None)

    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True,
                              num_workers=hp.num_workers, pin_memory=hp.pin_memory,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=hp.batch_size, shuffle=False,
                            num_workers=hp.num_workers, pin_memory=hp.pin_memory,
                            collate_fn=collate_fn, drop_last=False)

    # model
    sample = next(iter(train_loader))
    C = sample["z_mix"].shape[-1]
    cfg = SeparatorV7Config(
        in_dim=C,
        layers=hp.layers,
        head_size_a=hp.head_size_a,
        hidden_dim=hp.hidden_dim,
        dir_drop_p=hp.dir_drop_p,
        use_mask=hp.use_mask,
        enforce_bf16=hp.enforce_bf16
    )
    model = RWKVv7Separator(cfg).to(device)

    os.makedirs(hp.ckpt_dir, exist_ok=True)
    with open(Path(hp.ckpt_dir) / "hparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, betas=hp.betas, weight_decay=hp.weight_decay)
    ema = EMA(model, hp.ema_decay)

    # teacher setup
    latent_teacher, decode_fn, embed_fn = setup_teachers(hp, cfg, device)

    # Training
    global_step = 0
    steps_per_epoch = len(train_loader)
    total_steps = hp.epochs * steps_per_epoch
    scaler = None  # bf16 autocast path uses no scaler

    for epoch in range(hp.epochs):
        model.train()
        # running averages for the epoch
        run_loss = 0.0
        run_pit  = 0.0
        prog = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{hp.epochs}", leave=True)

        for step_in_epoch, batch in enumerate(prog):
            # Move to device & crop
            for k in ("z_mix","z_s1","z_s2","mask","sr"):
                if k in batch and torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device, non_blocking=True)

            # NOTE: if you’re padding to CHUNK_LEN inside the model, keep crop chunk_len=None here
            batch = crop_batch_to_seconds(batch, hp.seg_seconds, hp.latent_fps, chunk_len=None)

            z_mix, z_s1, z_s2, mask = batch["z_mix"], batch["z_s1"], batch["z_s2"], batch["mask"]
            B,T,C = z_mix.shape

            # forward
            if device.type == "cuda" and hp.enforce_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(z_mix)
                    loss, logs, perm = total_separator_loss(
                        out, z_s1, z_s2, mask,
                        lambda_residual_l2=hp.lambda_residual_l2,
                        lambda_mask_entropy=hp.lambda_mask_entropy
                    )
            else:
                out = model(z_mix)
                loss, logs, perm = total_separator_loss(
                    out, z_s1, z_s2, mask,
                    lambda_residual_l2=hp.lambda_residual_l2,
                    lambda_mask_entropy=hp.lambda_mask_entropy
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            opt.step()
            ema.update(model)

            # LR schedule (still by global step)
            lr_now = cosine_lr(global_step, total_steps, hp.lr, hp.cosine_min_lr_ratio)
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            # update running meters & progress bar every step
            run_loss += float(logs["loss/total"])
            run_pit  += float(logs["loss/pit_latent"])
            avg_loss = run_loss / (step_in_epoch + 1)
            avg_pit  = run_pit  / (step_in_epoch + 1)
            prog.set_postfix(lr=f"{lr_now:.2e}", loss=f"{avg_loss:.4f}", pit=f"{avg_pit:.4f}")

            global_step += 1

        # ----- end of epoch: print summary, run full validation, save ckpt -----
        epoch_train_loss = run_loss / max(1, steps_per_epoch)
        epoch_train_pit  = run_pit  / max(1, steps_per_epoch)
        print(f"[epoch {epoch+1}] train_loss {epoch_train_loss:.4f} | train_pit {epoch_train_pit:.4f}")

        # Standard validation
        val_loss = evaluate(model, val_loader, device)

        # EMA validation
        ema.store(model)
        ema.copy_to(model)
        val_loss_ema = evaluate(model, val_loader, device)
        model.load_state_dict(ema.backup)

        # Save epoch checkpoint
        ckpt_path = Path(hp.ckpt_dir) / f"epoch{epoch+1:03d}_val{val_loss:.4f}_valEMA{val_loss_ema:.4f}.pt"
        torch.save(
            {"epoch": epoch + 1, "step": global_step, "model": model.state_dict(),
             "ema": ema.shadow, "hp": asdict(hp)},
            ckpt_path
        )
        print(f"[ckpt] saved {ckpt_path}")

@torch.no_grad()
def evaluate(model, loader, device, latent_teacher, decode_fn, embed_fn):
    model.eval()
    losses = []
    for batch in loader:
        z_mix = batch["z_mix"].to(device)
        z_s1  = batch["z_s1"].to(device)
        z_s2  = batch["z_s2"].to(device)
        mask  = batch["mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type=="cuda")):
            out = model(z_mix)
            loss, logs, _ = total_separator_loss(
                out, z_s1, z_s2, mask,
                lambda_residual_l2=hp.lambda_residual_l2,
                lambda_mask_entropy=hp.lambda_mask_entropy,
                el_mode=hp.el_mode,
                lambda_el=hp.lambda_el,
                el_cosine=hp.el_cosine,
                latent_teacher=latent_teacher,
                decode_fn=decode_fn,
                embed_fn=embed_fn,
            )
        losses.append(float(logs["loss/total"]))
    return sum(losses) / max(1, len(losses))

if __name__ == "__main__":
    main()
