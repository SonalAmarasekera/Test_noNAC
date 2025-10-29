# separate_test.py
import os, glob, argparse, torch, torchaudio
from pathlib import Path
from rwkv_separator_v7_bi import RWKVv7Separator, SeparatorV7Config

# --- RWKV v7 CUDA settings (must be set before importing RWKV model) ---
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")
# -----------------------------------------------------------------------

# ---- import your model exactly as in training ----
# Example:
# from model import build_model   # def build_model() -> torch.nn.Module
# OR: from my_separator import Separator
# Replace the next line with your actual constructor:
def build_model(): 
    raise NotImplementedError("Replace build_model() with your model constructor")

# ---- metrics (same as your training loss) ----
def sisdr(x, s, eps=1e-8):
    # x, s: [T] or [1,T]
    x = x.view(-1)
    s = s.view(-1)
    dot = torch.sum(x * s)
    s_energy = torch.sum(s * s) + eps
    s_target = (dot / s_energy) * s
    e_noise = x - s_target
    num = torch.sum(s_target ** 2) + eps
    den = torch.sum(e_noise ** 2) + eps
    return 10.0 * torch.log10(num / den)

def best_perm_reorder(yhat1, yhat2, y1, y2):
    """Permutation alignment per utterance using SI-SDR (PIT)."""
    sdr_12 = sisdr(yhat1, y1) + sisdr(yhat2, y2)
    sdr_21 = sisdr(yhat1, y2) + sisdr(yhat2, y1)
    if sdr_21 > sdr_12:
        return yhat2, yhat1, sdr_21
    return yhat1, yhat2, sdr_12

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_root", required=True)   # .../tt directory
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_audio", action="store_true", help="write wavs")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) Instantiate the exact same model architecture/config as training
    cfg_rwkv = SeparatorV7Config(
        in_dim=STFT_CFG["n_fft"] // 2 + 1,  # 513
        layers=hp.layers,
        head_size_a=hp.head_size_a,
        hidden_dim=hp.hidden_dim,
        dir_drop_p=hp.dir_drop_p,           # keep 0.0 for the smoke test
        use_mask=hp.use_mask,               # mask-only head
        enforce_bf16=hp.enforce_bf16
    )
    model = RWKVv7Separator(cfg_rwkv).to(device)
    model.eval()

    # 2) Load weights (handle common ckpt formats)
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict({k.replace("module.", ""): v for k,v in state["state_dict"].items()})
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict({k.replace("module.", ""): v for k,v in state["model"].items()})
    else:
        # raw state_dict
        model.load_state_dict({k.replace("module.", ""): v for k,v in state.items()})

    # 3) Collect test files
    mix_dir = Path(args.test_root) / "mix"
    s1_dir  = Path(args.test_root) / "s1"
    s2_dir  = Path(args.test_root) / "s2"
    out_s1  = Path(args.out_dir) / "s1"
    out_s2  = Path(args.out_dir) / "s2"
    out_s1.mkdir(parents=True, exist_ok=True)
    out_s2.mkdir(parents=True, exist_ok=True)

    mix_paths = sorted(glob.glob(str(mix_dir / "*.wav")))
    assert mix_paths, f"No wavs found in {mix_dir}"

    # 4) Loop and separate (batch=1 avoids padding headaches)
    sum_sisdr1 = sum_sisdr2 = 0.0
    sum_improv1 = sum_improv2 = 0.0
    count = 0

    torch.set_grad_enabled(False)
    for mix_path in mix_paths:
        utt = Path(mix_path).name
        ref1_path = s1_dir / utt
        ref2_path = s2_dir / utt
        mix, sr = torchaudio.load(mix_path)   # [1, T]
        y1, _ = torchaudio.load(str(ref1_path))
        y2, _ = torchaudio.load(str(ref2_path))

        # Ensure mono shape [T]
        mix = mix.to(device).mean(0)
        y1  = y1.to(device).mean(0)
        y2  = y2.to(device).mean(0)

        # 4a) Forward
        # Your model’s forward signature should match training (waveform in, two waveforms out)
        # Example: yhat1, yhat2 = model(mix[None, :])  # [1,T] -> [1,T],[1,T]
        yhat1, yhat2 = model(mix[None, :])  # adapt if your forward differs
        yhat1 = yhat1.squeeze(0)
        yhat2 = yhat2.squeeze(0)

        # 4b) Permutation alignment (PIT at inference)
        yhat1, yhat2, _ = best_perm_reorder(yhat1, yhat2, y1, y2)

        # 4c) Truncate/Pad to reference length if tiny mismatches
        T = min(y1.numel(), y2.numel(), yhat1.numel(), yhat2.numel(), mix.numel())
        y1, y2 = y1[:T], y2[:T]
        yhat1, yhat2 = yhat1[:T], yhat2[:T]
        mix = mix[:T]

        # 4d) Metrics
        # mixture-to-source (baseline) and estimate-to-source (system)
        mix_sisdr1 = sisdr(mix, y1).item()
        mix_sisdr2 = sisdr(mix, y2).item()
        est_sisdr1 = sisdr(yhat1, y1).item()
        est_sisdr2 = sisdr(yhat2, y2).item()
        sum_sisdr1 += est_sisdr1
        sum_sisdr2 += est_sisdr2
        sum_improv1 += (est_sisdr1 - mix_sisdr1)  # SI-SDRi
        sum_improv2 += (est_sisdr2 - mix_sisdr2)
        count += 1

        # 4e) Optional: save audio
        if args.save_audio:
            torchaudio.save(str(out_s1 / utt), yhat1.detach().cpu().unsqueeze(0), sr)
            torchaudio.save(str(out_s2 / utt), yhat2.detach().cpu().unsqueeze(0), sr)

    # 5) Report corpus-level results
    print(f"Mean SI-SDR (spk1): {sum_sisdr1 / count:.2f} dB | (spk2): {sum_sisdr2 / count:.2f} dB")
    print(f"Mean SI-SDRi (spk1): {sum_improv1 / count:.2f} dB | (spk2): {sum_improv2 / count:.2f} dB")
    print(f"Files processed: {count}")

if __name__ == "__main__":
    main()