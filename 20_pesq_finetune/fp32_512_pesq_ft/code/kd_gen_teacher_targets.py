"""Generate KD targets: run the saturated FP32-512 w12 PESQ-ft teacher over the
binaural VCTK train/val sets and save enhanced waveforms named like the clean
files (noisy stem minus the _snr suffix), so run_pesq_finetune.sh can consume
them via CLEAN_TRAIN/CLEAN_VAL as drop-in distillation targets.

Run from fp32_512_pesq_ft/code (hydra config lives there):
  CUDA_VISIBLE_DEVICES=<gpu> python ../kd_gen_teacher_targets.py
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from hydra import initialize, compose

TEACHER = "/raid/R12K41024/20_pesq_finetune/fp32_512_pesq_ft/w12sat_best.ckpt"
SETS = {
    "trainset": ("/raid/R12K41024/BCCTN/Dataset/noisy_trainset",
                 "/raid/R12K41024/BCCTN/Dataset/clean_trainset"),
    "valset": ("/raid/R12K41024/BCCTN/Dataset/noisy_valset",
               "/raid/R12K41024/BCCTN/Dataset/clean_valset"),
}
OUT_ROOT = Path("/raid/R12K41024/20_pesq_finetune/data/kd_teacher_w12sat")
BATCH = 32


def load_model():
    sys.path.insert(0, ".")
    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name="config")
    from DCNN.trainer import DCNNLightningModule
    module = DCNNLightningModule(cfg)
    ckpt = torch.load(TEACHER, map_location="cpu")
    state = ckpt["state_dict"]
    result = module.load_state_dict(state, strict=False)
    bad = [k for k in result.unexpected_keys if not k.startswith(("loss.", "disc."))]
    assert not bad, f"teacher load mismatch: {bad[:5]}"
    module.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return module.model.to(dev), dev


def main():
    model, dev = load_model()
    strip_snr = re.compile(r"_snr[-+]?\d+(\.\d+)?$")
    for name, (noisy_dir, clean_dir) in SETS.items():
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(Path(noisy_dir).glob("*.wav"))
        n_clean = len(list(Path(clean_dir).glob("*.wav")))
        print(f"[{name}] {len(files)} noisy files (clean set has {n_clean})", flush=True)
        done = skip = 0
        buf, metas = [], []

        def flush():
            nonlocal done
            if not buf:
                return
            maxlen = max(x.shape[1] for x in buf)
            batch = torch.zeros(len(buf), 2, maxlen)
            for i, x in enumerate(buf):
                batch[i, :, :x.shape[1]] = x
            with torch.no_grad():
                out = model(batch.to(dev)).cpu().numpy()
            for i, (path, n, sr) in enumerate(metas):
                enh = np.clip(out[i, :, :n], -1.0, 1.0)
                sf.write(path, enh.T, sr, subtype="PCM_16")
            done += len(buf)
            if done % (BATCH * 20) < BATCH:
                print(f"  {done}/{len(files)}", flush=True)
            buf.clear(); metas.clear()

        for f in files:
            base = strip_snr.sub("", f.stem)
            out_path = out_dir / f"{base}.wav"
            if out_path.exists():
                skip += 1
                continue
            data, sr = sf.read(f, dtype="float32", always_2d=True)
            buf.append(torch.from_numpy(data.T))
            metas.append((out_path, data.shape[0], sr))
            if len(buf) >= BATCH:
                flush()
        flush()
        n_out = len(list(out_dir.glob("*.wav")))
        print(f"[{name}] wrote/kept {n_out} targets (skipped existing {skip}); "
              f"clean-set count {n_clean}", flush=True)


if __name__ == "__main__":
    main()
