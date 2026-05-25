"""
Knowledge Distillation (KD) helpers for the all-LMul recovery experiment.

Teacher: production gate / d1+c3 ckpt (real-mult conv, packet+L-Mul attention)
Student: all-LMul m4 datapath (current v3/v4 setup)

Total loss:  task_loss(student, clean) + lambda_kd * kd_loss(student_out, teacher_out)

This file provides:
  - `load_kd_teacher(...)`           live-teacher loader (legacy; not used in production
                                     KD because we precompute teacher outputs to disk)
  - `kd_loss_waveform_mse(s, t)`     time-domain MSE; lambda ~ O(1e3)+
  - `kd_loss_stft_l1(s, t, ...)`     STFT magnitude L1; lambda ~ O(1)-O(50)
  - `kd_loss_binaural(s, t, loss)`   BinauralLoss with teacher as target; signed,
                                     lambda ~ O(0.01)-O(0.3); preserves ILD/IPD

Integration is ALREADY WIRED (see INTEGRATION_NOTES below for the wiring map).
Selecting a KD mode is done via `model.kd.kd_mode` config + calibrated lambda
(use scripts/probe_kd_loss_scale.py before launching any new mode/lambda).
"""
from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf


INTEGRATION_NOTES = """
KD integration status: ALREADY WIRED (precomputed-teacher path).

Current production wiring (see DCNN/trainer.py, base_trainer.py, datasets/__init__.py):

  config:
    model.kd.enabled         -> turns KD on/off
    model.kd.teacher_output_dir   -> precomputed teacher .pt cache root
    model.kd.teacher_noisy_root   -> root for relative-path mapping
    model.kd.lambda_kd       -> KD term weight in total loss
    model.kd.kd_mode         -> one of: waveform_mse | stft_l1 | binaural
    model.kd.stft_n_fft / stft_hop  -> STFT params for stft_l1 mode

  dataloader:
    datasets/__init__.py picks KDPrecomputedDataset only for mode=="training"
    when kd.enabled==true. val/test still use BaseDataset (2-tuple).

  loss:
    base_trainer.py _step() unpacks (x, y, teacher) when batch is 3-tuple
    and routes through kd_loss_waveform_mse / kd_loss_stft_l1 / kd_loss_binaural
    based on kd_mode. Total = task_loss + lambda_kd * kd_loss.

KD mode quick guide:
  waveform_mse  - sample-level MSE. Sensitive to phase; lambda needs ~1e3 to be
                  meaningful (per loss-scale probe).
  stft_l1       - STFT magnitude L1. Drops phase. Good for spectral envelope.
                  lambda ~O(1)-O(10).
  binaural      - reuses BinauralLoss (SNR+STOI+ILD+IPD) with teacher as target.
                  Same scale as task_loss; lambda ~O(0.01)-O(0.3). Preserves
                  spatial cues (ILD/IPD) and intelligibility (STOI). The
                  signed-loss caveat: like task_loss, kd_binaural is negative
                  when student/teacher are similar; magnitude-sizing uses abs().

  ALWAYS run scripts/probe_kd_loss_scale.py before picking lambda for a new mode;
  do not guess.

Live-teacher (legacy, not used): `load_kd_teacher()` below loads a teacher model
on demand and runs it inside training. Disabled in favor of precomputed cache
because live forward roughly doubles per-step wall-clock.
"""


def _apply_overrides(cfg: DictConfig, overrides) -> DictConfig:
    """Deep-copy cfg and apply 'key.dotted.path=value' overrides or a dict."""
    teacher_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            OmegaConf.update(teacher_cfg, k, v, merge=True)
    elif isinstance(overrides, (list, tuple)):
        for item in overrides:
            if "=" in item:
                k, v = item.split("=", 1)
                # Try to coerce numeric/bool
                if v.lower() in ("true", "false"):
                    v_cast = v.lower() == "true"
                else:
                    try:
                        v_cast = int(v)
                    except ValueError:
                        try:
                            v_cast = float(v)
                        except ValueError:
                            v_cast = v
                OmegaConf.update(teacher_cfg, k.strip(), v_cast, merge=True)
    else:
        raise ValueError(f"unsupported overrides type: {type(overrides)}")
    return teacher_cfg


def load_kd_teacher(student_cfg: DictConfig,
                    teacher_overrides,
                    ckpt_path: str) -> nn.Module:
    """Build a frozen teacher model from `student_cfg` with `teacher_overrides`
    applied, then load `ckpt_path` (Lightning state_dict format).

    Args:
        student_cfg: the DictConfig the student trainer received (we copy + override).
        teacher_overrides: dict OR list of "key=value" strings describing the
            differences between teacher and student configs. For a d1+c3 teacher
            on top of a v3/v4 student, typical contents:
                ["model.attention_datapath.lmul_all_projection_ops=false",
                 "model.low_power.all_lmul_multiply.enabled=false"]
            (Everything else stays the same so the architectures match.)
        ckpt_path: path to teacher checkpoint (Lightning .ckpt).

    Returns:
        nn.Module with all parameters frozen and in eval mode.
    """
    teacher_cfg = _apply_overrides(student_cfg, teacher_overrides)

    # Lazy import to avoid circulars
    from DCNN.trainer import DCNNLightningModule
    lm = DCNNLightningModule(teacher_cfg)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("loss."):
            continue
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v
    res = lm.model.load_state_dict(cleaned, strict=False)
    if res.missing_keys:
        print(f"[kd] teacher load: {len(res.missing_keys)} missing keys; "
              f"first few: {res.missing_keys[:3]}")
    if res.unexpected_keys:
        print(f"[kd] teacher load: {len(res.unexpected_keys)} unexpected; "
              f"first few: {res.unexpected_keys[:3]}")

    lm.model.eval()
    for p in lm.model.parameters():
        p.requires_grad_(False)

    print(f"[kd] teacher loaded from {ckpt_path} "
          f"({sum(p.numel() for p in lm.model.parameters())} params, frozen)")
    return lm.model


def kd_loss_waveform_mse(student_out: torch.Tensor,
                         teacher_out: torch.Tensor) -> torch.Tensor:
    """Output-level MSE on the time-domain waveform.
    Both [B, C, T] float tensors.
    """
    if student_out.shape != teacher_out.shape:
        T = min(student_out.shape[-1], teacher_out.shape[-1])
        student_out = student_out[..., :T]
        teacher_out = teacher_out[..., :T]
    return F.mse_loss(student_out, teacher_out)


def kd_loss_stft_l1(student_out: torch.Tensor,
                    teacher_out: torch.Tensor,
                    n_fft: int = 128,
                    hop: int = 80) -> torch.Tensor:
    """Output-level L1 on STFT magnitude (per channel, averaged).

    Reduces sensitivity to phase, more like a spectral-distance objective
    than time-domain MSE. Useful when phase is unstable but we care about
    spectral envelope alignment.

    Magnitude-only — does NOT preserve teacher's phase / IPD information.
    For MBSTOI recovery scenarios where binaural cues (ILD/IPD) also matter,
    prefer `kd_loss_binaural` so spatial cues stay constrained.

    Both inputs [B, C, T].
    """
    if student_out.shape != teacher_out.shape:
        T = min(student_out.shape[-1], teacher_out.shape[-1])
        student_out = student_out[..., :T]
        teacher_out = teacher_out[..., :T]
    losses = []
    for ch in range(student_out.shape[1]):
        s = torch.stft(student_out[:, ch, :], n_fft=n_fft, hop_length=hop,
                       return_complex=True)
        t = torch.stft(teacher_out[:, ch, :], n_fft=n_fft, hop_length=hop,
                       return_complex=True)
        losses.append((s.abs() - t.abs()).abs().mean())
    return sum(losses) / len(losses)


def kd_loss_binaural(student_out: torch.Tensor,
                     teacher_out: torch.Tensor,
                     binaural_loss_module) -> torch.Tensor:
    """Reuse the model's BinauralLoss with the teacher as the target.

    Concretely, the same SNR + STOI + ILD + IPD composite that drives task
    training is applied a second time with `teacher_out` swapped in for the
    clean target. This way KD shares the *same* objective shape as the task,
    so spatial cues (ILD/IPD) and intelligibility (STOI) are all transferred,
    not just magnitude.

    Compared to `kd_loss_stft_l1` (magnitude only, drops phase):
      - keeps phase/IPD information by routing through the BinauralLoss
        which has its own IPD term
      - keeps ILD by including BinauralLoss's ILD term
      - keeps STOI structure (envelope-correlation flavoured, closer to MBSTOI
        than raw STFT magnitude L1)

    Caveats:
      - magnitudes will be on the same scale as `task_loss` (roughly -12 to
        -15), so calibrated lambda is typically O(0.1)~O(1), not 1e3+.
      - `binaural_loss_module` should be the *same* loss instance the training
        step already uses (`self.loss` inside the LightningModule); the loss
        module precomputes some filterbanks on first call and reusing the
        instance avoids paying that cost twice per step.

    Args:
        student_out: [B, C, T] student waveform
        teacher_out: [B, C, T] precomputed teacher waveform (cast to student
            dtype/device BEFORE calling this function)
        binaural_loss_module: callable, typically self.loss from LightningModule

    Returns: scalar tensor.
    """
    if student_out.shape != teacher_out.shape:
        T = min(student_out.shape[-1], teacher_out.shape[-1])
        student_out = student_out[..., :T]
        teacher_out = teacher_out[..., :T]
    return binaural_loss_module(student_out, teacher_out)
