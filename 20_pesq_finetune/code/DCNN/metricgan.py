"""MetricGAN-style PESQ finetuning for the W4A8 deploy BCCTN.

Implements the MetricGAN+ core loop (Fu et al., 2021) adapted to the binaural
enhancer: a spectrogram discriminator D learns to predict the (normalized)
true wideband PESQ of (degraded, clean) pairs; the generator G (the enhancer)
gets an extra loss term pushing D's prediction of its output toward 1.0.
Unlike a fixed differentiable surrogate (torch_pesq), D is re-fitted to the
*actual* ITU PESQ throughout training, which resists surrogate-gaming.

Binaural handling: D scores each ear independently; scores are averaged.
The generator keeps the full BinauralLoss (STOI/IPD/ILD/SNR anchors) so the
spatial-cue selling points are held while PESQ is pushed.

v2 (2026-07-10, per external code review):
  * D warm-up: `model.d_warmup_epochs` epochs train D only (G frozen), so G
    never chases an uninitialized critic. The deploy MBSTOI gate margin is
    ~0.0007 — one bad early epoch can burn it.
  * True-PESQ label failures are EXCLUDED from D's regression (they were
    silently labeled 0.0 == "worst possible", biasing D).
  * D also sees (noisy, clean) pairs, widening its calibrated PESQ range
    below the generator's output quality.
  * D parameters are frozen during the G step (no wasted grads/memory).
  * NaN/Inf gradient sanitization + clip_grad_norm_(1.0) before every
    optimizer step (the quantized complex-MHA backward has a non-finite
    gradient history; BaseTrainer had this, manual optimization does not).
  * D-quality monitoring each validation epoch on a bounded subset:
    MAE / Pearson / Spearman of D prediction vs true PESQ, and the true-PESQ
    mean. g_gan going down only proves G fools D; these prove D tracks PESQ.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from DCNN.loss import BinauralLoss


def true_pesq_norm(clean: np.ndarray, deg: np.ndarray, sr: int = 16000):
    """ITU-T P.862.2 wideband PESQ mapped to [0, 1]; None if it fails."""
    try:
        from pesq import pesq as itu_pesq
        score = itu_pesq(sr, clean.astype(np.float64), deg.astype(np.float64), "wb")
        return float((score + 0.5) / 5.0)  # wb PESQ in [-0.5, 4.5] -> [0, 1]
    except Exception:
        return None


class SpecDiscriminator(nn.Module):
    """CNN metric predictor on stacked (degraded, clean) magnitude spectrograms.

    Input: (B, 2, F, T) magnitude pair -> predicted normalized PESQ in [0, 1].
    BatchNorm-free (metric regression on small batches); LeakyReLU CNN with
    global average pooling, as in MetricGAN's CNN variant.
    """

    def __init__(self, base=16):
        super().__init__()
        ch = [2, base, base * 2, base * 4, base * 4]
        blocks = []
        for i in range(4):
            blocks += [
                nn.utils.spectral_norm(
                    nn.Conv2d(ch[i], ch[i + 1], kernel_size=(5, 5), stride=(2, 2), padding=2)),
                nn.LeakyReLU(0.3),
            ]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(ch[-1], 64), nn.LeakyReLU(0.3),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, deg_mag, clean_mag):
        x = torch.stack([deg_mag, clean_mag], dim=1)  # (B, 2, F, T)
        h = self.features(x)
        h = h.mean(dim=(2, 3))  # global average pool
        return self.head(h).squeeze(-1)  # (B,)


class MetricGANModule(pl.LightningModule):
    """Manual-optimization Lightning module: G = binaural enhancer, D = PESQ critic."""

    def __init__(self, model, config):
        super().__init__()
        self.automatic_optimization = False
        self.model = model  # generator (BinauralAttentionDCNN, possibly QAT-wrapped)
        self.disc = SpecDiscriminator()

        m = config["model"]
        self.base_loss = BinauralLoss(
            ild_weight=m["ild_weight"], ipd_weight=m["ipd_weight"],
            stoi_weight=m["stoi_weight"], snr_loss_weight=m["snr_loss_weight"],
            pesq_weight=0,  # PESQ pressure comes from D, not torch_pesq
            snr_loss_mode=m.get("snr_loss_mode", "snr"),
        )
        self.gan_weight = float(m.get("gan_weight", 12.0))
        self.d_warmup_epochs = int(m.get("d_warmup_epochs", 0))
        self.d_label_subset = int(m.get("d_label_subset", 8))
        self.grad_clip = 1.0
        self.lr_g = float(config["training"]["learning_rate"])
        self.lr_d = float(config["training"].get("lr_d", 5e-4))
        # STFT for D's magnitude input (independent of the model's front end)
        self.n_fft, self.hop = 512, 100
        self.register_buffer("win", torch.hann_window(512), persistent=False)
        self.val_step_outputs = []
        self._dmon = []          # (d_pred, true_pesq_norm) pairs from validation
        self._dmon_max = 128

    # ---- helpers -----------------------------------------------------------
    def _mag(self, wav):
        # wav: (B, N) -> magnitude (B, F, T)
        spec = torch.stft(wav, self.n_fft, self.hop, window=self.win,
                          return_complex=True, center=True)
        return spec.abs()

    def _align(self, a, b):
        """Crop two waveform batches to equal length; the model output can be
        a few samples shorter than the target, which otherwise breaks the
        per-ear STFT stack (deg vs clean get different frame counts)."""
        n = min(a.shape[-1], b.shape[-1])
        return a[..., :n], b[..., :n]

    def _true_labels(self, deg, ref):
        """Per-sample normalized true PESQ labels + validity mask.

        PESQ failures (short/silent signals etc.) are EXCLUDED via the mask
        instead of being labeled 0.0 (== "worst possible", which biased D).
        """
        deg_np = deg.detach().cpu().numpy()
        ref_np = ref.detach().cpu().numpy()
        labels, valid = [], []
        for i in range(deg_np.shape[0]):
            s = true_pesq_norm(ref_np[i], deg_np[i])
            labels.append(0.0 if s is None else s)
            valid.append(s is not None)
        return (torch.tensor(labels, device=deg.device, dtype=torch.float32),
                torch.tensor(valid, device=deg.device, dtype=torch.bool))

    def _sanitize_and_clip(self, module):
        """NaN/Inf-sanitize then norm-clip gradients (quantized MHA backward
        can emit non-finite grads; clip_grad_norm_ on NaN poisons everything,
        so sanitize FIRST)."""
        params = [p for p in module.parameters() if p.grad is not None]
        for p in params:
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        if params:
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

    # ---- training ----------------------------------------------------------
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        x, y = batch
        warmup = self.current_epoch < self.d_warmup_epochs

        # ---------------- D step ----------------
        # True-PESQ labels are expensive (ITU pesq on CPU, ~30 ms/sample), so
        # label only a random subset per batch; D still sees fresh true-PESQ
        # supervision every step without stalling the loop.
        with torch.no_grad():
            out = self.model(x)
        out_a, y_a = self._align(out, y)
        n_lbl = min(self.d_label_subset, x.shape[0])
        sub = torch.randperm(x.shape[0], device=x.device)[:n_lbl]

        d_terms = []
        # (enhanced, clean) and (noisy, clean) pairs, per ear, true-PESQ labels.
        # The noisy pairs anchor D's low end (deploy outputs sit well above
        # noisy but below clean); without them D only ever sees G's narrow
        # quality band and extrapolates badly.
        for deg_src in (out_a, x):
            deg_a, ref_a = self._align(deg_src, y)
            for ch in range(2):
                deg, ref = deg_a[sub, ch], ref_a[sub, ch]
                lbl, valid = self._true_labels(deg, ref)
                if valid.any():
                    pred = self.disc(self._mag(deg[valid]), self._mag(ref[valid]))
                    d_terms.append(F.mse_loss(pred, lbl[valid]))
        # (clean, clean) anchor at 1.0
        for ch in range(2):
            ref = y_a[sub, ch]
            pred_ref = self.disc(self._mag(ref), self._mag(ref))
            d_terms.append(F.mse_loss(pred_ref, torch.ones_like(pred_ref)))

        d_loss = torch.stack([t for t in d_terms]).mean() if d_terms else None
        if d_loss is not None:
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            self._sanitize_and_clip(self.disc)
            opt_d.step()
            self.log("d_loss", d_loss, on_step=True, prog_bar=True)

        # ---------------- G step (skipped during D warm-up) ----------------
        if warmup:
            self.log("d_warmup", 1.0, on_step=True, prog_bar=True)
            return None

        for p in self.disc.parameters():
            p.requires_grad_(False)
        out = self.model(x)
        g_base = self.base_loss(out, y)       # raw out,y: identical to finetune
        out_a, y_a = self._align(out, y)      # align for the discriminator STFT
        g_gan = 0.0
        for ch in range(2):
            d_pred = self.disc(self._mag(out_a[:, ch]), self._mag(y_a[:, ch]))
            g_gan = g_gan + F.mse_loss(d_pred, torch.ones_like(d_pred))
        g_gan = g_gan / 2
        g_loss = g_base + self.gan_weight * g_gan
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        self._sanitize_and_clip(self.model)
        opt_g.step()
        for p in self.disc.parameters():
            p.requires_grad_(True)

        self.log("loss_epoch", g_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("g_gan", g_gan, on_step=True, prog_bar=True)
        return None

    # ---- validation (generator quality + D-vs-true-PESQ monitoring) --------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        vloss = self.base_loss(out, y)
        self.val_step_outputs.append(vloss.detach())

        # D-quality monitor: bounded subset, both ears, true PESQ vs D pred.
        if len(self._dmon) < self._dmon_max:
            out_a, y_a = self._align(out, y)
            k = min(8, x.shape[0])
            for ch in range(2):
                deg, ref = out_a[:k, ch], y_a[:k, ch]
                lbl, valid = self._true_labels(deg, ref)
                if valid.any():
                    pred = self.disc(self._mag(deg[valid]), self._mag(ref[valid]))
                    self._dmon.extend(zip(pred.detach().cpu().tolist(),
                                          lbl[valid].detach().cpu().tolist()))
        return vloss

    @staticmethod
    def _pearson(a, b):
        a = a - a.mean(); b = b - b.mean()
        denom = a.norm() * b.norm()
        return float((a @ b) / denom) if denom > 0 else float("nan")

    def on_validation_epoch_end(self):
        if self.val_step_outputs:
            v = torch.stack(self.val_step_outputs).mean()
            self.log("validation_loss", v, on_epoch=True, prog_bar=True)
            self.val_step_outputs = []
        if len(self._dmon) >= 8:
            pred = torch.tensor([p for p, _ in self._dmon])
            true = torch.tensor([t for _, t in self._dmon])
            mae = float((pred - true).abs().mean())
            pear = self._pearson(pred, true)
            # Spearman = Pearson on ranks (no scipy dependency)
            rp = torch.argsort(torch.argsort(pred)).float()
            rt = torch.argsort(torch.argsort(true)).float()
            spear = self._pearson(rp, rt)
            true_pesq_mean = float(true.mean()) * 5.0 - 0.5  # denormalize
            self.log("dmon_mae", mae, on_epoch=True)
            self.log("dmon_pearson", pear, on_epoch=True)
            self.log("dmon_spearman", spear, on_epoch=True)
            self.log("val_true_pesq", true_pesq_mean, on_epoch=True, prog_bar=True)
            print(f"[D-monitor] n={len(self._dmon)} MAE={mae:.4f} "
                  f"pearson={pear:.3f} spearman={spear:.3f} "
                  f"true_wb_pesq={true_pesq_mean:.3f}")
        self._dmon = []

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.model.parameters(), lr=self.lr_g)
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.lr_d)
        return opt_g, opt_d
