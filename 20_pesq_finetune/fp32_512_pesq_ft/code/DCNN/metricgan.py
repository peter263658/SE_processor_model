"""MetricGAN-style PESQ finetuning for the FP32-512 BCCTN.

Implements the MetricGAN+ core loop (Fu et al., 2021) adapted to the binaural
enhancer: a spectrogram discriminator D learns to predict the (normalized)
true wideband PESQ of (degraded, clean) pairs; the generator G (the enhancer)
gets an extra loss term pushing D's prediction of its output toward 1.0.
Unlike a fixed differentiable surrogate (torch_pesq), D is re-fitted to the
*actual* ITU PESQ throughout training, which resists surrogate-gaming.

Binaural handling: D scores each ear independently; scores are averaged.
The generator keeps the full BinauralLoss (STOI/IPD/ILD/SNR anchors) so the
spatial-cue selling points are held while PESQ is pushed.
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
        self.model = model  # generator (BinauralAttentionDCNN)
        self.disc = SpecDiscriminator()

        m = config["model"]
        self.base_loss = BinauralLoss(
            ild_weight=m["ild_weight"], ipd_weight=m["ipd_weight"],
            stoi_weight=m["stoi_weight"], snr_loss_weight=m["snr_loss_weight"],
            pesq_weight=0,  # PESQ pressure comes from D, not torch_pesq
        )
        self.gan_weight = float(m.get("gan_weight", 12.0))
        self.lr_g = float(config["training"]["learning_rate"])
        self.lr_d = float(config["training"].get("lr_d", 5e-4))
        # STFT for D's magnitude input (independent of the model's front end)
        self.n_fft, self.hop = 512, 100
        self.register_buffer("win", torch.hann_window(512), persistent=False)
        self.val_step_outputs = []

    # ---- helpers -----------------------------------------------------------
    def _mag(self, wav):
        # wav: (B, N) -> magnitude (B, F, T)
        spec = torch.stft(wav, self.n_fft, self.hop, window=self.win,
                          return_complex=True, center=True)
        return spec.abs()

    def _ear_pairs(self, out_wav, tgt_wav):
        """Yield per-ear (deg_mag, clean_mag, deg_np, clean_np) tensors."""
        for ch in range(2):
            deg = out_wav[:, ch]
            ref = tgt_wav[:, ch]
            yield self._mag(deg), self._mag(ref), deg, ref

    def _true_labels(self, deg, ref):
        """Per-sample normalized true PESQ labels (CPU, ITU pesq). NaN-safe."""
        deg_np = deg.detach().cpu().numpy()
        ref_np = ref.detach().cpu().numpy()
        labels = []
        for i in range(deg_np.shape[0]):
            s = true_pesq_norm(ref_np[i], deg_np[i])
            labels.append(0.0 if s is None else s)
        return torch.tensor(labels, device=deg.device, dtype=torch.float32)

    # ---- training ----------------------------------------------------------
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        x, y = batch

        # ---------------- D step ----------------
        # True-PESQ labels are expensive (ITU pesq on CPU, ~30 ms/sample), so
        # label only a random subset per batch; D still sees fresh true-PESQ
        # supervision every step without stalling the loop.
        with torch.no_grad():
            out = self.model(x)
        n_lbl = min(8, x.shape[0])
        sub = torch.randperm(x.shape[0], device=x.device)[:n_lbl]
        d_loss = 0.0
        for deg_mag, ref_mag, deg, ref in self._ear_pairs(out, y):
            lbl = self._true_labels(deg[sub], ref[sub])
            d_pred_deg = self.disc(deg_mag[sub], ref_mag[sub])
            d_pred_ref = self.disc(ref_mag[sub], ref_mag[sub])
            d_loss = d_loss + F.mse_loss(d_pred_deg, lbl) \
                            + F.mse_loss(d_pred_ref, torch.ones_like(d_pred_ref))
        d_loss = d_loss / 2
        opt_d.zero_grad(); self.manual_backward(d_loss); opt_d.step()

        # ---------------- G step ----------------
        out = self.model(x)
        g_base = self.base_loss(out, y)
        g_gan = 0.0
        for deg_mag, ref_mag, _, _ in self._ear_pairs(out, y):
            d_pred = self.disc(deg_mag, ref_mag)
            g_gan = g_gan + F.mse_loss(d_pred, torch.ones_like(d_pred))
        g_gan = g_gan / 2
        g_loss = g_base + self.gan_weight * g_gan
        opt_g.zero_grad(); self.manual_backward(g_loss); opt_g.step()

        self.log("loss_epoch", g_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("d_loss", d_loss, on_step=True, prog_bar=True)
        self.log("g_gan", g_gan, on_step=True, prog_bar=True)
        return None

    # ---- validation (generator quality only) --------------------------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        vloss = self.base_loss(out, y)
        self.val_step_outputs.append(vloss.detach())
        return vloss

    def on_validation_epoch_end(self):
        if self.val_step_outputs:
            v = torch.stack(self.val_step_outputs).mean()
            self.log("validation_loss", v, on_epoch=True, prog_bar=True)
            self.val_step_outputs = []

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.model.parameters(), lr=self.lr_g)
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.lr_d)
        return opt_g, opt_d
