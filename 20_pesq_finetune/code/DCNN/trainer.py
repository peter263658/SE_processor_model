from omegaconf import OmegaConf
import torch

from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from DCNN.models.binaural_attention_model import BinauralAttentionDCNN
from DCNN.models.model import DCNN
from DCNN.loss import BinauralLoss, Loss
from DCNN.utils.base_trainer import (
    BaseTrainer, BaseLightningModule
)


class DCNNTrainer(BaseTrainer):
    def __init__(self, config):
        lightning_module = DCNNLightningModule(config)

        # Two mutually-exclusive ckpt mechanisms:
        #   training.resume_ckpt_path  → true Lightning resume (weights + optimizer + scheduler + epoch)
        #   training.train_checkpoint_path → warm-start (weights only, fresh Adam)
        # If both are set, resume wins.
        resume_ckpt_path = config["training"].get("resume_ckpt_path", None)
        checkpoint_path = config["training"].get("train_checkpoint_path", None)

        if resume_ckpt_path:
            print(f"🔥 Resume mode: full Lightning state restore from {resume_ckpt_path}")
            checkpoint_path = None  # skip warm-start; Lightning will load full state in fit()
        elif checkpoint_path:
            print(f"🔥 Fine-tuning mode: Loading pre-trained model from {checkpoint_path}")
        else:
            print("🔥 Training from scratch")

        super().__init__(lightning_module,
                         config["training"]["n_epochs"],
                         early_stopping_config=config["training"]["early_stopping"],
                         checkpoint_path=checkpoint_path,
                         accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 1),
                         num_sanity_val_steps=config["training"].get("num_sanity_val_steps", 2),
                         limit_train_batches=config["training"].get("limit_train_batches", 1.0),
                         limit_val_batches=config["training"].get("limit_val_batches", 1.0),
                         checkpoint_save_top_k=config["training"].get("checkpoint_save_top_k", 1),
                         checkpoint_save_every_epoch=config["training"].get("checkpoint_save_every_epoch", False),
                        #  strategy=config["training"]["strategy"],
                         accelerator=config["training"]["accelerator"])
                        # accelerator='mps')

        self._resume_ckpt_path = resume_ckpt_path

    def fit(self, train_dataloaders, val_dataloaders=None):
        super().fit(self._lightning_module, train_dataloaders,
                    val_dataloaders=val_dataloaders,
                    ckpt_path=self._resume_ckpt_path)

    def test(self, test_dataloaders):
        super().test(self._lightning_module, test_dataloaders, ckpt_path="best")


class DCNNLightningModule(BaseLightningModule):
    """This class abstracts the
       training/validation/testing procedures
       used for training a DCNN
    """

    def __init__(self, config):
        config = OmegaConf.to_container(config)
        self.config = config

        if config["model"]["binaural"]:
            if config["model"]["attention"]:
                model = model = BinauralAttentionDCNN(**self.config["model"])
            else:
                model = BinauralAttentionDCNN(**self.config["model"])
            loss = BinauralLoss(
            
                ild_weight=self.config["model"]["ild_weight"],
                ipd_weight=self.config["model"]["ipd_weight"],
                
                stoi_weight=config["model"]["stoi_weight"],

                snr_loss_weight=self.config["model"]["snr_loss_weight"],

                pesq_weight=self.config["model"].get("pesq_weight", 0),

                snr_loss_mode=self.config["model"].get("snr_loss_mode", "snr"),
                )
        else:    
            model = DCNN(**self.config["model"])

            loss = Loss(loss_mode=self.config["model"]["loss_mode"],
                        STOI_weight=self.config["model"]["STOI_weight"],
                        SNR_weight=self.config["model"]["snr_weight"])

        training_cfg = self.config.get("training", {})

        # Mixed-Precision W4A8 QAT takes precedence over plain FP8 QAT.
        if training_cfg.get("mixed_precision_qat", False):
            from DCNN.utils.mixed_precision_qat import enable_mixed_precision_qat

            num_heads = self.config["model"].get("num_heads", 16)
            gsz = training_cfg.get("int4_group_size", 256)
            scale_format = training_cfg.get("int4_scale_format", "fp32")
            mp_cfg = {
                "encoder_dw": {"weight_bits": 4, "weight_scale": "per_channel",
                               "scale_format": scale_format},
                "encoder_pw": {"weight_bits": 4, "weight_scale": "per_group",
                               "group_size": gsz, "scale_format": scale_format},
                "decoder_dw": {"weight_bits": 4, "weight_scale": "per_channel",
                               "scale_format": scale_format},
                "decoder_pw_outlier": {"weight_bits": 8, "weight_format": "fp8_e4m3"},
                "decoder_pw_normal":  {"weight_bits": 4, "weight_scale": "per_group",
                                       "group_size": gsz, "scale_format": scale_format},
                "mha": {"weight_bits": 4, "weight_scale": "per_head",
                        "num_heads": num_heads, "scale_format": scale_format},
            }

            # Scope override: "mha_only" keeps all conv layers in FP8, only MHA uses INT4.
            scope = training_cfg.get("qat_scope", "full")
            if scope == "mha_only":
                fp8_entry = {"weight_bits": 8, "weight_format": "fp8_e4m3"}
                for k in ("encoder_dw", "encoder_pw", "decoder_dw",
                          "decoder_pw_outlier", "decoder_pw_normal"):
                    mp_cfg[k] = fp8_entry
                print(f"🔒 qat_scope={scope}: conv layers preserved in FP8, only MHA uses W4A8.")
            model = enable_mixed_precision_qat(
                model,
                config=mp_cfg,
                activation_format=training_cfg.get("fp8_dtype", "e4m3"),
                top_k=training_cfg.get("top_k_attention", 4),
                fp16_accumulator=training_cfg.get("fp16_accumulator", False),
                attention_datapath=self.config["model"].get("attention_datapath", {}),
            )
        elif training_cfg.get("fp8_qat", False):
            from DCNN.utils.fp8_qat import enable_fp8_qat
            fp8_dtype = training_cfg.get("fp8_dtype", "e4m3")
            model = enable_fp8_qat(model, dtype_str=fp8_dtype)

        all_lmul_cfg = (
            self.config.get("model", {})
            .get("low_power", {})
            .get("all_lmul_multiply", {})
        )
        if all_lmul_cfg.get("enabled", False):
            from DCNN.utils.all_lmul_multiply import enable_all_lmul_multiply
            lbc_cfg = all_lmul_cfg.get("learnable_bias_correction", {}) or {}
            model = enable_all_lmul_multiply(
                model,
                mantissa_bits=all_lmul_cfg.get("mantissa_bits", 4),
                bit_exact=all_lmul_cfg.get("bit_exact", True),
                linear_chunk_size=all_lmul_cfg.get("linear_chunk_size", 0),
                checkpoint_chunks=all_lmul_cfg.get("checkpoint_chunks", False),
                attention_mantissa_bits=all_lmul_cfg.get("attention_mantissa_bits", None),
                learnable_bias_correction=bool(lbc_cfg.get("enabled", False)),
                learnable_bc_init=float(lbc_cfg.get("init_value", 0.0)),
                ste_mode=all_lmul_cfg.get("ste_mode", None),
            )

        # Selective FP8×FP8 L-Mul on the decoder_pw_outlier layers only (B1b
        # full-multiplier-free target). Runs after mixed_precision_qat tagged
        # model._fp8_preserved_names; does NOT touch MHA / INT4 conv.
        outlier_cfg = (
            self.config.get("model", {})
            .get("low_power", {})
            .get("outlier_fp8_lmul", {})
        )
        if outlier_cfg.get("enabled", False):
            from DCNN.utils.all_lmul_multiply import enable_outlier_fp8_lmul
            model = enable_outlier_fp8_lmul(
                model,
                mantissa_bits=outlier_cfg.get("mantissa_bits", 4),
                bit_exact=outlier_cfg.get("bit_exact", True),
            )

        super().__init__(model, loss)

        kd_cfg = self.config.get("model", {}).get("kd", {}) or {}
        self._kd_enabled = bool(kd_cfg.get("enabled", False))
        self._kd_lambda = float(kd_cfg.get("lambda_kd", 0.1))
        self._kd_mode = str(kd_cfg.get("kd_mode", "waveform_mse"))
        self._kd_stft = (
            int(kd_cfg.get("stft_n_fft", 128)),
            int(kd_cfg.get("stft_hop", 80)),
        )
        if self._kd_enabled:
            teacher_dir = kd_cfg.get("teacher_output_dir", None)
            if not teacher_dir:
                raise ValueError("model.kd.enabled=true requires teacher_output_dir")
            print(
                f"[kd] enabled: mode={self._kd_mode}, "
                f"lambda={self._kd_lambda}, teacher_output_dir={teacher_dir}"
            )

    def on_load_checkpoint(self, checkpoint):
        # Resume robustness when toggling learnable_bias_correction across
        # stages (e.g. curriculum stage 1 from v4 Ep5 enables learnable δ for
        # the first time → ckpt has no learnable_bc keys → strict load fails).
        # We pad any missing learnable_bc keys with zeros and silently drop
        # any unexpected ones if learnable δ is disabled. Other key mismatches
        # are left alone so Lightning still raises on real architecture drift.
        state_dict = checkpoint.get("state_dict", {})
        if not state_dict:
            return
        current = self.state_dict()
        current_lbc = {k for k in current if k.endswith(".learnable_bc")}
        ckpt_lbc = {k for k in state_dict if k.endswith(".learnable_bc")}
        missing = current_lbc - ckpt_lbc
        unexpected = ckpt_lbc - current_lbc
        for k in missing:
            state_dict[k] = torch.zeros_like(current[k])
        for k in unexpected:
            del state_dict[k]
        if missing:
            print(f"[ckpt-resume] padded {len(missing)} missing learnable_bc keys with zeros")
        if unexpected:
            print(f"[ckpt-resume] dropped {len(unexpected)} unexpected learnable_bc keys "
                  f"(learnable_bias_correction disabled in current run)")

    def configure_optimizers(self):
        training_cfg = self.config["training"]
        lr = training_cfg["learning_rate"]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        sched_name = training_cfg.get("lr_scheduler", "multistep")
        if sched_name == "plateau":
            plateau_cfg = training_cfg.get("plateau", {})
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=plateau_cfg.get("factor", 0.5),
                patience=plateau_cfg.get("patience", 2),
                threshold=plateau_cfg.get("threshold", 0.05),
                threshold_mode="abs",
                min_lr=plateau_cfg.get("min_lr", 1e-6),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "validation_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        decay_step = training_cfg["learning_rate_decay_steps"]
        decay_value = training_cfg["learning_rate_decay_values"]
        scheduler = MultiStepLR(optimizer, decay_step, decay_value)
        return [optimizer], [scheduler]
