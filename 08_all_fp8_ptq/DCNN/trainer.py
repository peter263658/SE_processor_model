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
        
        # 🔥 READ CHECKPOINT PATH FROM CONFIG FOR FINE-TUNING
        checkpoint_path = config["training"].get("train_checkpoint_path", None)
        
        if checkpoint_path:
            print(f"🔥 Fine-tuning mode: Loading pre-trained model from {checkpoint_path}")
        else:
            print("🔥 Training from scratch")
        
        super().__init__(lightning_module,
                         config["training"]["n_epochs"],
                         early_stopping_config=config["training"]["early_stopping"],
                         checkpoint_path=checkpoint_path,  # 🔥 PASS CHECKPOINT PATH
                        #  strategy=config["training"]["strategy"],
                         accelerator=config["training"]["accelerator"])
                        # accelerator='mps')

    def fit(self, train_dataloaders, val_dataloaders=None):
        super().fit(self._lightning_module, train_dataloaders,
                    val_dataloaders=val_dataloaders)

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
            mp_cfg = {
                "encoder_dw": {"weight_bits": 4, "weight_scale": "per_channel"},
                "encoder_pw": {"weight_bits": 4, "weight_scale": "per_group",
                               "group_size": gsz},
                "decoder_dw": {"weight_bits": 4, "weight_scale": "per_channel"},
                "decoder_pw_outlier": {"weight_bits": 8, "weight_format": "fp8_e4m3"},
                "decoder_pw_normal":  {"weight_bits": 4, "weight_scale": "per_group",
                                       "group_size": gsz},
                "mha": {"weight_bits": 4, "weight_scale": "per_head",
                        "num_heads": num_heads},
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
                fp8_attention=training_cfg.get("fp8_attention", False),
                fp8_scale=training_cfg.get("fp8_scale", False),
            )
        elif training_cfg.get("fp8_qat", False):
            from DCNN.utils.fp8_qat import enable_fp8_qat
            fp8_dtype = training_cfg.get("fp8_dtype", "e4m3")
            model = enable_fp8_qat(model, dtype_str=fp8_dtype)

        super().__init__(model, loss)

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