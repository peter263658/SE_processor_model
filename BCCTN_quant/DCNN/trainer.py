"""
Complete trainer with quantization-only support (no pruning)
"""

from omegaconf import OmegaConf
import torch
from torch.optim.lr_scheduler import MultiStepLR

from DCNN.models.binaural_attention_model import BinauralAttentionDCNN
from DCNN.models.model import DCNN
from DCNN.loss import BinauralLoss, Loss
from DCNN.utils.base_trainer import BaseTrainer, BaseLightningModule
from DCNN.quantization_only import MixedPrecisionQuantizer


class QuantizationOnlyLightningModule(BaseLightningModule):
    """Lightning module with quantization-only support"""
    
    def __init__(self, config):
        config = OmegaConf.to_container(config)
        self.config = config
        
        # Initialize model
        if config["model"]["binaural"]:
            if config["model"]["attention"]:
                model = BinauralAttentionDCNN(**self.config["model"])
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
        
        super().__init__(model, loss)
        
        # Quantization settings
        self.quantizer = None
        self.quantization_enabled = False
        self.calibration_done = False
        self.quantization_phase = "normal"  # "normal", "calibration", "training"
        
    def apply_quantization(self, conv_precision="int8", attention_precision="fp16"):
        """Apply quantization to model"""
        print(f"Applying quantization: Conv={conv_precision}, Attention={attention_precision}")
        
        self.quantizer = MixedPrecisionQuantizer(
            model=self.model,
            conv_precision=conv_precision,
            attention_precision=attention_precision
        )
        
        self.quantizer.quantize_model()
        self.quantization_enabled = True
        
        # Get and log statistics
        stats = self.quantizer.get_quantization_stats()
        print("Quantization Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return stats
    
    def calibrate_quantization(self, dataloader, num_batches=10):
        """Run calibration for INT8 quantization"""
        if not self.quantizer or self.calibration_done:
            return
        
        print(f"Running calibration for {num_batches} batches...")
        
        self.quantizer.enable_calibration()
        self.eval()
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
        
        self.quantizer.disable_calibration()
        self.calibration_done = True
        print("Calibration complete!")
    
    def on_train_start(self):
        """Handle quantization calibration at training start"""
        super().on_train_start()
        
        if self.quantizer and self.quantization_phase == "calibration":
            print("Starting QAT calibration phase...")
            self.quantizer.enable_calibration()
    
    def training_step(self, batch, batch_idx):
        """Enhanced training step with quantization support"""
        # Handle calibration phase
        if (self.quantizer and 
            self.quantization_phase == "calibration" and 
            not self.calibration_done):
            
            # Run forward pass for calibration (no gradient updates)
            with torch.no_grad():
                x, y = batch
                _ = self.model(x)
            
            # Switch to training phase after some batches
            if batch_idx >= 50:  # Calibrate for 50 batches
                print("Calibration complete, switching to QAT training...")
                self.quantizer.disable_calibration()
                self.calibration_done = True
                self.quantization_phase = "training"
            
            # Return dummy loss during calibration
            return {"loss": torch.tensor(0.0, device=self.device, requires_grad=False)}
        
        # Normal training step
        return super().training_step(batch, batch_idx)
    
    def on_train_epoch_end(self):
        """Enhanced epoch end with quantization logging"""
        super().on_train_epoch_end()
        
        # Log quantization statistics periodically
        if self.quantizer and self.current_epoch % 10 == 0:
            stats = self.quantizer.get_quantization_stats()
            self.log("compression_ratio", stats["compression_ratio"], on_epoch=True)
            self.log("quantized_layers", stats["quantized_layers"], on_epoch=True)
    
    def on_test_start(self):
        """Apply quantization before testing if configured"""
        super().on_test_start()
        
        # Check if quantization test is configured
        quant_test_config = self.config.get("quantization_test", {})
        if quant_test_config.get("enable", False):
            # Get current test configuration
            # This would be set by the test script
            conv_prec = getattr(self, 'test_conv_precision', 'fp32')
            attn_prec = getattr(self, 'test_attention_precision', 'fp32')
            
            if conv_prec != 'fp32' or attn_prec != 'fp32':
                self.apply_quantization(conv_prec, attn_prec)
    
    def configure_optimizers(self):
        """Configure optimizers (lower LR for quantized fine-tuning)"""
        lr = self.config["training"]["learning_rate"]
        
        # Use lower learning rate if quantization is enabled
        if self.quantization_enabled:
            lr = lr * 0.1
            print(f"Using reduced learning rate for quantized model: {lr}")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        decay_step = self.config["training"]["learning_rate_decay_steps"]
        decay_value = self.config["training"]["learning_rate_decay_values"]
        scheduler = MultiStepLR(optimizer, decay_step, decay_value)
        
        return [optimizer], [scheduler]
    
    def export_quantized_model(self, filepath: str):
        """Export the quantized model for deployment"""
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'quantization_stats': self.quantizer.get_quantization_stats() if self.quantizer else None,
            'quantization_enabled': self.quantization_enabled,
            'quantized_layers': getattr(self.quantizer, 'quantized_layers', {}) if self.quantizer else {}
        }
        
        torch.save(model_state, filepath)
        print(f"Quantized model exported to: {filepath}")


class QuantizationAwareTrainer(BaseTrainer):
    """Trainer with quantization-aware training support"""
    
    def __init__(self, config):
        # Use QuantizationOnlyLightningModule
        lightning_module = QuantizationOnlyLightningModule(config)
        
        super().__init__(
            lightning_module,
            config["training"]["n_epochs"],
            early_stopping_config=config["training"]["early_stopping"],
            checkpoint_path=config["training"].get("train_checkpoint_path"),
            accelerator=config["training"]["accelerator"]
        )
    
    def quantize_and_finetune(self, train_dataloader, val_dataloader,
                             conv_precision="int8", attention_precision="fp16",
                             num_epochs=10):
        """Apply quantization and fine-tune the model"""
        print(f"\nApplying quantization and fine-tuning...")
        
        # Apply quantization
        stats = self._lightning_module.apply_quantization(conv_precision, attention_precision)
        
        # Run calibration if using INT8
        if conv_precision == "int8":
            calibration_batches = self._lightning_module.config.get(
                "quantization_test", {}).get("calibration", {}).get("num_batches", 10)
            self._lightning_module.calibrate_quantization(train_dataloader, calibration_batches)
        
        # Fine-tune for a few epochs
        original_epochs = self.trainer.max_epochs
        self.trainer.max_epochs = num_epochs
        self._lightning_module.quantization_phase = "training"
        
        super().fit(self._lightning_module, train_dataloader, val_dataloaders=val_dataloader)
        
        # Restore original epochs
        self.trainer.max_epochs = original_epochs
        
        print("Quantization-aware fine-tuning complete!")
        return stats
    
    def fit_with_quantization_phases(self, train_dataloader, val_dataloader,
                                   conv_precision="int8", attention_precision="fp16"):
        """
        Multi-phase training for quantization:
        1. Normal training (if not pre-trained)
        2. Apply quantization
        3. Calibration phase (for INT8)
        4. QAT fine-tuning
        """
        print("Starting multi-phase quantization-aware training...")
        
        # Phase 1: Normal training (if not pre-trained)
        pretrained_path = self._lightning_module.config.get("training", {}).get("train_checkpoint_path")
        if not pretrained_path:
            print("\n=== Phase 1: Normal Training ===")
            self._lightning_module.quantization_phase = "normal"
            super().fit(train_dataloader, val_dataloader)
        else:
            print("Using pre-trained model, skipping normal training phase")
        
        # Phase 2: Apply quantization
        print(f"\n=== Phase 2: Applying Quantization ===")
        stats = self._lightning_module.apply_quantization(conv_precision, attention_precision)
        
        # Phase 3: Calibration (for INT8)
        if conv_precision == "int8":
            print("\n=== Phase 3: Calibration ===")
            self._lightning_module.quantization_phase = "calibration"
            
            # Run a few epochs for calibration
            original_epochs = self.trainer.max_epochs
            self.trainer.max_epochs = min(2, original_epochs)
            super().fit(train_dataloader, val_dataloader)
            self.trainer.max_epochs = original_epochs
        
        # Phase 4: QAT fine-tuning
        print(f"\n=== Phase 4: QAT Fine-tuning ===")
        self._lightning_module.quantization_phase = "training"
        
        # Continue training with quantized model
        super().fit(train_dataloader, val_dataloader)
        
        print("Multi-phase quantization-aware training complete!")
        return stats
    
    def test_quantized(self, test_dataloader, conv_precision="int8", attention_precision="fp16"):
        """Test with specific quantization configuration"""
        # Set test configuration
        self._lightning_module.test_conv_precision = conv_precision
        self._lightning_module.test_attention_precision = attention_precision
        
        # Run test
        super().test(self._lightning_module, test_dataloader)
    
    def export_for_deployment(self, filepath: str):
        """Export the final quantized model for deployment"""
        self._lightning_module.export_quantized_model(filepath)
    
    def benchmark_quantization(self, test_dataloader, quantization_configs):
        """
        Benchmark different quantization configurations
        
        Args:
            test_dataloader: Test dataset
            quantization_configs: List of (conv_precision, attention_precision) tuples
        """
        results = []
        
        for conv_prec, attn_prec in quantization_configs:
            print(f"\nBenchmarking: Conv={conv_prec}, Attention={attn_prec}")
            
            # Reset model to original state
            original_state = self._lightning_module.model.state_dict()
            
            # Apply quantization
            stats = self._lightning_module.apply_quantization(conv_prec, attn_prec)
            
            # Test performance
            test_results = super().test(self._lightning_module, test_dataloader)
            
            # Combine results
            result = {
                "conv_precision": conv_prec,
                "attention_precision": attn_prec,
                **stats,
                **test_results
            }
            results.append(result)
            
            # Restore original model
            self._lightning_module.model.load_state_dict(original_state)
            self._lightning_module.quantizer = None
            self._lightning_module.quantization_enabled = False
        
        return results


def create_quantization_config(enable_quantization=True, 
                             conv_precision="int8",
                             attention_precision="fp16",
                             calibration_batches=10):
    """Create quantization configuration for training"""
    return {
        "quantization_test": {
            "enable": enable_quantization,
            "conv_precision": conv_precision,
            "attention_precision": attention_precision,
            "calibration": {
                "num_batches": calibration_batches
            }
        }
    }