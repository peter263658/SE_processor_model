from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import MultiStepLR
from DCNN.models.binaural_attention_model import BinauralAttentionDCNN
from DCNN.models.model import DCNN
from DCNN.loss import BinauralLoss, Loss
from DCNN.utils.base_trainer import BaseTrainer, BaseLightningModule
from DCNN.quantization import PruningAwareQAT, initialize_pruning_aware_qat
import torch.nn.utils.prune as prune


class DCNNTrainer(BaseTrainer):
    def __init__(self, config):
        lightning_module = DCNNLightningModule(config)
        super().__init__(lightning_module,
                         config["training"]["n_epochs"],
                         early_stopping_config=config["training"]["early_stopping"],
                         checkpoint_path=None,
                         accelerator=config["training"]["accelerator"])

    def fit(self, train_dataloaders, val_dataloaders=None):
        super().fit(self._lightning_module, train_dataloaders,
                    val_dataloaders=val_dataloaders)

    def test(self, test_dataloaders):
        super().test(self._lightning_module, test_dataloaders, ckpt_path="best")


class DCNNLightningModule(BaseLightningModule):
    """Enhanced Lightning module with Pruning-aware QAT support"""

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

        # Load pretrained weights BEFORE pruning/QAT setup
        pretrained_path = config.get("training", {}).get("train_checkpoint_path")
        if pretrained_path and config.get("quantization", {}).get("enable_pruning_qat", False):
            print(f"Loading pretrained weights before QAT setup...")
            try:
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                # Only load model weights, ignore optimizer/scheduler states
                model_state = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() 
                              if k.startswith('model.')}
                model.load_state_dict(model_state, strict=False)
                print("✓ Pretrained weights loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")


        # Initialize pruning-aware QAT if enabled
        self.pqat_manager = None
        if config.get("quantization", {}).get("enable_pruning_qat", False):
            self._setup_pruning_aware_qat(model, config)

        super().__init__(model, loss)
        
        # QAT training phase tracking
        self.qat_phase = config.get("quantization", {}).get("qat_phase", "normal")
        self.calibration_steps = 0
        self.max_calibration_steps = config.get("quantization", {}).get("calibration_steps", 100)

    def _setup_pruning_aware_qat(self, model, config):
        """Setup pruning-aware quantization aware training"""
        qat_config = config.get("quantization", {})
        
        # Get pruning configuration
        pruning_config = qat_config.get("pruning_config", None)
        base_pruning_ratio = qat_config.get("base_pruning_ratio", 0.3)
        quantization_type = qat_config.get("type", "int8")
        
        print(f"Initializing Pruning-aware QAT with {quantization_type} quantization...")
        print(f"Base pruning ratio: {base_pruning_ratio:.1%}")
        
        # Initialize pruning-aware QAT
        self.pqat_manager = initialize_pruning_aware_qat(
            model=model,
            cfg_type=quantization_type,
            pruning_config=pruning_config,
            base_pruning_ratio=base_pruning_ratio
        )
        
        # Print compression statistics
        stats = self.pqat_manager.get_compression_stats()
        print("Compression Statistics:")
        print(f"  Total parameters: {stats['total_parameters']:,}")
        print(f"  Effective parameters: {stats['effective_parameters']:,}")
        print(f"  Compression ratio: {stats['compression_ratio']:.3f}")
        print(f"  Pruning ratio: {float(stats['pruning_ratio']):.1%}")
        print(f"  Theoretical speedup: {stats.get('theoretical_speedup', 1.0):.2f}x")


    def on_train_start(self):
        """Handle QAT calibration at training start"""
        super().on_train_start()
        
        if self.pqat_manager and self.qat_phase == "calibration":
            print("Starting QAT calibration phase...")
            self.pqat_manager.enable_calibration()

    # Replace training_step method in DCNNLightningModule
    def training_step(self, batch, batch_idx):
        """Enhanced training step with QAT support"""
        # Handle calibration phase
        if (self.pqat_manager and 
            self.qat_phase == "calibration" and 
            self.calibration_steps < self.max_calibration_steps):
            
            # Run forward pass for calibration (no gradient updates)
            x, y = batch
            _ = self.model(x)
            
            self.calibration_steps += 1
            
            # Switch to training phase after calibration
            if self.calibration_steps >= self.max_calibration_steps:
                print("Calibration complete, switching to QAT training...")
                self.pqat_manager.disable_calibration()
                self.qat_phase = "training"
            
            # Return dummy loss during calibration (must require gradients)
            return {"loss": torch.zeros(1, device=self.device, requires_grad=True)}
        
        # Normal training step
        return super().training_step(batch, batch_idx)

    def on_train_epoch_end(self):
        """Enhanced epoch end with QAT logging"""
        super().on_train_epoch_end()
        
        # Log compression statistics periodically
        if self.pqat_manager and self.current_epoch % 10 == 0:
            stats = self.pqat_manager.get_compression_stats()
            self.log("compression_ratio", stats["compression_ratio"], on_epoch=True)
            self.log("effective_params", stats["effective_parameters"], on_epoch=True)

    def configure_optimizers(self):
        """Enhanced optimizer configuration for QAT"""
        lr = self.config["training"]["learning_rate"]
        decay_step = self.config["training"]["learning_rate_decay_steps"]
        decay_value = self.config["training"]["learning_rate_decay_values"]

        # Use different learning rates for QAT phase
        if self.pqat_manager and self.qat_phase == "training":
            # Lower learning rate for QAT fine-tuning
            lr = lr * 0.1
            print(f"Using reduced learning rate for QAT: {lr}")

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # Create scheduler with current learning rate
        if self.pqat_manager and self.qat_phase == "training":
            # Use lambda scheduler for QAT phase
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
        else:
            scheduler = MultiStepLR(optimizer, decay_step, decay_value)

        return [optimizer], [scheduler]

    # Update finalize_pruning in DCNNLightningModule
    def finalize_pruning(self):
        """Finalize pruning by removing masks and flattening weights"""
        if self.pqat_manager:
            print("Finalizing pruning by removing masks...")
            
            # Remove pruning masks and flatten weights
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight_mask'):
                    try:
                        # Get the pruned weight before removing mask
                        pruned_weight = module.weight.data.clone()
                        # Remove the parametrization
                        prune.remove(module, 'weight')
                        # Set the weight to the pruned version
                        module.weight.data = pruned_weight
                        print(f"✓ Finalized pruning for {name}")
                    except Exception as e:
                        print(f"Could not finalize pruning for {name}: {e}")
                
                # Handle complex modules
                elif hasattr(module, 'conv_r') and hasattr(module.conv_r, 'weight_mask'):
                    try:
                        pruned_weight_r = module.conv_r.weight.data.clone()
                        pruned_weight_i = module.conv_i.weight.data.clone()
                        prune.remove(module.conv_r, 'weight')
                        prune.remove(module.conv_i, 'weight')
                        module.conv_r.weight.data = pruned_weight_r
                        module.conv_i.weight.data = pruned_weight_i
                        print(f"✓ Finalized complex conv pruning for {name}")
                    except Exception as e:
                        print(f"Could not finalize complex conv pruning for {name}: {e}")
            
            # Recalculate statistics after finalization
            final_stats = self.pqat_manager.get_compression_stats()
            print("Final Model Statistics (after mask removal):")
            print(f"  Effective parameters: {final_stats['effective_parameters']:,}")
            print(f"  Final compression ratio: {final_stats['compression_ratio']:.3f}")
            print(f"  Model size reduction: {(1-final_stats['compression_ratio'])*100:.1f}%")
            print(f"  Theoretical speedup: {final_stats['theoretical_speedup']:.2f}x")

    def export_optimized_model(self, filepath: str):
        """Export the optimized model for deployment"""
        if self.pqat_manager:
            # Finalize pruning first
            self.finalize_pruning()
        
        # Save the model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'compression_stats': self.pqat_manager.get_compression_stats() if self.pqat_manager else None
        }, filepath)
        
        print(f"Optimized model exported to: {filepath}")


class PruningAwareQATTrainer(DCNNTrainer):
    """Specialized trainer for pruning-aware QAT workflow"""
    
    def __init__(self, config):
        # Ensure QAT is enabled
        if "quantization" not in config:
            config["quantization"] = {}
        config["quantization"]["enable_pruning_qat"] = True
        
        super().__init__(config)
    
    # In PruningAwareQATTrainer class, replace fit_with_qat_phases method
    # Fix the _create_trainer method in PruningAwareQATTrainer
    def fit_with_qat_phases(self, train_dataloaders, val_dataloaders=None):
        """Multi-phase training with fixed config handling"""
        print("Starting multi-phase Pruning-aware QAT training...")
        
        # Phase 1: Normal training (if not pre-trained)
        if not self._lightning_module.config.get("quantization", {}).get("skip_normal_training", False):
            print("\n=== Phase 1: Normal Training ===")
            self._lightning_module.qat_phase = "normal"
            super().fit(train_dataloaders, val_dataloaders)
        
        # Import here to avoid circular imports
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning import loggers as pl_loggers
        from pathlib import Path
        
        def _create_trainer(max_epochs: int, phase_name: str = ""):
            # Fixed config handling - use dict access instead of OmegaConf.select
            training_config = self._lightning_module.config.get("training", {})
            
            # Since config was converted to dict, use dict methods
            if isinstance(training_config, dict):
                accelerator = training_config.get("accelerator", "auto")
            else:
                # Fallback for OmegaConf objects
                from omegaconf import OmegaConf
                accelerator = OmegaConf.select(training_config, "accelerator", default="auto")
            
            # Create checkpoint directory in current working directory
            checkpoint_dir = Path.cwd() / "lightning_logs" / phase_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            callbacks = [
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    monitor="validation_loss", 
                    save_top_k=1, 
                    mode="min",
                    filename=f"{phase_name}-{{epoch:02d}}-{{validation_loss:.2f}}"
                )
            ]
            
            # Use CSV logger instead of disabling completely
            logger = pl_loggers.CSVLogger(
                save_dir=Path.cwd() / "lightning_logs",
                name=phase_name
            )
            
            return pl.Trainer(
                max_epochs=max_epochs,
                callbacks=callbacks,
                accelerator=accelerator,
                devices=1,
                enable_progress_bar=True,
                logger=logger,
                log_every_n_steps=50
            )

        # Phase 2: Calibration
        print("\n=== Phase 2: QAT Calibration ===")
        self._lightning_module.qat_phase = "calibration"
        self._lightning_module.calibration_steps = 0
        
        calib_trainer = _create_trainer(max_epochs=1, phase_name="calibration")
        calib_trainer.fit(self._lightning_module, train_dataloaders, val_dataloaders)
        
        # Phase 3: QAT fine-tuning with configurable epochs
        print("\n=== Phase 3: QAT Fine-tuning ===")
        self._lightning_module.qat_phase = "training"
        
        # Use configurable QAT epochs
        qat_epochs = self._lightning_module.config.get("quantization", {}).get("qat_fine_tune_epochs", 5)
        print(f"Running QAT fine-tuning for {qat_epochs} epochs...")
        
        qat_trainer = _create_trainer(max_epochs=qat_epochs, phase_name="qat_training")
        qat_trainer.fit(self._lightning_module, train_dataloaders, val_dataloaders)
        
        # Finalize the model
        print("\n=== Finalizing Optimized Model ===")
        self._lightning_module.finalize_pruning()
        
        print("Pruning-aware QAT training complete!")
    
    def export_for_deployment(self, filepath: str):
        """Export the final optimized model for deployment"""
        self._lightning_module.export_optimized_model(filepath)