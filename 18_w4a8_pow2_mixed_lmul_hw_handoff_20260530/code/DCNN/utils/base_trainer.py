import pickle
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import (TQDMProgressBar,
                                         ModelCheckpoint, EarlyStopping
                                         )
from pytorch_lightning import loggers as pl_loggers

from DCNN.utils.model_utilities import merge_list_of_dicts



SAVE_DIR = "logs/"


class BaseTrainer(pl.Trainer):
    def __init__(self,
                 lightning_module,
                 n_epochs,
                 use_checkpoint_callback=True,
                 checkpoint_path=None,
                 early_stopping_config=None,
                 strategy=None,              # ← 預設 None
                 accelerator=None,           # ← 也是 None
                 profiler=None,
                 accumulate_grad_batches=1,
                 num_sanity_val_steps=2,
                 limit_train_batches=1.0,
                 limit_val_batches=1.0,
                 checkpoint_save_top_k=1,
                 checkpoint_save_every_epoch=False):

        gpu_count = torch.cuda.device_count()

        if accelerator is None:
            accelerator = "cuda" if torch.cuda.is_available() else "cpu"
        # if accelerator == "mac":
        #     accelerator = "auto" 


        # strategy = strategy if gpu_count > 1 else None
        strategy = strategy if gpu_count > 1 else "auto"

        progress_bar = CustomProgressBar()
        # save_top_k>1 keeps the N best (by validation_loss) epoch ckpts so we
        # can MBSTOI-screen them post-hoc — validation_loss is a poor MBSTOI
        # proxy, so the val-best is often NOT the MBSTOI-best. filename embeds
        # epoch/step/val for traceability.
        checkpoint_callback = ModelCheckpoint(
            monitor="validation_loss",
            mode="min",
            save_top_k=int(checkpoint_save_top_k),
            save_last=True,
            save_weights_only=False,
            filename="epoch={epoch}-step={step}-val={validation_loss:.4f}",
            auto_insert_metric_name=False,
        )
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=SAVE_DIR)

        callbacks = [progress_bar]
        if use_checkpoint_callback:
            callbacks.append(checkpoint_callback)
            if checkpoint_save_every_epoch:
                callbacks.append(ModelCheckpoint(
                    save_top_k=-1,
                    every_n_epochs=1,
                    save_last=False,
                    save_weights_only=False,
                    filename="all-epoch={epoch}-step={step}-val={validation_loss:.4f}",
                    auto_insert_metric_name=False,
                ))
        if early_stopping_config is not None and early_stopping_config.get("enabled", True):
            callbacks.append(EarlyStopping(
                monitor=early_stopping_config["key_to_monitor"],
                min_delta=early_stopping_config["min_delta"],
                patience=early_stopping_config["patience_in_epochs"],
                mode="min",
                verbose=True,
            ))

        # super().__init__(
        #     max_epochs=n_epochs,
        #     callbacks=[progress_bar,
        #                checkpoint_callback  # feature_map_callback
        #                ],
        #     logger=[tb_logger], # csv_logger],
        #     accelerator=accelerator,
        #     strategy=strategy,
        #     devices=gpu_count if gpu_count > 0 else None,
        #     log_every_n_steps=400, enable_progress_bar=True, detect_anomaly=False)
        trainer_kwargs = dict(
            max_epochs=n_epochs,
            callbacks=callbacks,
            logger=[tb_logger],
            accelerator=accelerator or ("gpu" if torch.cuda.is_available() else "cpu"),
            devices=max(1, gpu_count),
            log_every_n_steps=400,
            enable_progress_bar=True,
            detect_anomaly=False,
            profiler=profiler,
            accumulate_grad_batches=max(1, int(accumulate_grad_batches or 1)),
            num_sanity_val_steps=max(0, int(num_sanity_val_steps or 0)),
            # float in (0,1] = fraction of batches, int = #batches, 1.0 = full.
            # Lets a fast probe run a partial epoch (e.g. 0.5) for an early
            # validation signal without committing a full ~2 h train epoch.
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
        )
        # 多 GPU 才需要 strategy；單 GPU 交給 Lightning 內部判定
        if gpu_count > 1:
            trainer_kwargs["strategy"] = strategy

        super().__init__(**trainer_kwargs)

        if checkpoint_path is not None:
            _load_checkpoint(lightning_module.model, checkpoint_path)


        self._lightning_module = lightning_module


class BaseLightningModule(pl.LightningModule):
    """Class which abstracts interactions with Hydra
    and basic training/testing/validation conventions
    """

    # def __init__(self, model, loss,
    #              log_step=400):
    #     super().__init__()

    #     self.is_cuda_available = torch.cuda.is_available()

    #     self.model = model
    #     self.loss = loss

    #     self.log_step = log_step

    def __init__(self, model, loss, log_step=400):
        super().__init__()
        self.is_cuda_available = torch.cuda.is_available()
        self.model = model
        self.loss = loss
        self.log_step = log_step
        # Add these lines:
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []

    def _step(self, batch, batch_idx, log_model_output=False,
              log_labels=False):

        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            x, y, teacher_output = batch
        else:
            x, y = batch
            teacher_output = None
        # 1. Compute model output and loss
        output = self.model(x)
        # model_target = self.model(y)
        task_loss = self.loss(output, y)
        loss = task_loss
        kd_loss = None
        if (
            teacher_output is not None
            and self.training
            and getattr(self, "_kd_enabled", False)
        ):
            from DCNN.utils.kd import (
                kd_loss_binaural,
                kd_loss_stft_l1,
                kd_loss_waveform_mse,
            )

            teacher_output = teacher_output.to(
                device=output.device,
                dtype=output.dtype,
                non_blocking=True,
            )
            kd_mode = getattr(self, "_kd_mode", "waveform_mse")
            if kd_mode == "stft_l1":
                kd_loss = kd_loss_stft_l1(
                    output,
                    teacher_output,
                    *getattr(self, "_kd_stft", (128, 80)),
                )
            elif kd_mode == "waveform_mse":
                kd_loss = kd_loss_waveform_mse(output, teacher_output)
            elif kd_mode == "binaural":
                # Reuse the same BinauralLoss instance the task uses, with the
                # teacher waveform as the target.  Same SNR+STOI+ILD+IPD
                # composite as task_loss but oriented toward teacher.
                kd_loss = kd_loss_binaural(output, teacher_output, self.loss)
            else:
                raise ValueError(f"unsupported kd_mode: {kd_mode}")
            loss = task_loss + float(getattr(self, "_kd_lambda", 0.1)) * kd_loss
            self.log("task_loss", task_loss, on_step=False, on_epoch=True)
            self.log("kd_loss", kd_loss, on_step=False, on_epoch=True)
        # from GPUtil import showUtilization as gpu_usage

        output_dict = {
            "loss": loss
        }
        if kd_loss is not None:
            output_dict["task_loss"] = task_loss
            output_dict["kd_loss"] = kd_loss

        # TODO: Add these to a callback
        # 2. Log model output
        if log_model_output:

            output_dict["model_output"] = output

        # 3. Log step metrics
        self.log("loss_epoch", output_dict["loss"],
                 on_step=False, prog_bar=False,on_epoch=True)

        return output_dict

    # def training_step(self, batch, batch_idx):

    #     return self._step(batch, batch_idx,log_model_output=False)

    # def validation_step(self, batch, batch_idx):
    #     return self._step(batch, batch_idx,
    #                       log_model_output=False, log_labels=True)

    # def test_step(self, batch, batch_idx):
    #     return self._step(batch, batch_idx,
    #                       log_model_output=False, log_labels=True)

    def training_step(self, batch, batch_idx):
        output_dict = self._step(batch, batch_idx, log_model_output=False)
        self.train_step_outputs.append(output_dict)
        return output_dict

    def validation_step(self, batch, batch_idx):
        output_dict = self._step(batch, batch_idx, log_model_output=False, log_labels=True)
        self.val_step_outputs.append(output_dict)
        return output_dict

    def test_step(self, batch, batch_idx):
        output_dict = self._step(batch, batch_idx, log_model_output=False, log_labels=True)
        self.test_step_outputs.append(output_dict)
        return output_dict

    def _epoch_end(self, outputs, epoch_type="train", save_pickle=False):
        # 1. Compute epoch metrics
        outputs = merge_list_of_dicts(outputs)
        epoch_stats = {
            f"{epoch_type}_loss": outputs["loss"].mean(),
            f"{epoch_type}_std": outputs["loss"].std()
            
        }

        # outputs.detach()
        # 2. Log epoch metrics
        for key, value in epoch_stats.items():

            self.log(key, value, on_epoch=True, prog_bar=True)

        # 3. Save complete epoch data on pickle
        if save_pickle:
            pickle_filename = f"{epoch_type}.pickle"
            with open(pickle_filename, "wb") as f:
                pickle.dump(outputs, f)

        return epoch_stats

    # def training_epoch_end(self, outputs):
    #     self._epoch_end(outputs)

    # def validation_epoch_end(self, outputs):
    #     self._epoch_end(outputs, epoch_type="validation")

    # def test_epoch_end(self, outputs):
    #     self._epoch_end(outputs, epoch_type="test", save_pickle=True)

    # 在 LightningModule 中
    def on_train_start(self):
        print("root device :", self.trainer.strategy.root_device)
        print("first param  :", next(self.parameters()).device)
    

    def on_train_epoch_end(self):
        self._epoch_end(self.train_step_outputs)
        self.train_step_outputs = []

    def on_validation_epoch_end(self):
        self._epoch_end(self.val_step_outputs, epoch_type="validation")
        self.val_step_outputs = []

    def on_test_epoch_end(self):
        self._epoch_end(self.test_step_outputs, epoch_type="test", save_pickle=True)
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def fit(self, dataset_train, dataset_val):
        super().fit(self.model, dataset_train, val_dataloaders=dataset_val)

    def test(self, dataset_test, ckpt_path="best"):
        super().test(self.model, dataset_test, ckpt_path=ckpt_path)


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


def _load_checkpoint(model, checkpoint_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = {}
    # Strip ONLY the leading "model." (LightningModule wrapper). Inner `.model.`
    # path components — e.g. `encoder.model.0.0...` — must be preserved.
    PREFIX = "model."
    LOSS_PREFIX = "loss."
    for k, v in checkpoint["state_dict"].items():
        if k.startswith(LOSS_PREFIX):
            continue  # loss buffers are recreated when the loss module is built
        if k.startswith(PREFIX):
            k = k[len(PREFIX):]
        state_dict[k] = v

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"[load_checkpoint] {len(result.missing_keys)} missing keys; first few:")
        for k in result.missing_keys[:5]:
            print(f"    {k}")
    if result.unexpected_keys:
        print(f"[load_checkpoint] {len(result.unexpected_keys)} unexpected keys; first few:")
        for k in result.unexpected_keys[:5]:
            print(f"    {k}")
