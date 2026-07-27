# src/train.py
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import inspect
from pathlib import Path
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
torch.set_float32_matmul_precision('high')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

# -------------------------
# 项目文件导入
# -------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datasetloader.DataLoader import RadioMapSeerLoader
# ----------模型定义导入----------------
from models import PADCNet as Model
# -------------------------------------

try:
    from torchmetrics.functional.image import structural_similarity_index_measure
except ImportError:
    structural_similarity_index_measure = None


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_cfg(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# DatasetLoader类
# ============================================================
class RadioMapDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg: dict, seed: int = 42):
        super().__init__()
        self.data_cfg = data_cfg
        self.seed = seed

        self.dir_dataset = data_cfg["dataset_root_dir"]
        self.batch_size = int(data_cfg["batch_size"])
        self.num_workers = int(data_cfg.get("num_workers", 4))

        self.num_tx = int(data_cfg.get("num_tx_per_map", 80))
        self.cars_simul = data_cfg.get("cars_simul", "no")
        self.cars_input = data_cfg.get("cars_input", "no")
        self.fspl_input = data_cfg.get("fspl_input", "no")
        self.reverse_input = data_cfg.get("reverse_input", "no")
        self.thresh = float(data_cfg.get("thresh", 0.0))

        self.transform = transforms.ToTensor()

    def prepare_data(self):
        if not os.path.exists(self.dir_dataset):
            raise FileNotFoundError(f"Dataset path not found: {self.dir_dataset}")

    def setup(self, stage: str = None):
        import numpy as np

        maps_inds = np.arange(0, 700, 1, dtype=np.int16)
        np.random.seed(self.seed)
        np.random.shuffle(maps_inds)

        base_args = dict(
            dir_dataset=self.dir_dataset,
            numTx=self.num_tx,
            carsSimul=self.cars_simul,
            carsInput=self.cars_input,
            fsplInput=self.fspl_input,
            reverseInput=self.reverse_input,
            thresh=self.thresh,
            maps_inds=maps_inds,
            transform=self.transform,
            augment_cfg=self.data_cfg.get("augmentation", {}),
        )

        if stage in (None, "fit"):
            self.train_dataset = RadioMapSeerLoader(phase="train", **base_args)
            self.val_dataset = RadioMapSeerLoader(phase="val", **base_args)
        if stage in (None, "test"):
            self.test_dataset = RadioMapSeerLoader(phase="test", **base_args)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


# ============================================================
# Callback：验证图像保存
# ============================================================
class ValidationImageSaver(Callback):
    def __init__(
        self,
        save_dir: str,
        num_samples: int = 4,
        log_to_tensorboard: bool = True,
        save_to_disk: bool = True,
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.num_samples = int(num_samples)
        self.log_to_tensorboard = bool(log_to_tensorboard)
        self.save_to_disk = bool(save_to_disk)

    def setup(self, trainer, pl_module, stage=None):
        if trainer.global_rank == 0 and self.save_to_disk:
            ensure_dir(self.save_dir)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return

        outs = getattr(pl_module, "val_cache", None)
        if not outs:
            return

        batch = outs[0]
        targets = batch["targets"]
        preds = batch["preds"]
        n = min(self.num_samples, targets.size(0))
        grid = make_grid(torch.cat([targets[:n], preds[:n]]), nrow=n)

        if self.log_to_tensorboard and trainer.logger and hasattr(trainer.logger.experiment, "add_image"):
            trainer.logger.experiment.add_image(
                "Validation/GT_vs_Pred", grid, global_step=trainer.global_step
            )

        if self.save_to_disk:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(n * 3, 7))
            ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
            ax.set_title(f"GT(top) vs Pred(bottom) | step={trainer.global_step}")
            ax.axis("off")
            out_path = self.save_dir / f"step_{trainer.global_step}_val.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

# 验证环节指标展示Processbar
class DeviceInfoPrinter(Callback):
    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return

        root_device = trainer.strategy.root_device
        print(f"[Device] trainer.root_device = {root_device}")
        print(f"[Device] pl_module.device = {pl_module.device}")
        print(f"[Device] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")

        if torch.cuda.is_available() and getattr(root_device, 'type', None) == 'cuda':
            device_index = root_device.index if root_device.index is not None else torch.cuda.current_device()
            print(f"[Device] torch.cuda.current_device = {torch.cuda.current_device()}")
            print(f"[Device] using cuda:{device_index} -> {torch.cuda.get_device_name(device_index)}")
            props = torch.cuda.get_device_properties(device_index)
            print(f"[Device] total_memory = {props.total_memory / (1024 ** 3):.2f} GB")
        else:
            print("[Device] CUDA not available, training is not using a GPU.")


class TrainProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate=1, leave=True):
        super().__init__(refresh_rate=refresh_rate, leave=leave)

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)

        # 去掉版本号
        items.pop("v_num", None)

        # 训练阶段只保留两个指标
        keep_train_keys = ["tr_loss", "tr_ssim"]

        # 验证阶段若 progress bar 出现，则这些指标保留 6 位小数
        keep_val_keys = [
            "val_total_loss",
            "val_l1_loss",
            "val_mse_loss",
            "val_ssim_score",
            "val_ssim_loss",
            "val_gradient_loss",
        ]

        remove_keys = []
        for k in list(items.keys()):
            if k.startswith("train_"):
                remove_keys.append(k)
        for k in remove_keys:
            items.pop(k, None)

        # 训练短指标格式化
        for k in keep_train_keys:
            if k in items:
                v = items[k]
                try:
                    if hasattr(v, "item"):
                        v = v.item()
                    items[k] = f"{float(v):.6f}"
                except Exception:
                    pass

        # 验证指标格式化
        for k in keep_val_keys:
            if k in items:
                v = items[k]
                try:
                    if hasattr(v, "item"):
                        v = v.item()
                    items[k] = f"{float(v):.6f}"
                except Exception:
                    pass

        return items

# ============================================================
# LightningModule：核心训练逻辑
# ============================================================
class LightningRadioModel(pl.LightningModule):
    def __init__(self, model_params: dict, training_cfg: dict):
        super().__init__()
        self.save_hyperparameters()

        model_kwargs = dict(
            in_channels=model_params["in_channels"],
            out_channels=model_params["out_channels"],
            dims=model_params["dims"],
            depths=model_params["depths"],
            ssm_d_state=model_params["ssm_d_state"],
            ssm_d_conv=model_params["ssm_d_conv"],
            ssm_expand=model_params["ssm_expand"],
        )
        model_signature = inspect.signature(Model.__init__).parameters
        for optional_key in [
            "polar_k",
            "ptn_depth",
            "polar_radial_bins",
            "polar_theta_bins",
            "polar_scale_init",
            "cart_scale_init",
            "cart_gate_bias_init",
            "cart_correction_dropout",
            "polar_mamba_depths",
            "use_pgcm",
            "use_pgcm_at_stage1",
            "task",
        ]:
            if optional_key in model_params and optional_key in model_signature:
                model_kwargs[optional_key] = model_params[optional_key]

        self.model = Model(**model_kwargs)

        self.lr = float(training_cfg["learning_rate"])
        self.wd = float(training_cfg.get("weight_decay", 0.0))

        lw = training_cfg.get("loss_weights", {})
        self.w_l1 = float(lw.get("l1", 0.0))
        self.w_mse = float(lw.get("mse", 1.0))
        self.w_ssim = float(lw.get("ssim", 0.0))
        self.w_grad = float(lw.get("gradient", 0.0))

        self.ssim_train_from_epoch = int(training_cfg.get("ssim_train_from_epoch", 0))
        self.ssim_detach = True

        self.lr_patience = int(training_cfg.get("lr_scheduler_patience", 8))
        self.lr_monitor = str(training_cfg.get("lr_monitor", "val_mse_loss"))

        self.l1_fn = nn.L1Loss()
        self.mse_fn = nn.MSELoss()

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        self.val_cache = []
        self._best_monitor_value = None

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        self.ssim_detach = (self.current_epoch < self.ssim_train_from_epoch)
        self.log("train_ssim_detach", float(self.ssim_detach), on_step=False, on_epoch=True, logger=True)

    def _gradient_loss(self, pred, tgt):
        pred_c = pred if (pred.ndim == 4 and pred.size(1) == 1) else pred.unsqueeze(1)
        tgt_c = tgt if (tgt.ndim == 4 and tgt.size(1) == 1) else tgt.unsqueeze(1)

        gx_p = F.conv2d(pred_c, self.sobel_x, padding="same")
        gy_p = F.conv2d(pred_c, self.sobel_y, padding="same")
        gx_t = F.conv2d(tgt_c, self.sobel_x, padding="same")
        gy_t = F.conv2d(tgt_c, self.sobel_y, padding="same")
        return 0.5 * (self.l1_fn(gx_p, gx_t) + self.l1_fn(gy_p, gy_t))

    def _compute_losses(self, out, tgt):
        # SSIM is numerically unreliable in BF16; keep the model under AMP but
        # calculate all losses and reported metrics in FP32.
        with torch.amp.autocast(device_type=out.device.type, enabled=False):
            out_f = out.float()
            tgt_f = tgt.float()

            l1 = self.l1_fn(out_f, tgt_f) if self.w_l1 > 0 else out_f.new_tensor(0.0)
            mse = self.mse_fn(out_f, tgt_f) if self.w_mse > 0 else out_f.new_tensor(0.0)
            grad = self._gradient_loss(out_f, tgt_f) if self.w_grad > 0 else out_f.new_tensor(0.0)

            ssim_score = None
            ssim_loss = out_f.new_tensor(0.0)
            if structural_similarity_index_measure is not None and self.w_ssim > 0:
                out_ssim = out_f.clamp(0.0, 1.0)
                tgt_ssim = tgt_f.clamp(0.0, 1.0)
                if self.ssim_detach:
                    out_ssim = out_ssim.detach()
                ssim_score = structural_similarity_index_measure(
                    out_ssim,
                    tgt_ssim,
                    data_range=1.0,
                    reduction="elementwise_mean",
                )
                ssim_score = ssim_score.clamp(0.0, 1.0)
                ssim_loss = 1.0 - ssim_score

            total = self.w_l1 * l1 + self.w_mse * mse + self.w_ssim * ssim_loss + self.w_grad * grad

        c_l1 = self.w_l1 * l1
        c_mse = self.w_mse * mse
        c_ssim = self.w_ssim * ssim_loss
        c_grad = self.w_grad * grad

        return total, l1, mse, ssim_score, ssim_loss, grad, c_l1, c_mse, c_ssim, c_grad

    def _log_weighted_ratios(self, c_l1, c_mse, c_ssim, c_grad, prefix: str, batch_size: int):
        denom = (c_l1 + c_mse + c_ssim + c_grad).detach() + 1e-12
        r_l1 = c_l1.detach() / denom
        r_mse = c_mse.detach() / denom
        r_ssim = c_ssim.detach() / denom
        r_grad = c_grad.detach() / denom

        self.log_dict(
            {
                f"{prefix}_ratio_l1": r_l1,
                f"{prefix}_ratio_mse": r_mse,
                f"{prefix}_ratio_ssim": r_ssim,
                f"{prefix}_ratio_grad": r_grad,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
            batch_size=batch_size,
        )

    def _log_model_diagnostics(self, prefix: str, batch_size: int):
        fusion = getattr(self.model, "bottleneck_fusion", None)
        diagnostics_fn = getattr(fusion, "diagnostics", None)
        if diagnostics_fn is None:
            return
        diagnostics = diagnostics_fn()
        if not diagnostics:
            return
        self.log_dict(
            {f"{prefix}_{key}": value for key, value in diagnostics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
            batch_size=batch_size,
        )


    def training_step(self, batch, batch_idx):
        x, y, *_ = batch
        bs = x.size(0)
        out = self(x)

        total, l1, mse, ssim_score, ssim_loss, grad, c_l1, c_mse, c_ssim, c_grad = self._compute_losses(out, y)

        # 完整训练指标写入 logger
        self.log_dict(
            {
                "train_total_loss": total,
                "train_l1_loss": l1,
                "train_mse_loss": mse,
                "train_ssim_score": ssim_score,
                "train_ssim_loss": ssim_loss,
                "train_gradient_loss": grad,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
            batch_size=bs,
        )

        # progress bar 只显示两个最重要的训练指标
        self.log(
            "tr_loss",
            total,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
            sync_dist=False,
            batch_size=bs,
        )

        if ssim_score is not None:
            self.log(
                "tr_ssim",
                ssim_score,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=False,
                sync_dist=False,
                batch_size=bs,
            )

        self._log_weighted_ratios(c_l1, c_mse, c_ssim, c_grad, prefix="train", batch_size=bs)
        self._log_model_diagnostics(prefix="train", batch_size=bs)
        return total

    def validation_step(self, batch, batch_idx):
        x, y, *_ = batch
        bs = x.size(0)
        out = self(x)

        total, l1, mse, ssim_score, ssim_loss, grad, c_l1, c_mse, c_ssim, c_grad = self._compute_losses(out, y)

        # 验证指标只写 logger，不走 progress bar
        self.log_dict(
            {
                "val_total_loss": total,
                "val_l1_loss": l1,
                "val_mse_loss": mse,
                "val_ssim_score": ssim_score,
                "val_ssim_loss": ssim_loss,
                "val_gradient_loss": grad,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
            batch_size=bs,
        )

        self._log_weighted_ratios(c_l1, c_mse, c_ssim, c_grad, prefix="val", batch_size=bs)
        self._log_model_diagnostics(prefix="val", batch_size=bs)

        if self.trainer.global_rank == 0 and batch_idx == 0:
            self.val_cache.append(
                {"targets": y.detach().cpu(), "preds": out.detach().cpu().clamp(0.0, 1.0)}
            )

        return total

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        if self.trainer.global_rank != 0:
            return

        metrics = self.trainer.callback_metrics

        keys = [
            "val_total_loss",
            "val_l1_loss",
            "val_mse_loss",
            "val_ssim_score",
            "val_ssim_loss",
            "val_gradient_loss",
        ]

        values = {}
        msg = []
        for k in keys:
            if k in metrics:
                v = metrics[k]
                if hasattr(v, "item"):
                    v = v.item()
                values[k] = float(v)
                msg.append(f"{k}={float(v):.6f}")

        if msg:
            print("\n[VAL] " + " | ".join(msg))

        current = metrics.get(self.lr_monitor, None)
        if current is not None:
            if hasattr(current, "item"):
                current = current.item()
            current = float(current)

            if not hasattr(self, "_best_monitor_value"):
                self._best_monitor_value = None

            improved = (
                self._best_monitor_value is None
                or current < self._best_monitor_value
            )

            if improved:
                self._best_monitor_value = current
                best_msg = []
                best_msg.append(f"{self.lr_monitor}={current:.6f}")
                if "val_total_loss" in values:
                    best_msg.append(f"val_total_loss={values['val_total_loss']:.6f}")
                if "val_ssim_score" in values:
                    best_msg.append(f"val_ssim_score={values['val_ssim_score']:.6f}")
                print("New best scheduler monitor | " + " | ".join(best_msg))

        if self.val_cache:
            self.val_cache.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.1, patience=self.lr_patience
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": self.lr_monitor,
            },
        }


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="PL training entry (YAML-driven).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    seed = int(cfg.get("seed", 42))
    pl.seed_everything(seed, workers=True)

    model_cfg = cfg["Model"]
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    trainer_cfg = dict(cfg.get("trainer_config", {}))
    fit_ctrl = cfg.get("fit_control", {})
    paths_cfg = cfg.get("paths", {})
    cb_cfg = cfg.get("callbacks", {})

    model_name = str(model_cfg.get("name", "unknown_model"))
    model_version = str(model_cfg.get("version", "v0"))
    exp_name = f"{model_name}_{model_version}"
    expected_model_class = str(model_cfg.get("class_name", f"{model_name}Net{model_version}"))
    actual_model_class = Model.__name__

    print(f"[Model] config model name/version = {model_name}/{model_version}")
    print(f"[Model] expected class from config = {expected_model_class}")
    print(f"[Model] imported class in train.py = {actual_model_class}")
    if expected_model_class != actual_model_class:
        print(
            f"[Warning] config expects {expected_model_class}, "
            f"but train.py is using {actual_model_class}."
        )

    results_root = Path(paths_cfg.get("results_root", "./results"))
    tb_root = Path(paths_cfg.get("tb_root", "./tb_logs"))

    run_dir = results_root / exp_name
    ckpt_dir = run_dir / "checkpoints"
    val_img_dir = run_dir / "val_images"

    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(val_img_dir)
    ensure_dir(tb_root)

    dump_cfg(cfg, run_dir / "hparams.yaml")

    logger = TensorBoardLogger(
        save_dir=str(tb_root),
        name=exp_name,
        version="",
        default_hp_metric=False,
    )

    callbacks = []
    callbacks.append(DeviceInfoPrinter())
    callbacks.append(TrainProgressBar(refresh_rate=1, leave=True))

    if "checkpoint_best" in cb_cfg:
        best_cfg = dict(cb_cfg["checkpoint_best"])
        best_cfg["dirpath"] = str(ckpt_dir)
        best_cfg["filename"] = f"{exp_name}_{best_cfg['filename']}"
        callbacks.append(ModelCheckpoint(**best_cfg))

    if "checkpoint_latest" in cb_cfg:
        latest_cfg = dict(cb_cfg["checkpoint_latest"])
        latest_cfg["dirpath"] = str(ckpt_dir)
        latest_cfg["filename"] = f"{exp_name}_{latest_cfg['filename']}"
        callbacks.append(ModelCheckpoint(**latest_cfg))

    if "early_stopping" in cb_cfg:
        callbacks.append(EarlyStopping(**cb_cfg["early_stopping"]))

    save_img_cfg = cb_cfg.get("save_validation_images", {})
    if save_img_cfg.get("enabled", False):
        callbacks.append(
            ValidationImageSaver(
                save_dir=str(val_img_dir),
                num_samples=save_img_cfg.get("num_samples", 4),
                log_to_tensorboard=save_img_cfg.get("log_to_tensorboard", True),
                save_to_disk=save_img_cfg.get("save_to_disk", True),
            )
        )

    model_params = {
        "in_channels": model_cfg["in_channels"],
        "out_channels": model_cfg["out_channels"],
        "dims": model_cfg["dims"],
        "depths": model_cfg["depths"],
        "ssm_d_state": model_cfg["ssm_d_state"],
        "ssm_d_conv": model_cfg["ssm_d_conv"],
        "ssm_expand": model_cfg["ssm_expand"],
    }
    for optional_key in [
        "polar_k",
        "ptn_depth",
        "polar_radial_bins",
        "polar_theta_bins",
        "polar_scale_init",
        "cart_scale_init",
        "cart_gate_bias_init",
        "cart_correction_dropout",
        "polar_mamba_depths",
        "use_pgcm",
        "use_pgcm_at_stage1",
        "task",
    ]:
        if optional_key in model_cfg:
            model_params[optional_key] = model_cfg[optional_key]

    dm = RadioMapDataModule(data_cfg=data_cfg, seed=seed)
    lit = LightningRadioModel(model_params=model_params, training_cfg=training_cfg)

    log_interval = int(trainer_cfg.pop("log_interval", 50))
    val_check_interval = trainer_cfg.pop("val_interval", None)

    trainer_cfg["log_every_n_steps"] = log_interval

    train_mode = str(fit_ctrl.get("train_mode", "epoch")).lower()
    max_epochs = int(fit_ctrl.get("max_epochs", 50))
    max_steps = int(fit_ctrl.get("max_steps", -1))

    if train_mode == "epoch":
        trainer_cfg["max_epochs"] = max_epochs
        trainer_cfg["max_steps"] = -1
        trainer_cfg.pop("val_interval", None)
        trainer_cfg["check_val_every_n_epoch"] = 1
    else:
        trainer_cfg["max_steps"] = max_steps if max_steps > 0 else 100000
        trainer_cfg["max_epochs"] = trainer_cfg.get("max_epochs", 60)
        trainer_cfg["val_check_interval"] = val_check_interval
        trainer_cfg["check_val_every_n_epoch"] = None

    resume_ckpt = str(training_cfg.get("resume_from_checkpoint", "")).strip()
    ckpt_path = resume_ckpt if resume_ckpt else None

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **trainer_cfg,
    )

    trainer.fit(lit, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
