"""Generate a PA-DCNet training configuration."""

import argparse
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a PA-DCNet YAML configuration.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "configs" / "padcnet.yaml")
    parser.add_argument("--task", choices=("srm", "drm"), default="srm")
    parser.add_argument("--version", default="custom")
    parser.add_argument("--dataset-root", default="./data/RadioMapSeer")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--polar-radial-bins", type=int, default=256)
    parser.add_argument("--polar-theta-bins", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--no-augmentation", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> dict:
    is_drm = args.task == "drm"
    return {
        "seed": 42,
        "Model": {
            "name": "PADCNet",
            "version": args.version,
            "class_name": "PADCNet",
            "description": "PA-DCNet with Tx-centered polar guidance and Cartesian correction.",
            "in_channels": 3,
            "out_channels": 1,
            "dims": [48, 96, 192, 384],
            "depths": [2, 3, 4, 2],
            "ssm_d_state": 32,
            "ssm_d_conv": 4,
            "ssm_expand": 2,
            "polar_radial_bins": args.polar_radial_bins,
            "polar_theta_bins": args.polar_theta_bins,
            "polar_mamba_depths": [0, 1, 1, 2],
            "use_pgcm": True,
            "use_pgcm_at_stage1": False,
            "task": args.task,
        },
        "data": {
            "dataset_root_dir": args.dataset_root,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "num_tx_per_map": 80,
            "cars_simul": "yes" if is_drm else "no",
            "cars_input": "yes" if is_drm else "no",
            "fspl_input": "no",
            "reverse_input": "no",
            "thresh": 0.0,
            "augmentation": {
                "enabled": not args.no_augmentation,
                "hflip_p": 0.5,
                "vflip_p": 0.5,
                "rot90": True,
            },
        },
        "training": {
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-4,
            "loss_weights": {"l1": 0.4, "mse": 0.1, "ssim": 0.2, "gradient": 0.3},
            "lr_scheduler_patience": 8,
            "lr_monitor": "val_mse_loss",
            "resume_from_checkpoint": "",
        },
        "paths": {"results_root": "results", "tb_root": "tb_logs"},
        "trainer_config": {
            "accelerator": "gpu",
            "devices": [args.device],
            "precision": args.precision,
            "log_interval": 220,
            "val_interval": 220,
            "gradient_clip_val": 0.0,
        },
        "fit_control": {"train_mode": "epoch", "max_epochs": args.max_epochs, "max_steps": 100000},
        "callbacks": {
            "checkpoint_best": {
                "monitor": "val_total_loss",
                "filename": "best-{epoch}-{step}-{val_total_loss:.4f}",
                "save_top_k": 1,
                "mode": "min",
            },
            "checkpoint_latest": {"filename": "latest", "every_n_epochs": 1, "save_top_k": 1},
            "early_stopping": {"monitor": "val_total_loss", "patience": 14, "mode": "min", "verbose": True},
            "save_validation_images": {
                "enabled": True,
                "num_samples": 4,
                "log_to_tensorboard": True,
                "save_to_disk": False,
            },
        },
        "testing": {"checkpoint_path": "", "test_batch_size": 1},
    }


def main() -> None:
    args = build_parser().parse_args()
    config = build_config(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    print(f"Saved configuration: {args.output}")


if __name__ == "__main__":
    main()
