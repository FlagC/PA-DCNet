import os
import argparse
import yaml
import glob
import time
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import RadioMapDataModule, LightningRadioModel


def save_prediction_image(pred_tensor, save_path):
    pred_np = pred_tensor.squeeze().detach().cpu().numpy()
    pred_np = (np.clip(pred_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pred_np, mode="L").save(save_path)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_best_checkpoint(checkpoint_dir, filename_prefix):
    checkpoint_dir = str(checkpoint_dir)
    candidate_checkpoints = glob.glob(os.path.join(checkpoint_dir, filename_prefix + "*.ckpt"))
    if not candidate_checkpoints:
        return None

    checkpoints_with_metric = []
    for ckpt in candidate_checkpoints:
        try:
            loss_part = ckpt.split("val_total_loss=")[-1].split(".ckpt")[0]
            metric_val = float(loss_part)
            checkpoints_with_metric.append((metric_val, ckpt))
        except (ValueError, IndexError):
            continue

    if checkpoints_with_metric:
        checkpoints_with_metric.sort(key=lambda x: x[0])
        return checkpoints_with_metric[0][1]

    candidate_checkpoints.sort(key=os.path.getmtime, reverse=True)
    return candidate_checkpoints[0]


def find_latest_checkpoint(checkpoint_dir, filename_prefix="latest"):
    checkpoint_dir = str(checkpoint_dir)
    candidate_checkpoints = glob.glob(os.path.join(checkpoint_dir, filename_prefix + "*.ckpt"))
    if not candidate_checkpoints:
        return None
    candidate_checkpoints.sort(key=os.path.getmtime, reverse=True)
    return candidate_checkpoints[0]


def test_model(config_path, specified_checkpoint_path=None, device=None):
    cfg = load_config(config_path)

    model_cfg = cfg["Model"]
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    testing_cfg = cfg.get("testing", {})
    paths_cfg = cfg["paths"]

    exp_name = f"{model_cfg['name']}_{model_cfg['version']}"
    results_root = Path(paths_cfg["results_root"])
    run_dir = results_root / exp_name
    checkpoint_dir = run_dir / "checkpoints"

    # 优先从 testing 中读取结果保存目录
    output_dir = testing_cfg.get("results_save_dir", None)
    if output_dir is None or str(output_dir).strip() == "":
        output_dir = run_dir / "test_predictions"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment name: {exp_name}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Prediction images will be saved to: {output_dir}")

    model_params = {
        "in_channels": model_cfg["in_channels"],
        "out_channels": model_cfg["out_channels"],
        "dims": model_cfg["dims"],
        "depths": model_cfg["depths"],
        "ssm_d_state": model_cfg["ssm_d_state"],
        "ssm_d_conv": model_cfg["ssm_d_conv"],
        "ssm_expand": model_cfg["ssm_expand"],
    }
    # Keep this list in sync with src.train so eval uses the same model geometry as training.
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

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 优先级：命令行指定 > testing.checkpoint_path > 自动搜索
    checkpoint_path = specified_checkpoint_path
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        checkpoint_path = testing_cfg.get("checkpoint_path", None)

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        best_prefix = f"{exp_name}_" + cfg["callbacks"]["checkpoint_best"]["filename"].split("{")[0]
        checkpoint_path = find_best_checkpoint(checkpoint_dir, best_prefix)

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        latest_prefix = f"{exp_name}_latest"
        checkpoint_path = find_latest_checkpoint(checkpoint_dir, latest_prefix)

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find a valid checkpoint. Current path: {checkpoint_path}")

    print(f"Loading model from checkpoint: {checkpoint_path}")

    model = LightningRadioModel.load_from_checkpoint(
        checkpoint_path,
        model_params=model_params,
        training_cfg=training_cfg,
        map_location=device,
    )

    model.to(device)
    model.eval()
    print(f"Loaded Lightning model wraps: {type(model.model).__name__}")
    polar_transform = getattr(model.model, "polar_transform", None)
    if polar_transform is not None:
        print(
            "[Model] polar bins = "
            f"{getattr(polar_transform, 'radial_bins', '<none>')} x "
            f"{getattr(polar_transform, 'theta_bins', '<none>')}"
        )

    test_data_cfg = data_cfg.copy()
    test_data_cfg["batch_size"] = int(testing_cfg.get("test_batch_size", 1))

    data_module = RadioMapDataModule(
        data_cfg=test_data_cfg,
        seed=cfg.get("seed", 42),
    )
    data_module.setup(stage="test")
    test_dataloader = data_module.test_dataloader()

    num_save_images = int(testing_cfg.get("num_save_images", len(data_module.test_dataset)))
    saved_count = 0
    total_processing_time = 0.0
    total_inference_time = 0.0

    print(f"Test dataloader length: {len(test_dataloader)}")
    print(f"Test dataset length: {len(data_module.test_dataset)}")
    print(f"Test batch size: {test_data_cfg['batch_size']}")
    print(f"Max images to save: {num_save_images}")

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Generating Predictions"):
            inputs, _, names = batch
            inputs = inputs.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_start_time = time.time()

            predictions = model(inputs)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_end_time = time.time()

            batch_processing_time = batch_end_time - batch_start_time
            total_processing_time += batch_processing_time
            total_inference_time += batch_processing_time

            batch_size_now = predictions.shape[0]
            for i in range(batch_size_now):
                if saved_count >= num_save_images:
                    break
                image_name = names[i]
                save_full_path = output_dir / image_name
                save_prediction_image(predictions[i], save_full_path)
                saved_count += 1

            if saved_count >= num_save_images:
                break

    if saved_count > 0:
        avg_processing_time = total_processing_time / saved_count
        avg_inference_time = total_inference_time / saved_count
        avg_io_time = avg_processing_time - avg_inference_time
    else:
        avg_processing_time = 0.0
        avg_inference_time = 0.0
        avg_io_time = 0.0

    print("\n--- Prediction Complete ---")
    print(f"Saved {saved_count} prediction images to {output_dir}")
    print(f"Total processing time: {total_processing_time:.8f} seconds")
    print(f"Total inference time: {total_inference_time:.8f} seconds")
    print(f"Average total processing time per image: {avg_processing_time:.8f} seconds")
    print(f"Average inference time per image: {avg_inference_time:.8f} seconds")
    print(f"Average I/O time per image: {avg_io_time:.8f} seconds")

    time_file_path = run_dir / "test_average_time.txt"
    with open(time_file_path, "w", encoding="utf-8") as f:
        f.write("# Detailed timing information per image (seconds)\n")
        f.write(f"Experiment name: {exp_name}\n")
        f.write(f"Checkpoint path: {checkpoint_path}\n")
        f.write(f"Output dir: {output_dir}\n")
        f.write(f"Total images processed: {saved_count}\n")
        f.write(f"Total processing time: {total_processing_time:.8f}\n")
        f.write(f"Total inference time: {total_inference_time:.8f}\n")
        f.write(f"Average total processing time: {avg_processing_time:.8f}\n")
        f.write(f"Average inference time: {avg_inference_time:.8f}\n")
        f.write(f"Average I/O time: {avg_io_time:.8f}\n")

    print(f"Detailed timing information saved to: {time_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model with YAML-driven settings.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional specific checkpoint path. Higher priority than YAML testing.checkpoint_path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device string, e.g. cuda:0 or cpu",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    test_model(args.config, args.checkpoint, device)
