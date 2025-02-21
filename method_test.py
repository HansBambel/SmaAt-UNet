import json
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import lightning.pytorch as pl

from root import ROOT_DIR
from utils import dataset_precip, model_classes

def get_metrics(
    test_dl,
    threshold: float = 0.5,
    mode: str = "model",  # Options: "model", "persistence"
    model: torch.nn.Module = None,
    device: str = "cpu",
    denormalize: bool = True,
):
    """
    Computes losses and classification metrics.
    
    Parameters:
      test_dl: DataLoader providing (x, y_true) tuples.
      threshold: Threshold to convert continuous predictions into binary masks.
      mode: "model" uses model(x) for predictions; "persistence" uses the last time slice.
      model: Required if mode is "model".
      device: Device string (e.g., "cpu" or "cuda") for model mode.
      denormalize: If True, applies a factor (47.83) to undo normalization.
      
    Returns:
      A dictionary containing:
         - mse_image: Average MSE computed on raw (normalized) predictions.
         - mse_denorm_image: Average MSE computed on denormalized predictions.
         - mse_pixel: MSE per pixel (denormalized loss divided by total pixel count).
         - precision, recall, accuracy, f1, csi, far, hss: Classification metrics.
    """
    loss_func = nn.functional.mse_loss
    # Factor to denormalize if needed.
    factor = 47.83 if denormalize else 1.0

    # Initialize accumulators.
    total_loss = 0.0
    total_loss_denorm = 0.0
    total_pixels = 0
    total_tp = total_fp = total_tn = total_fn = 0

    if mode == "model":
        if model is None:
            raise ValueError("A model must be provided when mode is 'model'.")
        model.to(device)
        model.eval()
        with torch.no_grad():
            for x, y_true in tqdm(test_dl, leave=True, desc="Evaluating Model"):
                x = x.to(device)
                y_true = y_true.to(device)
                y_pred = model(x)

                #TODO: Add to put y_pred and y_true in the same shape (1 happens to be 1,1,288,288)

                # Denormalize and convert to mm/h
                y_pred_adj = y_pred.squeeze() * factor
                y_true_adj = y_true.squeeze() * factor

                # Calculate loss
                total_loss += loss_func(y_pred, y_true, reduction="sum")
                total_loss_denorm += loss_func(y_pred_adj, y_true_adj, reduction="sum")
                total_pixels += y_true.numel()

                # Convert from mm/5min to mm/h.
                y_pred_adj_h = y_pred_adj * 12
                y_true_adj_h = y_true_adj * 12

                # Convert to masks for comparison
                y_pred_mask = y_pred_adj_h > threshold
                y_true_mask = y_true_adj_h > threshold

                # Compute confusion matrix
                # tn, fp, fn, tp = np.bincount((y_true_mask * 2 + y_pred_mask).numpy(), minlength=4)
                tn, fp, fn, tp = np.bincount(y_true_mask.cpu().view(-1) * 2 + y_pred_mask.cpu().view(-1), minlength=4)

                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn

    elif mode == "persistence":
        # In persistence mode, use the last time slice as the forecast.
        with torch.no_grad():
            for x, y_true in tqdm(test_dl, leave=False, desc="Evaluating Persistence"):
                # Assume x has shape [batch, time, ...] and pick the last time step.
                y_pred = x[:, -1, :]

                #TODO: Add to put y_pred and y_true in the same shape (1 happens to be 1,1,288,288)


                # Add batch loss
                total_loss += loss_func(y_pred, y_true, reduction="sum") / y_true.size(0)
                total_loss_denorm += loss_func(y_pred * factor, y_true * factor, reduction="sum") / y_true.size(0)
                total_pixels += y_true.numel()

                # Denormalize and convert to mm/h
                y_pred_adj = y_pred * factor * 12
                y_true_adj = y_true * factor * 12

                # Convert to masks for comparison
                y_pred_mask = y_pred_adj > threshold
                y_true_mask = y_true_adj > threshold

                # Compute confusion matrix
                tn, fp, fn, tp = np.bincount(y_true_mask.view(-1) * 2 + y_pred_mask.view(-1), minlength=4)

                # Accumulate metrics
                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn

    else:
        raise ValueError("mode must be either 'model' or 'persistence'")

    # Compute average losses.
    mse_image = total_loss / len(test_dl)
    mse_denorm_image = total_loss_denorm / len(test_dl)
    mse_pixel = total_loss_denorm / total_pixels if total_pixels > 0 else float("nan")

    # Compute classification metrics.
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float("nan")
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float("nan")
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else float("nan")
    csi = total_tp / (total_tp + total_fn + total_fp) if (total_tp + total_fn + total_fp) > 0 else float("nan")
    far = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float("nan")
    
    # Heidke Skill Score (HSS).
    denom = ((total_tp + total_fn) * (total_fn + total_tn) + (total_tp + total_fp) * (total_fp + total_tn))
    hss = ((total_tp * total_tn) - (total_fn * total_fp)) / denom if denom != 0 else float("nan")

    return {
        "mse_image": mse_image.item() if torch.is_tensor(mse_image) else mse_image,
        "mse_denorm_image": mse_denorm_image.item() if torch.is_tensor(mse_denorm_image) else mse_denorm_image,
        "mse_pixel": mse_pixel.item() if torch.is_tensor(mse_pixel) else mse_pixel,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "csi": csi,
        "far": far,
        "hss": hss,
    }

# ----- Experiment Runner -----
def run_experiments(model_folder, data_file, threshold=0.5):
    results = {}

    # Load the dataset.
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=data_file, num_input_images=12, num_output_images=6, train=False
    )

    # Create dataloaders.
    dl_persistence = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True, persistent_workers=True)
    dl_model = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True, persistent_workers=True)
    dl_test_step = torch.utils.data.DataLoader(dataset, batch_size=6, num_workers=1, shuffle=False, pin_memory=True, persistent_workers=True)

    # ---- 1. Persistence metrics from test_precip_lightning.py (mode="persistence") ----
    results["Persistence"] = [get_metrics(dl_persistence, threshold=threshold, mode="persistence", denormalize=True)]

    models = [f for f in os.listdir(model_folder) if f.endswith(".ckpt")]
    if not models:
        raise ValueError("No checkpoint files found in the model folder.")

    trainer = pl.Trainer(logger=False, enable_checkpointing=False)

    for model_file in tqdm(models, desc="Models", leave=True):
        res_model = {}

        # Load the model
        model, model_name = model_classes.get_model_class(model_file)
        loaded_model = model.load_from_checkpoint(os.path.join(model_folder, model_file))

        # ---- 2. Model metrics from calc_metrics_test_set.py (mode="model") ----
        res_model["test_step"] = trainer.test(model=loaded_model, dataloaders=[dl_test_step])
        results[model_name] = res_model

        # ---- 3. test_step() MSE metrics ----
        loaded_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        res_model["get_metrics"] = get_metrics(dl_model, threshold=threshold, mode="model", model=loaded_model, device=device, denormalize=True)
        results[model_name] = res_model

    return results

if __name__ == "__main__":
    # Choose threshold and file paths.
    threshold = 0.5
    model_folder = ROOT_DIR / "checkpoints" / "comparison"
    if threshold == 0.2:
        data_file = ROOT_DIR / "data" / "precipitation" / "train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_20.h5"
    else:
        data_file = ROOT_DIR / "data" / "precipitation" / "train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_50.h5"
    
    results = run_experiments(model_folder, data_file, threshold=threshold)
    with open(f"results_method_test_{threshold}.json", "w") as f:
        json.dump(results, f, indent=4)
    

    # In order to unify the metrics, I need to:
    #TODO: Persistence vs model mode – is one preferable over the other?
    #TODO: test_dls – Batch size 1 (calc_metrics_test_set.py) vs 6 (test_precip_lightning.py): Should I pick one?
    #TODO: regression_lightning.py – test_step(): MSE and MSE_denormalized

    #TODO: Make https://lightning.ai/docs/torchmetrics/stable/pages/implement.html ?
    #TODO: Add plotting of losses as in test_precip_lightning.py
    #TODO: Add command line arguments for threshold, output format, plotting
