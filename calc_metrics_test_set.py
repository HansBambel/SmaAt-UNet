import torch
from torch import nn

from root import ROOT_DIR
from utils import dataset_precip, model_classes
from tqdm import tqdm
import os
import numpy as np
import json


def get_metrics_from_model(model, test_dl, threshold=0.5, device: str = "cpu"):
    device = torch.device(device)
    # Precision = tp/(tp+fp)
    # Recall = tp/(tp+fn)
    # Accuracy = (tp+tn)/(tp+fp+tn+fn)
    # F1 = 2 x precision*recall/(precision+recall)
    loss_func = nn.functional.mse_loss
    with torch.no_grad():
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        loss: torch.Tensor = 0.0
        loss_denorm: torch.Tensor = 0.0
        for x, y_true in tqdm(test_dl, leave=True):
            # Move data to device
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = model(x)
            # denormalize
            y_pred_adj = y_pred.squeeze() * 47.83
            y_true_adj = y_true.squeeze() * 47.83

            # calc loss
            loss += loss_func(y_pred.squeeze(), y_true.squeeze(), reduction="sum")
            loss_denorm += loss_func(y_pred_adj, y_true_adj, reduction="sum")

            # convert from mm/5min to mm/h
            y_pred_adj *= 12
            y_true_adj *= 12
            # convert to masks for comparison
            y_pred_mask = y_pred_adj > threshold
            y_true_mask = y_true_adj > threshold

            tn, fp, fn, tp = np.bincount(y_true_mask.cpu().view(-1) * 2 + y_pred_mask.cpu().view(-1), minlength=4)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            # get metrics for sample
        mse_image = loss / len(test_dl)
        mse_denormalized_image = loss_denorm / len(test_dl)
        mse_pixel = mse_denormalized_image / torch.numel(y_true)

        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        f1 = 2 * precision * recall / (precision + recall)
        csi = total_tp / (total_tp + total_fn + total_fp)
        far = total_fp / (total_tp + total_fp)
        hss = ((total_tp * total_tn) - (total_fn * total_fp)) / (
            (total_tp + total_fn) * (total_fn + total_tn) + (total_tp + total_fp) * (total_fp + total_tn)
        )

    return (
        mse_image.item(),
        mse_denormalized_image.item(),
        mse_pixel.item(),
        precision,
        recall,
        accuracy,
        f1,
        csi,
        far,
        hss,
    )


def calculate_metrics_for_models(model_folder, threshold: float = 0.5):
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=ROOT_DIR
        / "data"
        / "precipitation"
        / f"train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_{int(threshold*100)}.h5",
        num_input_images=12,
        num_output_images=6,
        train=False,
    )

    # Move both the model and the data to the same device
    # When using a Mac change this to "mps"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The batch_size and num_workers can/should be adapted for the current hardware specs
    test_dl = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1, persistent_workers=True
    )

    models = [m for m in os.listdir(model_folder) if ".ckpt" in m]

    # go through test set and calculate acc, precision, recall and F1
    model_metrics = {}
    # go through models
    for model_file in tqdm(models, desc="Models", leave=True):
        model, model_name = model_classes.get_model_class(model_file)
        model = model.load_from_checkpoint(model_folder / model_file)
        model.eval()

        mse_image, mse_denormalized_image, mse_pixel, precision, recall, accuracy, f1, csi, far, hss = (
            get_metrics_from_model(model, test_dl, threshold, device=device)
        )
        model_metrics[model_name] = {
            "mse": mse_image,
            "mse_denormalized_image": mse_denormalized_image,
            "mse_pixel": mse_pixel,
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "F1": f1,
            "CSI": csi,
            "FAR": far,
            "HSS": hss,
        }
        print(model_name, model_metrics[model_name])
    return model_metrics


if __name__ == "__main__":
    load_metrics = False

    model_folder = ROOT_DIR / "checkpoints" / "comparison"
    threshold = 0.5

    test_metrics_file = model_folder / f"model_metrics_{threshold}mmh.txt"
    if load_metrics:
        with open(test_metrics_file) as f:
            model_metrics = json.loads(f.read())
    else:
        model_metrics = calculate_metrics_for_models(model_folder, threshold=threshold)
        with open(test_metrics_file, "w") as f:
            json.dump(model_metrics, f, indent=4)
    print(model_metrics)
