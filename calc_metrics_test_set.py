import torch

from root import ROOT_DIR
from utils import dataset_precip, model_classes
from tqdm import tqdm
import os
import pickle
import numpy as np


def get_metrics_from_model(model, test_dl, threshold=0.5):
    # Precision = tp/(tp+fp)
    # Recall = tp/(tp+fn)
    # Accuracy = (tp+tn)/(tp+fp+tn+fn)
    # F1 = 2 x precision*recall/(precision+recall)
    with torch.no_grad():
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        for x, y_true in tqdm(test_dl, leave=False):
            y_pred = model(x)
            # denormalize and convert from mm/5min to mm/h
            y_pred_adj = y_pred.squeeze() * 47.83 * 12
            y_true_adj = y_true.squeeze() * 47.83 * 12
            # convert to masks for comparison
            y_pred_mask = y_pred_adj > threshold
            y_true_mask = y_true_adj > threshold

            tn, fp, fn, tp = np.bincount(y_true_mask.cpu().view(-1) * 2 + y_pred_mask.cpu().view(-1), minlength=4)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            # get metrics for sample
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        f1 = 2 * precision * recall / (precision + recall)
        csi = total_tp / (total_tp + total_fn + total_fp)
        far = total_fp / (total_tp + total_fp)
        hss = ((total_tp * total_tn) - (total_fn * total_fp)) / (
            (total_tp + total_fn) * (total_fn + total_tn) + (total_tp + total_fp) * (total_fp + total_tn)
        )

    return precision, recall, accuracy, f1, csi, far, hss


if __name__ == "__main__":
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=ROOT_DIR
        / "data"
        / "precipitation"
        / "train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_50.h5",
        num_input_images=12,
        num_output_images=6,
        train=False,
    )

    test_dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    model_folder = ROOT_DIR / "checkpoints" / "comparison"
    models = [m for m in os.listdir(model_folder) if ".ckpt" in m]

    # go through test set and calculate acc, precision, recall and F1
    threshold = 0.5  # mm/h

    model_metrics = {}
    # go through models
    for model_file in tqdm(models, desc="Models", leave=True):
        model, model_name = model_classes.get_model_class(model_file)
        model = model.load_from_checkpoint(model_folder / model_file)
        model.eval()

        precision, recall, accuracy, f1, csi, far, hss = get_metrics_from_model(model, test_dl, threshold)
        model_metrics[model_name] = {
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "F1": f1,
            "CSI": csi,
            "FAR": far,
            "HSS": hss,
        }
        print(model_name, model_metrics[model_name])
    with open(model_folder / f"model_metrics_{threshold}mmh.pkl", "wb") as f:
        pickle.dump(model_metrics, f)
