import torch
from utils import dataset_precip, model_classes
from models import unet_precip_regression_lightning as unet_regr
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
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
            x = x.to("cuda")
            y_true = y_true.to("cuda")
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
        csi = total_tp/(total_tp+total_fn+total_fp)
        far = total_fp/(total_tp+total_fp)
        hss = ((total_tp*total_tn)-(total_fn*total_fp))/((total_tp+total_fn)*(total_fn+total_tn)+(total_tp+total_fp)*(total_fp+total_tn))

    return precision, recall, accuracy, f1, csi, far, hss

if __name__ == '__main__':

    dataset = dataset_precip.precipitation_maps_oversampled_h5(
                in_file="data/precipitation/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5",
                num_input_images=12,
                num_output_images=6, train=False)

    test_dl = data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
        )

    model_folder = "lightning/precip_regression/comparison"
    models = [m for m in os.listdir(model_folder) if ".ckpt" in m]

    # go through test set and calculate acc, precision, recall and F1
    threshold = 0.5  # mm/h

    model_metrics = dict()
    # go through models
    for model_file in tqdm(models, desc="Models", leave=True):
        model, model_name = model_classes.get_model_class(model_file)
        model = model.load_from_checkpoint(f"{model_folder}/{model_file}")
        model.to("cuda").eval()

        precision, recall, accuracy, f1, csi, far, hss = get_metrics_from_model(model, test_dl, threshold)
        model_metrics[model_name] = {"Precision": precision,
                                     "Recall": recall,
                                     "Accuracy": accuracy,
                                     "F1": f1,
                                     "CSI": csi,
                                     "FAR": far,
                                     "HSS": hss}
        print(model_name, model_metrics[model_name])
    with open(model_folder+f"/model_metrics_{threshold}mmh.pkl", "wb") as f:
        pickle.dump(model_metrics, f)