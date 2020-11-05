import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from tqdm import tqdm

from utils import data_loader_precip, dataset_precip, data_loader_precip
from models import unet_precip_regression_lightning as unet_regr


def get_model_class(model_file):
    # This is for some nice plotting
    if "UNet_Attention" in model_file:
        model_name = "UNet Attention"
        model = unet_regr.UNet_Attention
    elif "UNetDS_Attention_4kpl" in model_file:
        model_name = "UNetDS Attention with 4kpl"
        model = unet_regr.UNetDS_Attention
    elif "BackbonedUNet" in model_file:
        model_name = "ResNet with UNet"
        model = unet_regr.BackbonedUNet
    elif "UNetDS_Attention_1kpl" in model_file:
        model_name = "UNetDS Attention with 1kpl"
        model = unet_regr.UNetDS_Attention
    elif "UNetDS_Attention_4CBAMs" in model_file:
        model_name = "UNetDS Attention 4CBAMs"
        model = unet_regr.UNetDS_Attention_4CBAMs
    elif "UNetDS_Attention" in model_file:
        model_name = "SmaAt-UNet"
        model = unet_regr.UNetDS_Attention
    elif "UNetDS" in model_file:
        model_name = "UNetDS"
        model = unet_regr.UNetDS
    elif "UNet" in model_file:
        model_name = "UNet"
        model = unet_regr.UNet
    else:
        raise NotImplementedError(f"Model not found")
    return model, model_name


def get_model_loss(model, test_dl, loss="mse", denormalize=True):
    model.eval()  # or model.freeze()?
    model.to("cuda")
    if loss.lower() == "mse":
        loss_func = nn.functional.mse_loss
    elif loss.lower() == "mae":
        loss_func = nn.functional.l1_loss
    factor = 1
    if denormalize:
        factor = 47.83
    # go through test set
    with torch.no_grad():
        loss_model = 0.0
        for x, y_true in tqdm(test_dl, leave=False):
            x = x.to("cuda")
            y_true = y_true.to("cuda")
            y_pred = model(x)
            loss_model += loss_func(y_pred.squeeze() * factor, y_true * factor, reduction="sum") / y_true.size(0)
        loss_model /= len(test_dl)
    return np.array(loss_model.cpu())


def get_persistence_metrics(test_dl, loss="mse", denormalize=True):
    if loss.lower() == "mse":
        loss_func = nn.functional.mse_loss
    elif loss.lower() == "mae":
        loss_func = nn.functional.l1_loss
    factor = 1
    if denormalize:
        factor = 47.83
    threshold = 0.5
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    loss_model = 0.0
    for x, y_true in tqdm(test_dl, leave=False):
        y_pred = x[:, -1, :]
        loss_model += loss_func(y_pred.squeeze() * factor, y_true * factor, reduction="sum") / y_true.size(0)
        # denormalize and convert from mm/5min to mm/h
        y_pred_adj = y_pred.squeeze() * 47.83 * 12
        y_true_adj = y_true.squeeze() * 47.83 * 12
        # convert to masks for comparison
        y_pred_mask = y_pred_adj > threshold
        y_true_mask = y_true_adj > threshold

        # tn, fp, fn, tp = confusion_matrix(y_true_mask.cpu().view(-1), y_pred_mask.cpu().view(-1),
        #                                   labels=[0, 1]).ravel()
        tn, fp, fn, tp = np.bincount(y_true_mask.view(-1) * 2 + y_pred_mask.view(-1), minlength=4)
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
    loss_model /= len(test_dl)
    return loss_model, precision, recall, accuracy, f1, csi, far


def print_persistent_metrics(data_file):
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=data_file,
        num_input_images=12,
        num_output_images=6, train=False)

    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    # persistence_loss = get_persistence_loss(test_dl, loss="mse", denormalize=True)
    # print(persistence_loss)
    loss_model, precision, recall, accuracy, f1, csi, far = get_persistence_metrics(test_dl, loss="mse",
                                                                                    denormalize=True)
    print(
        f"Loss Persistence (MSE): {loss_model}, precision: {precision}, recall: {recall}, accuracy: {accuracy}, f1: {f1}, csi: {csi}, far: {far}")
    return loss_model


def get_model_losses(model_folder, data_file, loss, denormalize):
    # Save it to a dict that can be saved (and plotted)
    test_losses = dict()
    persistence_loss = print_persistent_metrics(data_file)
    test_losses["Persistence"] = persistence_loss

    models = [m for m in os.listdir(model_folder) if ".ckpt" in m]
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=data_file,
        num_input_images=12,
        num_output_images=6, train=False)

    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=6,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # load the models
    for model_file in tqdm(models, desc="Models", leave=True):
        model, model_name = get_model_class(model_file)
        model = model.load_from_checkpoint(f"{model_folder}/{model_file}")
        model_loss = get_model_loss(model, test_dl, loss, denormalize=denormalize)

        test_losses[model_name] = model_loss
    return test_losses


def plot_losses(test_losses, loss):
    names = list(test_losses.keys())
    values = [test_losses[n] for n in test_losses.keys()]
    plt.figure()
    # for name in names:
    plt.bar(names, values)
    plt.xticks(rotation=45)
    plt.xlabel('Models')
    plt.ylabel(f'{loss.upper()} on test set')
    plt.title("Comparison of different models")

    plt.show()


if __name__ == '__main__':
    loss = "mse"
    denormalize = True
    # Models that are compared should be in this folder (the ones with the lowest validation error)
    model_folder = "checkpoints/comparison"
    data_file = 'data/precipitation/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5'

    # This changes whether to load or to run the model loss calculation
    load = False
    if load:
        # load the losses
        with open(f"checkpoints/comparison/model_losses_{loss.upper()}_denormalized.pkl", "rb") as f:
            test_losses = pickle.load(f)

    else:
        test_losses = get_model_losses(model_folder, data_file, loss, denormalize)
        # Save losses
        with open(model_folder + f"/model_losses_{loss.upper()}_{f'de' if denormalize else ''}normalized.pkl",
                  "wb") as f:
            pickle.dump(test_losses, f)

    # Plot results
    print(list(test_losses.keys()))
    plot_losses(test_losses, loss)

