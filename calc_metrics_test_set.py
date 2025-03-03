import argparse
from argparse import Namespace
import json
import csv
import numpy as np
import os
from pprint import pprint
import torch
from tqdm import tqdm
import lightning.pytorch as pl

from metric.precipitation_metrics import PrecipitationMetrics
from models.unet_precip_regression_lightning import PersistenceModel
from root import ROOT_DIR
from utils import dataset_precip, model_classes


def convert_tensors_to_python(obj):
    """Convert PyTorch tensors to Python native types recursively."""
    if isinstance(obj, torch.Tensor):
        # Convert tensor to float/int, handling nan values
        value = obj.item() if obj.numel() == 1 else obj.tolist()
        return float('nan') if isinstance(value, float) and np.isnan(value) else value
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_python(value) for value in obj]
    return obj


def save_metrics(results, file_path, file_type="json"):
    """
    Save metrics to a file in the specified format.
    
    Args:
        results (dict): Dictionary containing model metrics
        file_path (Path): Path to save the file
        file_type (str): File format - "json", "txt", or "csv"
    """
    with open(file_path, "w") as f:
        if file_type == "csv":
            writer = csv.writer(f)
            # Write header row
            if results:
                first_model = next(iter(results.values()))
                writer.writerow(["Model"] + list(first_model.keys()))
                # Write data rows
                for model_name, metrics in results.items():
                    writer.writerow([model_name] + list(metrics.values()))
        else:
            # For json and txt formats, use json.dump
            json.dump(results, f, indent=4)


def run_experiments(model_folder, data_file, threshold=0.5):
    """
    Run test experiments for all models in the model folder.

    Args:
        model_folder (Path): Path to the model folder
        data_file (Path): Path to the data file
        threshold (float): Threshold for the precipitation

    Returns:
        dict: Dictionary containing model metrics
    """
    results = {}

    # Check for CUDA availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set PyTorch to use higher precision for matrix multiplications
    torch.set_float32_matmul_precision('high')

    # Load the dataset
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=data_file, num_input_images=12, num_output_images=6, train=False
    )

    # Create dataloader
    test_dl = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True, persistent_workers=True)
    
    # Create trainer with device specification
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, accelerator='gpu' if torch.cuda.is_available() else 'cpu')

    # Find all models in the model folder
    models = ["PersistenceModel"] + [f for f in os.listdir(model_folder) if f.endswith(".ckpt")]
    if not models:
        raise ValueError("No checkpoint files found in the model folder.")

    for model_file in tqdm(models, desc="Models", leave=True):

        model, model_name = model_classes.get_model_class(model_file)
        print(f"Loading model: {model_name}")

        if model_file == "PersistenceModel":
            loaded_model = model(Namespace())
        else:
            loaded_model = model.load_from_checkpoint(os.path.join(model_folder, model_file))
        
        loaded_model.to(device)
        loaded_model.precip_metrics = PrecipitationMetrics(threshold=threshold).to(device)

        results[model_name] = trainer.test(model=loaded_model, dataloaders=[test_dl])[0]

    return results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate metrics for precipitation models")
    parser.add_argument("--threshold", type=float, default=0.5, choices=[0.2, 0.5],
                        help="Precipitation threshold (0.2 or 0.5)")
    parser.add_argument("--file-type", type=str, default="txt", choices=["json", "txt", "csv"],
                        help="Output file format (json, txt, or csv)")
    parser.add_argument("--load-metrics", action="store_true",
                        help="Load existing metrics instead of running experiments")
    parser.add_argument("--model-folder", type=str, default=None,
                        help="Path to model folder (default: ROOT_DIR/checkpoints/comparison)")
    
    args = parser.parse_args()
    
    # Variables from command line arguments
    load_metrics = args.load_metrics
    threshold = args.threshold
    file_type = args.file_type
    
    model_folder = ROOT_DIR / "checkpoints" / "comparison" if args.model_folder is None else args.model_folder
    data_file = ROOT_DIR / "data" / "precipitation" / f"train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_{int(threshold*100)}.h5"
    
    # Load metrics if available
    file_name = f"model_metrics_{threshold}.{file_type}"
    if load_metrics:
        with open(model_folder / file_name) as f:
            results = json.load(f)
    else:
        results = run_experiments(model_folder, data_file, threshold=threshold)
        # Convert tensors to Python native types
        results = convert_tensors_to_python(results)

    save_metrics(results, model_folder / file_name, file_type)

    print(f"Metrics saved to {model_folder / file_name}")
    pprint(results)


    #TODO: Add plotting of losses as in test_precip_lightning.py
