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
import matplotlib.pyplot as plt

from metric.precipitation_metrics import PrecipitationMetrics
from models.unet_precip_regression_lightning import PersistenceModel
from root import ROOT_DIR
from utils import dataset_precip, model_classes


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate metrics for precipitation models")

    parser.add_argument("--model-folder", type=str, default=None,
                        help="Path to model folder (default: ROOT_DIR/checkpoints/comparison)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Precipitation threshold)")
    parser.add_argument("--normalized", action="store_false", dest="denormalize", default=True,
                        help="Calculate metrics for normalized data")
    parser.add_argument("--load-metrics", action="store_true",
                        help="Load existing metrics instead of running experiments")
    parser.add_argument("--file-type", type=str, default="txt", choices=["json", "txt", "csv"],
                        help="Output file format (json, txt, or csv)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Path to save the metrics")
    parser.add_argument("--plot", action="store_true", help="Plot the metrics")
    
    return parser.parse_args()


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


def run_experiments(model_folder, data_file, threshold=0.5, denormalize=True):
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

    # Load the dataset
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=data_file, num_input_images=12, num_output_images=6, train=False
    )

    # Create dataloader
    test_dl = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True, persistent_workers=True)

    # Create trainer with device specification
    trainer = pl.Trainer(logger=False, enable_checkpointing=False)


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
        
        # Override the test metrics, required for testing with normalized data
        loaded_model.test_metrics = PrecipitationMetrics(threshold=threshold, denormalize=denormalize)

        results[model_name] = trainer.test(model=loaded_model, dataloaders=[test_dl])[0]

    return results


def plot_metrics(results, save_path=""):
    """
    Plot all metrics from the results dictionary.
    
    Args:
        results (dict): Dictionary containing metrics for different models
        save_path (str): Path to save the plots
    """
    model_names = list(results.keys())
    metrics = list(results[model_names[0]].keys())
    
    for metric_name in metrics:
        # Get values, replacing NaN with None to skip them in the plot
        values = []
        valid_names = []
        
        for model_name in model_names:
            val = results[model_name][metric_name]
            # Skip NaN values
            if not (isinstance(val, float) and np.isnan(val)):
                values.append(val)
                valid_names.append(model_name)
        
        if len(valid_names) == 0:
            continue
            
        plt.figure(figsize=(10, 6))
        plt.bar(valid_names, values)
        plt.xticks(rotation=45)
        plt.xlabel("Models")
        plt.ylabel(f"{metric_name.upper()}")
        plt.title(f"Comparison of different models - {metric_name.upper()}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/{metric_name}.png")
            
        plt.show()

    
def load_metrics_from_file(file_path, file_type):
    """
    Load metrics from a file in the specified format.
    
    Args:
        file_path (Path): Path to the metrics file
        file_type (str): File format - "json", "txt", or "csv"
        
    Returns:
        dict: Dictionary containing model metrics
    """
    results = {}
    
    with open(file_path) as f:
        if file_type == "json" or file_type == "txt":
            # Both json and txt files are saved in JSON format in this application
            results = json.load(f)
        elif file_type == "csv":
            reader = csv.reader(f)
            headers = next(reader)  # Get header row
            for row in reader:
                if len(row) > 0:
                    model_name = row[0]
                    # Create dictionary of metrics for this model
                    model_metrics = {}
                    for i in range(1, len(headers)):
                        if i < len(row):
                            # Try to convert to float if possible
                            try:
                                value = float(row[i])
                            except (ValueError, TypeError):
                                value = row[i]
                            model_metrics[headers[i]] = value
                    results[model_name] = model_metrics
    
    return results


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Variables from command line arguments
    load_metrics = args.load_metrics
    threshold = args.threshold
    file_type = args.file_type
    denormalize = args.denormalize

    # Define paths
    model_folder = ROOT_DIR / "checkpoints" / "comparison" if args.model_folder is None else args.model_folder
    data_file = ROOT_DIR / "data" / "precipitation" / f"train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_{int(threshold*100)}.h5"
    save_dir = model_folder / f"calc_metrics_test_set_results_{'denorm' if denormalize else 'norm'}" if args.save_dir is None else args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    file_name = f"model_metrics_{threshold}_{'denorm' if denormalize else 'norm'}.{file_type}"
    file_path = save_dir / file_name

    # Load metrics if available
    if load_metrics:
        print(f"Loading metrics from {file_path}")
        results = load_metrics_from_file(file_path, file_type)
    else:
        # Run experiments
        print(f"Running experiments with {threshold} threshold and {'denormalized' if denormalize else 'normalized'} data")
        results = run_experiments(model_folder, data_file, threshold=threshold, denormalize=denormalize)

        # Convert tensors to Python native types
        results = convert_tensors_to_python(results)

        save_metrics(results, file_path, file_type)

    # Display results
    print(f"Metrics saved to {file_path}")
    pprint(results)

    # Plot metrics if requested
    if args.plot:
        plots_dir = save_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)

        plot_metrics(results, plots_dir)
        print(f"Plots saved to {plots_dir}")


if __name__ == "__main__":
    main()