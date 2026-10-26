import torch
import numpy as np


def make_metrics_str(metrics_dict):
    return " | ".join(
        [
            f"{name}: {value.item() if isinstance(value, torch.Tensor) else value:.4f}"
            for name, value in metrics_dict.items()
            if not (isinstance(value, torch.Tensor) and torch.isnan(value))
            and not (not isinstance(value, torch.Tensor) and np.isnan(value))
        ]
    )
