import torch
from torchmetrics import Metric
import numpy as np


class PrecipitationMetrics(Metric):
    """
    A custom metric for precipitation forecasting that computes both regression metrics (MSE)
    and classification metrics (precision, recall, F1, etc.) based on a threshold.
    """
    
    def __init__(self, threshold=0.5, denormalize=True, dist_sync_on_step=False):
        """
        Args:
            threshold: Threshold to convert continuous predictions into binary masks (in mm/h)
            denormalize: If True, applies the factor to undo normalization
            dist_sync_on_step: Synchronize metric state across processes at each step
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.threshold = threshold
        self.denormalize = denormalize
        self.factor = 47.83 # Factor to denormalize predictions (mm/h)
        
        # Add states for regression metrics
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_loss_denorm", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_pixels", default=torch.tensor(0), dist_reduce_fx="sum")
        
        # Add states for classification metrics
        self.add_state("total_tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_fn", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds, target):
        """
        Update the metric states with new predictions and targets.
        
        Args:
            preds: Model predictions (normalized)
            target: Ground truth (normalized)
        """
        # Check for NaN values
        if torch.isnan(preds).any() or torch.isnan(target).any():
            print("Warning: NaN values detected in predictions or targets")
            return
        
        # Make sure preds and target have the same shape
        if preds.shape != target.shape:
            if len(preds.shape) < len(target.shape):
                preds = preds.unsqueeze(0)
            elif len(preds.shape) > len(target.shape):
                preds = preds.squeeze()
                # If squeezing made it too small, add dimension back
                if len(preds.shape) < len(target.shape):
                    preds = preds.unsqueeze(0)
        
        # Calculate MSE loss (normalized)
        batch_size = target.size(0)
        loss = torch.nn.functional.mse_loss(preds, target, reduction="sum") / batch_size
        self.total_loss += loss
        self.total_samples += batch_size  # Use batch_size instead of 1
        self.total_pixels += target.numel()

        # Denormalize if needed
        if self.denormalize:
            preds_updated = preds * self.factor
            target_updated = target * self.factor

            # Calculate denormalized MSE loss
            loss_denorm = torch.nn.functional.mse_loss(preds_updated, target_updated, reduction="sum") / batch_size
            self.total_loss_denorm += loss_denorm
        else:
            preds_updated = preds
            target_updated = target
            
        # Convert to mm/h for classification metrics (multiply by 12 for 5min to hourly rate)
        preds_hourly = preds_updated * 12
        target_hourly = target_updated * 12
        
        # Apply threshold to get binary masks
        preds_mask = preds_hourly > self.threshold
        target_mask = target_hourly > self.threshold
        
        # Compute confusion matrix
        confusion = target_mask.view(-1) * 2 + preds_mask.view(-1)
        bincount = torch.bincount(confusion, minlength=4)
        
        # Update confusion matrix states
        self.total_tn += bincount[0]
        self.total_fp += bincount[1]
        self.total_fn += bincount[2]
        self.total_tp += bincount[3]
    
    def compute(self):
        """
        Compute the final metrics from the accumulated states.
        
        Returns:
            A dictionary containing all metrics.
        """
        # Compute regression metrics
        mse = self.total_loss / self.total_samples
        mse_denorm = self.total_loss_denorm / self.total_samples if self.denormalize else torch.tensor(float('nan'))
        mse_pixel = self.total_loss_denorm / self.total_pixels if self.denormalize else torch.tensor(float('nan'))
        
        # Compute classification metrics
        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else torch.tensor(float('nan'))
        recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else torch.tensor(float('nan'))
        accuracy = (self.total_tp + self.total_tn) / (self.total_tp + self.total_tn + self.total_fp + self.total_fn)
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(float('nan'))
        
        # Critical Success Index (CSI) or Threat Score
        csi = self.total_tp / (self.total_tp + self.total_fn + self.total_fp) if (self.total_tp + self.total_fn + self.total_fp) > 0 else torch.tensor(float('nan'))
        
        # False Alarm Ratio (FAR)
        far = self.total_fp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else torch.tensor(float('nan'))
        
        # Heidke Skill Score (HSS)
        denom = ((self.total_tp + self.total_fn) * (self.total_fn + self.total_tn) + 
                 (self.total_tp + self.total_fp) * (self.total_fp + self.total_tn))
        hss = ((self.total_tp * self.total_tn) - (self.total_fn * self.total_fp)) / denom if denom > 0 else torch.tensor(float('nan'))
        
        return {
            "mse": mse,
            "mse_denorm": mse_denorm,
            "mse_pixel": mse_pixel,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1,
            "csi": csi,
            "far": far,
            "hss": hss
        } 