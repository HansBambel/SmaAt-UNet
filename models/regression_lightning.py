import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import dataset_precip
import argparse
import numpy as np


class UNet_base(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model', type=str, default='UNet',
                            choices=['UNet', 'UNetDS', 'UNet_Attention', 'UNetDS_Attention'])
        parser.add_argument('--n_channels', type=int, default=12)
        parser.add_argument('--n_classes', type=int, default=1)
        parser.add_argument('--kernels_per_layer', type=int, default=1)
        parser.add_argument('--bilinear', type=bool, default=True)
        parser.add_argument('--reduction_ratio', type=int, default=16)
        parser.add_argument('--lr_patience', type=int, default=5)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def forward(self, x):
        pass

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                              mode="min",
                                                              factor=0.1,
                                                              patience=self.hparams.lr_patience),
            'monitor': 'val_loss',  # Default: val_loss
        }
        return [opt], [scheduler]

    def loss_func(self, y_pred, y_true):
        # reduction="mean" is average of every pixel, but I want average of image
        return nn.functional.mse_loss(y_pred, y_true, reduction="sum") / y_true.size(0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred.squeeze(), y)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss_mean = 0.0
        for output in outputs:
            loss_mean += output['loss']

        loss_mean /= len(outputs)
        return {"log": {"train_loss": loss_mean},
                "progress_bar": {"train_loss": loss_mean}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = self.loss_func(y_pred.squeeze(), y)
        # val_loss += loss.item()
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = 0.0
        for output in outputs:
            avg_loss += output["val_loss"]
        avg_loss /= len(outputs)
        logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": logs,
                "progress_bar": {"val_loss": avg_loss}}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = self.loss_func(y_pred.squeeze(), y)
        return {"test_loss": val_loss}

    def test_epoch_end(self, outputs):
        avg_loss = 0.0
        for output in outputs:
            avg_loss += output["test_loss"]
        avg_loss /= len(outputs)
        logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": logs,
                "progress_bar": {"test_loss": avg_loss}}


class Precip_regression_base(UNet_base):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = UNet_base.add_model_specific_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_input_images", type=int, default=12)
        parser.add_argument("--num_output_images", type=int, default=6)
        parser.add_argument("--valid_size", type=float, default=0.1)
        parser.add_argument("--use_oversampled_dataset", type=bool, default=True)
        parser.n_channels = parser.parse_args().num_input_images
        parser.n_classes = 1
        return parser

    def __init__(self, hparams):
        super(Precip_regression_base, self).__init__(hparams=hparams)
        self.train_dataset = None
        self.valid_dataset = None
        self.train_sampler = None
        self.valid_sampler = None

    def prepare_data(self):
        # train_transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip()]
        # )
        train_transform = None
        valid_transform = None
        if self.hparams.use_oversampled_dataset:
            self.train_dataset = dataset_precip.precipitation_maps_oversampled_h5(
                in_file=self.hparams.dataset_folder, num_input_images=self.hparams.num_input_images,
                num_output_images=self.hparams.num_output_images, train=True,
                transform=train_transform
            )
            self.valid_dataset = dataset_precip.precipitation_maps_oversampled_h5(
                in_file=self.hparams.dataset_folder, num_input_images=self.hparams.num_input_images,
                num_output_images=self.hparams.num_output_images, train=True,
                transform=valid_transform
            )
        else:
            self.train_dataset = dataset_precip.precipitation_maps_h5(
                in_file=self.hparams.dataset_folder, num_input_images=self.hparams.num_input_images,
                num_output_images=self.hparams.num_output_images, train=True,
                transform=train_transform
            )
            self.valid_dataset = dataset_precip.precipitation_maps_h5(
                in_file=self.hparams.dataset_folder, num_input_images=self.hparams.num_input_images,
                num_output_images=self.hparams.num_output_images, train=True,
                transform=valid_transform
            )
        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.hparams.valid_size * num_train))

        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, sampler=self.train_sampler,
            num_workers=1, pin_memory=True
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=self.hparams.batch_size, sampler=self.valid_sampler,
            num_workers=1, pin_memory=True
        )
        return valid_loader
