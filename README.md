# SmaAt-UNet
Code for the Paper "SmaAt-UNet: Precipitation Nowcasting using a Small, Attentive UNetArchitecture"

The proposed model can be found in the model-folder under [SmaAt_UNet](models/SmaAt_UNet.py). 

---
For the paper we used the [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) -module (PL) which simplifies the training process and allows easy additions of loggers and checkpoint creations.
In order to use PL we created the model [UNetDS_Attention](models/unet_precip_regression_lightning.py) whose parent inherits from the pl.LightningModule. This model is the same as the pure PyTorch SmaAt-UNet implementation with the added PL functions.