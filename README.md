# SmaAt-UNet
Code for the Paper "SmaAt-UNet: Precipitation Nowcasting using a Small Attention-UNet Architecture" [Arxiv-link](https://arxiv.org/abs/2007.04417)

![SmaAt-UNet](SmaAt-UNet.png)

The proposed SmaAt-UNet can be found in the model-folder under [SmaAt_UNet](models/SmaAt_UNet.py).

---
For the paper we used the [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) -module (PL) which simplifies the training process and allows easy additions of loggers and checkpoint creations.
In order to use PL we created the model [UNetDS_Attention](models/unet_precip_regression_lightning.py) whose parent inherits from the pl.LightningModule. This model is the same as the pure PyTorch SmaAt-UNet implementation with the added PL functions.

### Training
An example [training script](train_SmaAtUNet.py) is given for a classification task (PascalVOC).

For training on the precipitation task we used the [train_precip_lightning.py](train_precip_lightning.py) file. 

### Precipitation dataset
The dataset consists of precipitation maps in 5-minute intervals from 2016-2019 resulting in about 420,000 images.

The dataset is based on radar precipitation maps from the [The Royal Netherlands Meteorological Institute (KNMI)](https://www.knmi.nl/over-het-knmi/about).
The original images were cropped as can be seen in the example below:
![Precip cutout](Precipitation%20map%20Cutout.png)

If you are interested in the dataset that we used please write me an e-mail: k.trebing@student.maastrichtuniversity.nl

The 50% dataset has 4GB in size and the 20% dataset has 16.5GB in size.
### Citation   
```
@misc{trebing2020smaatunet,
    title={SmaAt-UNet: Precipitation Nowcasting using a Small Attention-UNet Architecture},
    author={Kevin Trebing and Siamak Mehrkanoon},
    year={2020},
    eprint={2007.04417},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```   
