import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger, EarlyStopping
from pytorch_lightning import loggers
import argparse
from models import unet_precip_regression_lightning as unet_regr
import torchsummary
import os


def get_batch_size(hparams):
    if hparams.model == "UNetDS_Attention":
        net = unet_regr.UNetDS_Attention(hparams=hparams)
    elif hparams.model == "UNet_Attention":
        net = unet_regr.UNet_Attention(hparams=hparams)
    elif hparams.model == "UNet":
        net = unet_regr.UNet(hparams=hparams)
    elif hparams.model == "UNetDS":
        net = unet_regr.UNetDS(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")

    trainer = pl.Trainer(gpus=hparams.gpus)
    new_batch_size = trainer.scale_batch_size(net, mode='binsearch', init_val=8)
    print("New biggest batch_size: ", new_batch_size)
    return new_batch_size


def train_regression(hparams):
    if hparams.model == "UNetDS_Attention":
        net = unet_regr.UNetDS_Attention(hparams=hparams)
    elif hparams.model == "UNet_Attention":
        net = unet_regr.UNet_Attention(hparams=hparams)
    elif hparams.model == "UNet":
        net = unet_regr.UNet(hparams=hparams)
    elif hparams.model == "UNetDS":
        net = unet_regr.UNetDS(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")

    torchsummary.summary(net, (12, 288, 288), device="cpu")
    # return
    default_save_path = "lightning/precip_regression"

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd()+"/"+default_save_path+"/"+net.__class__.__name__+"/{epoch}-{val_loss:.6f}",
        save_top_k=-1,
        verbose=False,
        monitor='val_loss',
        mode='min',
        prefix=net.__class__.__name__+"_rain_threshhold_50_"
    )
    lr_logger = LearningRateLogger()
    tb_logger = loggers.TensorBoardLogger(save_dir=default_save_path, name=net.__class__.__name__)

    earlystopping_callback = EarlyStopping(monitor='val_loss',
                                           mode='min',
                                           patience=hparams.es_patience,  # is effectively half (due to a bug in pytorch-lightning)
                                           )
    trainer = pl.Trainer(fast_dev_run=hparams.fast_dev_run,
                         gpus=hparams.gpus,
                         weights_summary=None,
                         max_epochs=hparams.epochs,
                         default_save_path=default_save_path,
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=earlystopping_callback,
                         logger=tb_logger,
                         callbacks=[lr_logger],
                         resume_from_checkpoint=hparams.resume_from_checkpoint,
                         val_check_interval=hparams.val_check_interval,
                         overfit_pct=hparams.overfit_pct)
    trainer.fit(net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = unet_regr.Precip_regression_base.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--dataset_folder',
                        default='data/precipitation/RAD_NL25_RAC_5min_train_test_2016-2019.h5', type=str)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)

    args = parser.parse_args()

    # args.fast_dev_run = True
    args.n_channels = 12
    args.gpus = 1
    args.model = "UNetDS_Attention"
    args.lr_patience = 4
    args.es_patience = 30
    # args.val_check_interval = 0.25
    # args.overfit_pct = 0.1
    args.kernels_per_layer = 2
    args.use_oversampled_dataset = True
    args.dataset_folder = "data/precipitation/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5"
    # args.resume_from_checkpoint = f"lightning/precip_regression/{args.model}/UNetDS_Attention.ckpt"

    # args.batch_size = get_batch_size(hparams=args)
    train_regression(args)

    # for m in ["UNet", "UNetDS", "UNet_Attention", "UNetDS_Attention"]:
    #     args.model = m
    #     train_regression(args)
