import torch
import numpy as np
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from utils import dataset_precip


# Taken from: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
def get_train_valid_loader(
    data_dir,
    batch_size,
    random_seed,
    num_input_images,
    num_output_images,
    augment,
    classification,
    valid_size=0.1,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    # Since I am not dealing with RGB images I do not need this
    # normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2023, 0.1994, 0.2010],
    # )

    # define transforms
    valid_transform = None
    # valid_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    # ])
    if augment:
        # TODO flipping, rotating, sequence flipping (torch.flip seems very expensive)
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # normalize,
            ]
        )
    else:
        train_transform = None
        # train_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     normalize,
        # ])
    if classification:
        # load the dataset
        train_dataset = dataset_precip.precipitation_maps_classification_h5(
            in_file=data_dir,
            num_input_images=num_input_images,
            img_to_predict=num_output_images,
            train=True,
            transform=train_transform,
        )

        valid_dataset = dataset_precip.precipitation_maps_classification_h5(
            in_file=data_dir,
            num_input_images=num_input_images,
            img_to_predict=num_output_images,
            train=True,
            transform=valid_transform,
        )
    else:
        # load the dataset
        train_dataset = dataset_precip.precipitation_maps_h5(
            in_file=data_dir,
            num_input_images=num_input_images,
            num_output_images=num_output_images,
            train=True,
            transform=train_transform,
        )

        valid_dataset = dataset_precip.precipitation_maps_h5(
            in_file=data_dir,
            num_input_images=num_input_images,
            num_output_images=num_output_images,
            train=True,
            transform=valid_transform,
        )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader


def get_test_loader(
    data_dir,
    batch_size,
    num_input_images,
    num_output_images,
    classification,
    shuffle=False,
    num_workers=4,
    pin_memory=False,
):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    # Since I am not dealing with RGB images I do not need this
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    # )

    # define transform
    transform = None
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     # normalize,
    # ])
    if classification:
        dataset = dataset_precip.precipitation_maps_classification_h5(
            in_file=data_dir,
            num_input_images=num_input_images,
            img_to_predict=num_output_images,
            train=False,
            transform=transform,
        )
    else:
        dataset = dataset_precip.precipitation_maps_h5(
            in_file=data_dir,
            num_input_images=num_input_images,
            num_output_images=num_output_images,
            train=False,
            transform=transform,
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader


if __name__ == "__main__":
    folder = "C:/Users/hans-/Documents/weather_prediction/data/precipitation/"
    data = "RAD_NL25_RAC_5min_train_test_2016-2019.h5"
    train_dl, valid_dl = get_train_valid_loader(
        folder + data,
        batch_size=8,
        random_seed=1337,
        num_input_images=12,
        num_output_images=6,
        classification=True,
        augment=False,
        valid_size=0.1,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
    for xb, yb in train_dl:
        print("xb.shape: ", xb.shape)
        print("yb.shape: ", yb.shape)
        break
