# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""

"""Import all necessary packages"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

def Dataset_Load(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    train_transform,
    test_transform,
    batch_size: int,
    num_workers: int
):
    """Creates training, validation and testing DataLoaders.

    Takes in a training directory, validation directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Besides, take all true labels in testing dataset to make confusion matrix

    Args:
    train_dir: Path to training directory.
    val_dir: Path to training directory.
    test_dir: Path to testing directory.
    train_transform: torchvision transforms to perform on training data.
    test_transform: torchvision transforms to perform on validation and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

    Returns:
    A tuple of (train_dataloader, test_dataloader, class_names, targets).
    Where class_names is a list of the target classes.

    Example usage:
      train_dataloader, val_dataloader, test_dataloader, class_names, targets = \
                                                                    = create_dataloaders(train_dir=path/to/train_dir,
                                                                                             test_dir=path/to/test_dir,
                                                                                             transform=some_transform,
                                                                                             batch_size=32,
                                                                                             num_workers=4)
    """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=test_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
    )
    val_dataloader = DataLoader(
      val_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
    )
    test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
    )

    # Take true labels of testing dataset
    test_targets = torch.Tensor([target for _, target in test_data])
    print(class_names)

    return train_dataloader, val_dataloader, test_dataloader, class_names, test_targets
