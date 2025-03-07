import torch, ukko
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List, Union

# Parameters
batch_size = 32
n_samples = 1000
n_features = 3
sequence_length = 32
prediction_length = 5

def create_datasets(n_samples = 1000, n_features = 3, sequence_length = 32, prediction_length = 5, base_freq=0.1):
    """
    Generate training, test, and validation SineWaveDataset's

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    n_val = n_samples//5
    dataset = ukko.data.SineWaveDataset(n_samples+2*n_val, n_features, sequence_length, prediction_length, base_freq=base_freq)
    
    # Create indices for each split
    train_indices = list(range(n_samples))
    val_indices = list(range(n_samples, n_samples + n_val))
    test_indices = list(range(n_samples + n_val, n_samples + 2*n_val))
    
    train_dataset = dataset.Subset(train_indices)
    val_dataset = dataset.Subset(val_indices)
    test_dataset = dataset.Subset(test_indices)
    
    # print(
    #     train_indices[-1],
    #     val_indices[0],
    #     val_indices[-1],
    #     test_indices[0]
    # )
    return train_dataset, val_dataset, test_dataset
    


def create_dataloaders(datasets, batch_size=batch_size):

    if len(datasets) == 3:
        train_dataset, val_dataset, test_dataset = datasets
    else:
        print(f"Warning: Expected 3 datasets, but got {len(datasets)}")
    
    train_dataset, val_dataset, test_dataset = datasets
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader



def create_dataloaders2(
    datasets: Union[Tuple[Dataset, Dataset, Dataset], List[Dataset]], 
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test datasets.

    Args:
        datasets: A tuple or list containing exactly 3 PyTorch Dataset objects 
                 in the order (train_dataset, val_dataset, test_dataset)
        batch_size: Number of samples per batch. Defaults to 32.
        num_workers: Number of subprocesses for data loading. Defaults to 0.
        pin_memory: If True, copies tensors into CUDA pinned memory. Defaults to True.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader)

    Raises:
        ValueError: If datasets doesn't contain exactly 3 Dataset objects.
        TypeError: If any dataset is not a PyTorch Dataset instance.
    """
    # Validate input
    if not isinstance(datasets, (tuple, list)):
        raise TypeError(f"Expected tuple or list, got {type(datasets)}")
    
    if len(datasets) != 3:
        raise ValueError(f"Expected 3 datasets, but got {len(datasets)}")
    
    # Unpack datasets
    train_dataset, val_dataset, test_dataset = datasets
    
    # Validate each dataset
    for i, dataset in enumerate(['train', 'validation', 'test']):
        if not isinstance(datasets[i], Dataset):
            raise TypeError(f"{dataset} dataset must be a PyTorch Dataset instance")
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader