import numpy as np
import torch


def get_batch(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fetches a random batch of input-target pairs from the dataset for training.

    Args:
        dataset: The dataset to sample from.
        batch_size: Number of samples in the batch.
        context_length: Length of the input sequences.
        device: The device to load the tensors onto.
    Returns:
        A tuple where the first item
        is the sampled input sequences, and the second item is the corresponding
        language modeling labels.
    """
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)

    inputs = np.stack([dataset[start : start + context_length] for start in starts])
    targets = np.stack([dataset[start + 1 : start + 1 + context_length] for start in starts])
    inputs_tensor = torch.from_numpy(inputs).long().to(device)
    targets_tensor = torch.from_numpy(targets).long().to(device)

    return inputs_tensor, targets_tensor

# 5.2 checkpointing
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str) -> None:
    """
    Saves the model checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state to save.
        epoch: The current epoch number.
        path: The file path to save the checkpoint to.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch} to {path}")


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str) -> int:
    """
    Loads the model checkpoint.

    Args:
        model: The model to load the state into.
        optimizer: The optimizer to load the state into.
        path: The file path to load the checkpoint from.
    Returns:
        The epoch number from which the checkpoint was loaded.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {path} at epoch {epoch}")
    return epoch
