import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler


def shuffle_loader(data, shuffle_dataset=True, random_seed=42, batch_size=16):

    dataset_size = len(data)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Creating PT data samplers and loaders:
    sampler = SubsetRandomSampler(indices)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=sampler)

    return loader
