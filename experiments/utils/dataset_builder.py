import torch
import numpy as nn
import sys
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

sys.path.insert(0, os.path.join(sys.path[0], '../..'))

from dataloaders import MnistDataloader, FashionMnistDataloader


class NoisedCTDataset(Dataset):
    def __init__(self, x, b):
        self.x = x
        self.x_noise = torch.normal(self.x * (1-b)**(0.5), b**(0.5))
        self.y = torch.from_numpy(np.concatenate((np.zeros(self.x.shape[0]), np.ones(self.x.shape[0])), axis=0))
        self.x = torch.cat((self.x, self.x_noise), 0)
        self.y_hot = F.one_hot(self.y.long(), num_classes=2).to(float)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, i):
        return self.x[i], self.y_hot[i]

def save(dataset, datasetid, i, is_train):
    if is_train: 
        num = "{:04d}".format(i)
        filename = f"cache/dataset_cache/{datasetid}/train/{num}.pt"
        torch.save(dataset, filename)
    else:
        num = "{:04d}".format(i)
        filename = f"cache/dataset_cache/{datasetid}/test/{num}.pt"
        torch.save(dataset, filename)

def build_datasets(x_train, x_test, noise_schedule, datasetid):
    
    # Initializing x_train data, dequantizing and normalizing our original set
    x_train_torch = torch.from_numpy(np.array(x_train))
    x_train_norm = (x_train_torch + torch.rand(x_train_torch.shape)) / 255.
    x_test_torch = torch.from_numpy(np.array(x_test))
    x_test_norm = (x_test_torch + torch.rand(x_test_torch.shape)) / 255.

    # Populating our list of x_train datasets, each with added 50% from previous set and 50% with added guassian noise
    print('Initializing set ' + str(1))
    curr_train_set = NoisedCTDataset(x_train_norm, noise_schedule[0])
    curr_test_set = NoisedCTDataset(x_test_norm, noise_schedule[0])
    save(curr_train_set, datasetid, 0, True)
    save(curr_test_set, datasetid, 0, False)
    for i in range(1, len(noise_schedule)):
        print('Initializing set ' + str(i+1))
        curr_train_set = NoisedCTDataset(curr_train_set.x_noise, noise_schedule[i])
        curr_test_set = NoisedCTDataset(curr_test_set.x_noise, noise_schedule[i])
        save(curr_train_set, datasetid, i, True)
        save(curr_test_set, datasetid, i, False)

def get_dataset(name, datasetid, num_steps, noise_start, noise_end):

    noise_schedule = torch.linspace(noise_start, noise_end, num_steps)

    if name == 'mnist':
        dataloader = MnistDataloader()
        x_train, x_test = dataloader.load_data()
        build_datasets(x_train, x_test, noise_schedule, datasetid)
    elif name == 'fashion_mnist':
        dataloader = FashionMnistDataloader()
        x_train, x_test = dataloader.load_data()
        build_datasets(x_train, x_test, noise_schedule, datasetid)
    else:
        raise ValueError(f'Unknown dataset {name}')
        
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


