import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(sys.path[0], '../..'))

from experiments.utils.dataset_builder import get_dataset

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Distribution Analysis")
    parser.add_argument("--datasetid", type=str, 
                        help="dataset id for this run")
    parser.add_argument("--dataset", default="mnist", type=str, 
                        help="mnist|fashion_mnist")
    parser.add_argument("--size", default=5, type=int,
                        help="number of classifiers")
    parser.add_argument("--noise_start", default=0.0001, type=float,
                        help="start of noise schedule")
    parser.add_argument("--noise_end", default=0.02, type=float,
                        help="end of noise schedule")
    args = parser.parse_args()
    
    os.makedirs("cache", exist_ok=True)
    os.makedirs("cache/dataset_cache", exist_ok=True)
    os.makedirs(f'cache/dataset_cache/{args.datasetid}', exist_ok=True)
    os.makedirs(f'cache/dataset_cache/{args.datasetid}/train', exist_ok=True)
    os.makedirs(f'cache/dataset_cache/{args.datasetid}/test', exist_ok=True)

    get_dataset(args.dataset, args.datasetid, args.size, args.noise_start, args.noise_end)

    print("Built and saved datasets..")