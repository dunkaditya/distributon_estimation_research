import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(sys.path[0], '../..'))

from experiments.utils.dataset_builder import NoisedCTDataset
from experiments.utils.log_likelihoods import get_log_likelihood
from models import BasicNeuralNet, ResNet50

def get_result(task, final_test, prod, use_wandb=False):
    if task == 'default':
        return get_log_likelihood(prod, final_test, use_wandb)
    else:
        raise ValueError(f'Unknown task {name}')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Distribution Analysis")
    parser.add_argument("--modelid", type=str, 
                        help="modelid of test run")
    parser.add_argument("--datasetid", type=str, 
                        help="datasetid of test run")
    parser.add_argument("--size", type=int, 
                        help="size of test run") 
    parser.add_argument("--task", default="default", type=str, 
                    help="default") 
    parser.add_argument('--turnoff_wandb', action="store_true")
    args = parser.parse_args()
    
    use_wandb = not args.turnoff_wandb

    if use_wandb:
        import wandb
        wandb.init(project='distribution_analysis', name=f'results_model-{args.modelid}_dataset-{args.datasetid}')

    prod = torch.load(f"cache/results_cache/ratio_prod/{args.modelid}_{args.datasetid}.pt")
    print(f"Loaded ratio product: {prod}")

    num = "{:04d}".format(args.size-1)
    final_test = torch.load(f"cache/dataset_cache/{args.datasetid}/test/{num}.pt")

    result = get_result(args.task, final_test, prod, use_wandb)

    print(f"Found results.. {result}")