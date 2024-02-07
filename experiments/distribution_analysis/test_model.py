import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(sys.path[0], '../..'))

from experiments.utils.dataset_builder import NoisedCTDataset
from experiments.utils.log_likelihoods import get_prediction, get_density_ratio
from models import BasicNeuralNet, ResNet50, ResNet101

def get_result(task, test_sets, models, use_wandb=False):
    if task == 'default':
        return get_results(test_sets, models, use_wandb)
    else:
        raise ValueError(f'Unknown task {name}')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Distribution Analysis")
    parser.add_argument("--modelid", type=str, 
                        help="model id we're testing")
    parser.add_argument("--datasetid", type=str, 
                        help="dataset id we're testing on")
    parser.add_argument("--set", type=int, 
                    help="set and model number we want to test")        
    parser.add_argument('--turnoff_wandb', action="store_true")
    args = parser.parse_args()
    
    use_wandb = not args.turnoff_wandb

    if use_wandb:
        import wandb
        wandb.init(project='distribution_analysis', name=f'test_model-{args.modelid}_dataset-{args.datasetid}')

    num = "{:04d}".format(args.set)
    test_set = torch.load(f"cache/dataset_cache/{args.datasetid}/test/{num}.pt")

    print(f"Loading testing dataset {str(args.set)}..")

    model = ResNet101()
    model.load_state_dict(torch.load(f"cache/model_cache/{args.modelid}/{num}.pt"))

    outputs = get_prediction(test_set, args.set, model, use_wandb)
    density_ratio = get_density_ratio(outputs)

    print(f"Tested model {str(args.set)}..")

    os.makedirs("cache", exist_ok=True)
    os.makedirs("cache/results_cache", exist_ok=True)
    os.makedirs("cache/results_cache/ratio_prod", exist_ok=True)

    prod = None
    filename = f"cache/results_cache/ratio_prod/{args.modelid}_{args.datasetid}.pt"
    if os.path.isfile(filename):
        prod = torch.load(filename)
        print(f"Density ratio {args.set}: {density_ratio}")
        prod += density_ratio
        print(f"Prod {args.set}: {prod}")
    else:
        prod = density_ratio
    torch.save(prod, filename)

    print(f"Saved density ratio prod, model {str(args.set)}..")

