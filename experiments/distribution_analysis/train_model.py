import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(sys.path[0], '../..'))

from experiments.utils.dataset_builder import NoisedCTDataset
from experiments.utils.general_utils import seed_worker
from scripts import train_model
from models import BasicNeuralNet, ResNet50, ResNet101

def get_model(name):
    if name == 'basic':
        return BasicNeuralNet()
    elif name == 'resnet50':
        return ResNet50()
    elif name == 'resnet101':
        return ResNet101()
    else:
        raise ValueError(f'Unknown dataset {name}')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Distribution Analysis")
    parser.add_argument("--modelid", type=str, 
                        help="model id for this run")
    parser.add_argument("--datasetid", type=str, 
                        help="dataset id that we're training on")
    parser.add_argument("--set", type=int, 
                    help="number of dataset we want to train")   
    parser.add_argument("--model", default="resnet101", type=str, 
                        help="basic|resnet50|resnet101")         
    parser.add_argument("--n_epochs", default=10, type=int, 
                        help="number of epochs to train")        
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--turnoff_wandb', action="store_true")
    args = parser.parse_args()
    
    use_wandb = not args.turnoff_wandb

    if use_wandb:
        import wandb
        wandb.init(project='distribution_analysis', name=f'train_model-{args.modelid}_dataset-{args.datasetid}')

    num = "{:04d}".format(args.set)
    filename = f"cache/dataset_cache/{args.datasetid}/train/{num}.pt"
    train_set = torch.load(filename)

    print(f"Loaded training dataset {args.set}..")

    g = torch.Generator()
    g.manual_seed(0)

    train_generator = DataLoader(train_set,
                                batch_size=10,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True,
                                drop_last=True,
                                worker_init_fn=seed_worker,
                                generator=g)
    
    print("Created dataloader..")

    torch.manual_seed(args.seed)

    model = get_model(args.model)
    model = train_model(args.model, train_generator, model, args.set, use_wandb, args.n_epochs)

    print(f"Trained model {args.set}")
    
    os.makedirs("cache", exist_ok=True)
    os.makedirs("cache/model_cache", exist_ok=True)
    os.makedirs(f"cache/model_cache/{args.modelid}", exist_ok=True)

    filename = f"cache/model_cache/{args.modelid}/{num}.pt"
    torch.save(model.state_dict(), filename)

    print(f"Saved model {args.set}")