import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np

def store_gradients(model, step, use_wandb=True):
    if use_wandb:
        import wandb
        
    l = [module for module in list(model.named_modules())[1:] if isinstance(module[1], nn.Linear)]
    for name, layer in l:
        weight = torch.linalg.norm(layer.weight)
        bias = torch.linalg.norm(layer.bias)
        if use_wandb:
            wandb.log({f"{name} weights": weight.item()})
            wandb.log({f"{name} bias": bias.item()})

def train_basic(dl, f, model_num, use_wandb, n_epochs):

    if use_wandb:
        import wandb

    # Optimization
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    print(f"iscuda: {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    f = f.to(device)

    # Train model
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        total_loss = 0
        N = len(dl)
        for i, (x, y) in enumerate(dl):
            x, y = x.type(dtype).to(device), y.type(dtype).to(device)
            # Update the weights of the network
            opt.zero_grad() 
            loss_value = L(f(x), y) 
            loss_value.backward() 
            opt.step()
            total_loss += loss_value.item()
        average_loss = total_loss / N
        if use_wandb:
            wandb.log({
            "model": model_num,
            "model_step": epoch,
            "loss": average_loss})
    return f

def train_resnet(dl, f, model_num, use_wandb, n_epochs):

    if use_wandb:
        import wandb

    # Optimization
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    f = f.to(device)

    # Train model
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        total_loss = 0
        N = len(dl)
        for i, (inputs, targets) in enumerate(dl):
            # Update the weights of the network
            inputs, targets = inputs.type(dtype).to(device), targets.type(dtype).to(device)
            inputs = inputs.reshape([-1, 1, 28, 28])
            activations, outputs = f(inputs)
            loss = L(outputs, targets)
            opt.zero_grad() 
            tensor_loss = loss.mean()
            tensor_loss.backward() 
            opt.step()
            total_loss += tensor_loss.item()
        average_loss = total_loss / N
        if use_wandb:
            wandb.log({
            "model": model_num,
            "loss": average_loss}, step=epoch)
    return f

def train_model(model_type, dl, model, model_num, use_wandb, n_epochs):
    f = None
    if model_type == 'basic':
        f = train_basic(dl, model, model_num, use_wandb, n_epochs)
    elif model_type == 'resnet50' or model_type == 'resnet101' or model_type == 'resnet152':
        f = train_resnet(dl, model, model_num, use_wandb, n_epochs)
    return f

# Used to train a series of models
def train_models(train_dls, f_set, model_type, use_wandb=False, store_gradient=False):

    epoch_data_set = []
    loss_data_set = []
    new_f_set = []
    for i, model in enumerate(f_set):
        print("Training model differentiating set " + str(i) + " and set " + str(i+1))
        f = train_model(model_type, train_dls[i], model, i, use_wandb, 5)
        new_f_set.append(f)
        if(store_gradients):
            store_gradients(f, i, use_wandb=use_wandb)
    return new_f_set