import torch
import torch.nn as nn 
import numpy as np

def get_prediction(test_set, i, f, use_wandb):

    if use_wandb:
        import wandb
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    L = torch.nn.CrossEntropyLoss()
    
    f = f.to(device)

    inputs = test_set.x.type(dtype).to(device)
    inputs = inputs.reshape([-1, 1, 28, 28])
    expected = test_set.y.type(dtype).to(device)
    activations, outputs = f(inputs)

    outputs_argmax = outputs.argmax(axis=1)
    accuracy = torch.sum(outputs_argmax == expected)/len(expected)
    # validation_loss = L(outputs, expected)
    if use_wandb: 
        wandb.log({
        f"accuracy model {i}": accuracy, 
        f"validation loss model {i}": 0})

    return outputs

def get_density_ratio(outputs):

    softmax = nn.Softmax(dim=1)
    yhats_soft = softmax(outputs)
    density_ratio = torch.reciprocal(yhats_soft[:,0].detach())-1
    return density_ratio

def get_log_likelihood(prod, final_test, use_wandb):

    if use_wandb:
        import wandb

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    softmax = nn.Softmax(dim=1)

    # P(x)/Q(x) = 1/(P(x in Q(x)))-1
    prod.type(dtype).to(device)

    # we end up with a distribution!
    pure_noise_distribution = torch.distributions.normal.Normal(torch.from_numpy(np.full((28, 28), 0.)), torch.from_numpy(np.full((28, 28), 1.)))
    final_xs = final_test.x
    p_xn = pure_noise_distribution.log_prob(final_xs)

    # sum over pixels
    p_xn = torch.sum(torch.sum(p_xn, 1), 1)
    p_xn = p_xn.type(dtype).to(device)

    # get total likelihood
    p_x0 = torch.log(prod) + p_xn
    avg_likelihood = torch.mean(p_x0).item()

    if use_wandb:
        wandb.log({"total_likelihood": avg_likelihood})

    return avg_likelihood