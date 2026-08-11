import torch.nn as nn


def get_activation(activation):
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "selu": nn.SELU(),
        "softplus": nn.Softplus(),
        "none": nn.Identity(),
    }
    return activations.get(activation.lower(), nn.ReLU())


def get_loss_fn(loss_name):
    losses = {
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
        "huber": nn.HuberLoss(),
        "smooth_l1": nn.SmoothL1Loss(),
    }
    return losses[loss_name.lower()]
