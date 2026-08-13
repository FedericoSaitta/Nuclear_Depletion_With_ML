import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "none": nn.Identity,
}

LOSSES = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.HuberLoss,
    "smooth_l1": nn.SmoothL1Loss,
}


def _lookup(registry, name, kind):
    key = name.lower()
    if key not in registry:
        raise ValueError(f"Unknown {kind} {name!r}. Choose one of {sorted(registry)}.")
    return registry[key]()


def get_activation(activation):
    """Build the activation *activation* names.

    Unknown names raise: a typo used to fall back to ReLU, so a config asking
    for `gelu` and getting `relu` trained happily and reported nothing amiss.
    """
    return _lookup(ACTIVATIONS, activation, "activation")


def get_loss_fn(loss_name):
    return _lookup(LOSSES, loss_name, "loss")
