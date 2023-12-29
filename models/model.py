from .RDN import ResidualDenseNetwork
from .upscale import UpSample
import numpy as np
import torch.nn as nn


def get_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def get_model(
        D, 
        C, 
        G,
        ratio,
        verbose=False
    ):
    rdn = ResidualDenseNetwork(D=D, C=C, G=G)
    up = UpSample(ratio=ratio)
    if verbose:
        print("<<model info>>")
        print(rdn)
        print(up)
        print("parameter num :", get_parameters(rdn)+get_parameters(up))
    return nn.Sequential(
        rdn,
        up
    )
