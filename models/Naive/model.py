import torch
import torch.nn as nn


class OurPhysicsNetwork(nn.Module):
    def __init__(self, hparams):
        super(OurPhysicsNetwork, self).__init__()
        
