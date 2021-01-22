import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
from torch_scatter import scatter


def MLP(channels, bn=True):
    module_list = []
    for i in range(1, len(channels)):
        module_list.append(nn.Linear(channels[i - 1], channels[i]))
        if bn:
            module_list.append(nn.BatchNorm1d(channels[i]))
        module_list.append(nn.ReLU())

    return nn.Sequential(*module_list)


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = pyg.nn.PointConv(nn)

    def forward(self, x, pos, batch):
        idx = pyg.nn.fps(pos, batch, ratio=self.ratio)
        row, col = pyg.nn.radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = pyg.nn.global_mean_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = pyg.nn.knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip
