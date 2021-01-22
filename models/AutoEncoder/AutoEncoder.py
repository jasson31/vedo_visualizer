import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from models.AutoEncoder.layer import *


class AutoEncoder(pl.LightningModule):
    def __init__(self, hparams):
        super(AutoEncoder, self).__init__()
        if type(hparams) == dict:
            from argparse import Namespace
            hparams = Namespace(**hparams)
        self.hparams = hparams

        self.register_buffer('n_grid', torch.tensor(hparams.n_grid))
        self.register_buffer('grid_min', torch.tensor(hparams.grid_min))
        self.register_buffer('dx', torch.tensor(hparams.dx))
        self.register_buffer('gravity', torch.tensor([0, -9.81, 0]))

        grid_pos = list()
        D, W, H = hparams.n_grid
        indexes = itertools.product(range(D), range(W), range(H))
        for index in indexes:
            g_idx = torch.tensor(index, device=self.grid_min.device) \
                    + torch.tensor([0.5, 0.5, 0.5], device=self.grid_min.device)
            g_pos = self.grid_min + g_idx * self.dx
            grid_pos.append(g_pos)
        grid_pos = torch.stack(grid_pos, dim=0)
        self.register_buffer('grid_pos', grid_pos)

        # self.encoder = OurConvTranspose3d(3, 8, self.n_grid, self.grid_min, self.dx, bias=False)
        # self.decoder = OurConv3d(8, 3, self.n_grid, self.grid_min, self.dx, bias=False)
        self.encoder = CConvEncoder(3, 64, self.n_grid, self.grid_pos, self.dx)
        self.decoder = CConvDecoder(64, 3, self.n_grid, self.grid_pos, self.dx)

    def forward(self, data, acc_m=0, acc_std=1):
        pos, vel, acc = data
        # b_vel = torch.zeros_like(box)

        # particle_feats = torch.cat([pos, vel], dim=-1)  # [B, N, 6]
        # boundary_feats = torch.cat([box, b_vel], dim=-1)  # [B, M, 6]

        # input_acc = acc - self.gravity
        input_acc_norm = (acc - acc_m) / acc_std

        grid_feats = self.encoder(input_acc_norm, pos)  # [B, 6, D, H, W]

        grid_feats = F.leaky_relu(grid_feats)

        out_acc = self.decoder(grid_feats, pos)  # [B, N, 3]
        out_acc_denorm = out_acc * acc_std + acc_m

        pr_acc = out_acc_denorm  # + self.gravity

        return pr_acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-7, weight_decay=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        data0 = batch['data0']
        stat = batch['stat']

        acc_stat = stat[2][0]
        acc_m = acc_stat[:, 0]
        acc_std = acc_stat[:, 1]

        if len(data0) == 5:
            pos, vel, acc, box, box_normal = data0
        else:
            pos, vel, acc, obs, obs_normal, box, box_normal = data0

        pr_acc = self.forward((pos, vel, acc), acc_m, acc_std)
        loss = F.mse_loss(pr_acc, acc)

        self.log_dict({'train_loss': loss})

        return loss

    def validation_step(self, batch, batch_idx):
        data0 = batch['data0']

        if len(data0) == 5:
            pos, vel, acc, box, box_normal = data0
        else:
            pos, vel, acc, obs, obs_normal, box, box_normal = data0

        pr_acc = self.forward((pos, vel, acc))
        loss = F.mse_loss(pr_acc, acc)

        self.log_dict({'val_loss': loss})

        return loss


if __name__ == '__main__':
    from argparse import Namespace

    hparams = Namespace()
    hparams.n_grid = [20, 20, 20]
    hparams.grid_min = [0.0, 0.0, 0.0]
    hparams.dx = 0.05

    model = AutoEncoder(hparams).to('cuda')

    pos = torch.rand(1, 1000, 3, device='cuda')
    vel = torch.rand_like(pos)
    box = torch.zeros(0)
    data = (pos, vel, box)

    pr_vel = model(data)
    print(pr_vel.shape)