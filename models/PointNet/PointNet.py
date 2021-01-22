import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from models.PointNet.layer import *


class PointNet(pl.LightningModule):
    def __init__(self, hparams):
        super(PointNet, self).__init__()
        if type(hparams) == dict:
            from argparse import Namespace
            hparams = Namespace(**hparams)
        self.hparams = hparams

        self.register_buffer('gravity', torch.tensor([0, -9.81, 0]))

        self.sa1 = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64, 128]))
        self.sa2 = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3 = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3 = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2 = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1 = FPModule(3, MLP([128 + 3, 128, 128, 128]))

        self.last_mlp = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=0.5),
                                      nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=0.5),
                                      nn.Linear(128, 3))

    def forward(self, data, acc_m=0, acc_std=1):
        pos, vel, acc = data
        # b_vel = torch.zeros_like(box)

        # particle_feats = torch.cat([pos, vel], dim=-1)  # [B, N, 6]
        # boundary_feats = torch.cat([box, b_vel], dim=-1)  # [B, M, 6]

        # input_acc = acc - self.gravity
        input_acc_norm = (acc - acc_m) / acc_std

        input_x = input_acc_norm.reshape(-1, input_acc_norm.shape[-1])
        input_pos = pos.reshape(-1, pos.shape[-1])
        input_batch = torch.arange(pos.shape[0], device=self.device).repeat_interleave(pos.shape[1])

        input = (input_x, input_pos, input_batch)

        sa1_out = self.sa1(*input)
        sa2_out = self.sa2(*sa1_out)
        sa3_out = self.sa3(*sa2_out)

        fp3_out = self.fp3(*sa3_out, *sa2_out)
        fp2_out = self.fp2(*fp3_out, *sa1_out)
        x, _, _ = self.fp1(*fp2_out, *input)

        out_acc = self.last_mlp(x)
        out_acc_denorm = out_acc * acc_std + acc_m

        pr_acc = out_acc_denorm.reshape(acc.shape)  # + self.gravity

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

    model = PointNet(hparams).to('cuda')

    pos = torch.rand(1, 1000, 3, device='cuda')
    vel = torch.rand_like(pos)
    box = torch.zeros(0)
    data = (pos, vel, box)

    pr_vel = model(data)
    print(pr_vel.shape)