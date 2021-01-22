import torch
import torch.optim as optim
import pytorch_lightning as pl


class Naive(pl.LightningModule):
    def __init__(self, hparams):
        super(Naive, self).__init__()
        if type(hparams) == dict:
            from argparse import Namespace
            hparams = Namespace(**hparams)
        self.hparams = hparams

        self.pressure_acc = None
        self.b_pressure_acc = None
        self.viscosity_acc = None

    def forward(self, data):
        pos, vel, box = data
        p_acc = self.pressure_acc(pos, vel, box)
        b_p_acc = self.b_pressure_acc(pos, vel, box)
        total_p_acc = p_acc + b_p_acc

        v_acc = self.viscosity_acc(pos, vel, box)

        new_vel = vel + self.dt * (self.gravity + total_p_acc + v_acc)
        new_pos = pos + self.dt * new_vel

        return new_pos, new_vel

    def training_step(self, batch, batch_idx):
        data0 = batch['data0']
        data1 = batch['data1']

        pos0, vel0, box0 = data0
        pos1, vel1, box1 = data1

        pr_pos1, pr_vel1 = self.forward((pos0, vel0, box0))

        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
