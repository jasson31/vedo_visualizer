import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from torch_scatter import scatter


# original implementation of differential mpm
class MpmP2G(nn.Module):
    def __init__(self, n_grid, grid_min, dx):
        super(MpmP2G, self).__init__()

        self.register_buffer('n_grid', torch.tensor(n_grid))
        self.register_buffer('grid_min', torch.tensor(grid_min))
        self.register_buffer('dx', torch.tensor(dx))

    def forward(self, input, pos):
        # input: [N, 3]
        # pos: [N, 3]
        # result: [D, H, W, 3]
        device = input.device
        grid_value = torch.zeros([*(self.n_grid+2), 3], device=device)
        grid_weight = torch.zeros([*(self.n_grid+2), 1], device=device)
        D, H, W, _ = grid_value.shape

        normalized_pos = (pos - self.grid_min) / self.dx
        grid_pos = normalized_pos.int()
        local_pos = normalized_pos - grid_pos
        weight_list = [0.5 * torch.pow((1.0 - local_pos), 2),
                       0.75 - torch.pow((0.5 - local_pos), 2),
                       0.5 * torch.pow(local_pos, 2)]
        grid_pos += 1  # padding

        offsets = itertools.product(range(3), range(3), range(3))
        for offset in offsets:
            offset = torch.tensor(offset, dtype=torch.int32, device=device)
            weight = weight_list[offset[0]][:, 0:1] * weight_list[offset[1]][:, 1:2] * weight_list[offset[2]][:, 2:3]
            grid_idx = (grid_pos + offset - 1).long()
            add_value = input * weight

            grid_idx_1d = grid_idx[..., 0] * H * W + grid_idx[..., 1] * W + grid_idx[..., 2]
            dim_size = D * H * W

            new_grid_val = scatter(add_value, grid_idx_1d, dim=0, dim_size=dim_size, reduce="sum")
            grid_value += new_grid_val.view(D, H, W, 3)

            new_grid_weight = scatter(weight, grid_idx_1d, dim=0, dim_size=dim_size, reduce="sum")
            grid_weight += new_grid_weight.view(D, H, W, 1)

        grid_weight[grid_weight == 0] = 1

        return grid_value / grid_weight


class MpMG2P(nn.Module):
    def __init__(self, n_grid, grid_min, dx):
        super(MpMG2P, self).__init__()

        self.register_buffer('n_grid', torch.tensor(n_grid))
        self.register_buffer('grid_min', torch.tensor(grid_min))
        self.register_buffer('dx', torch.tensor(dx))

    def forward(self, input, pos):
        # input: [D, H, W, 3]
        # pos: [N, 3]
        # result: [N, 3]
        device = input.device
        new_val = torch.zeros_like(pos, device=device)

        normalized_pos = (pos - self.grid_min) / self.dx
        grid_pos = normalized_pos.int()
        local_pos = normalized_pos - grid_pos
        weight_list = [0.5 * torch.pow((1.0 - local_pos), 2),
                       0.75 - torch.pow((0.5 - local_pos), 2),
                       0.5 * torch.pow(local_pos, 2)]
        grid_pos += 1  # padding

        offsets = itertools.product(range(3), range(3), range(3))
        for offset in offsets:
            offset = torch.tensor(offset, dtype=torch.int32, device=device)
            weight = weight_list[offset[0]][:, 0:1] * weight_list[offset[1]][:, 1:2] * weight_list[offset[2]][:, 2:3]
            grid_idx = (grid_pos + offset - 1).long()

            g_val = input[grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]]
            new_val += g_val * weight

        return new_val


# our layer to test
class OurConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, n_grid, grid_min, dx, bias=True):
        super(OurConvTranspose3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer('n_grid', n_grid)
        self.register_buffer('grid_min', grid_min)
        self.register_buffer('dx', dx)

        self.kernel = nn.Parameter(torch.Tensor(3, 3, 3, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

    def forward(self, input, pos):
        # input: [B, N, C_in]
        # pos: [B, N, 3]
        # result: [B, C_out, D, H, W]
        device = input.device
        B, N, _ = pos.shape

        result = torch.zeros([B, self.out_channels, *(self.n_grid + 2)], device=device)
        _, _, D, H, W = result.shape

        Xp = (pos - self.grid_min) / self.dx  # [B, N, 3]
        base = Xp.int()  # [B, N, 3]
        fx = (Xp - base) - 0.5  # [B, N, 3]
        w = torch.stack([0.5 * (0.5 - fx) ** 2,
                         0.75 - fx ** 2,
                         0.5 * (0.5 + fx) ** 2], dim=-2)  # [B, N, 3(s), 3(xyz)]
        base = base + 1  # add 1 to all indices for padding

        # TODO: make parallel
        offsets = itertools.product(range(3), range(3), range(3))
        for offset in offsets:
            offset = torch.tensor(offset, dtype=torch.int, device=device)
            weight = (w[..., offset[0], 0] * w[..., offset[1], 1] * w[..., offset[2], 2]).unsqueeze(-1)  # [B, N, 1]

            indices = (base + offset.view(1, 1, 3) - 1).long()  # [B, N, 3]
            indices_1d = indices[..., 0] * H * W + indices[..., 1] * W + indices[..., 2]  # [B, N]

            if self.has_bias:
                values = weight * (torch.matmul(input, self.kernel[offset[0], offset[1], offset[2]])
                                   + self.bias.view(1, 1, -1))  # [B, N, C_out]
            else:
                values = weight * (torch.matmul(input, self.kernel[offset[0], offset[1], offset[2]]))  # [B, N, C_out]

            out = scatter(values, indices_1d, dim=1, dim_size=D * H * W, reduce="sum")  # [B, D*H*W, C_out]
            out = out.view(B, D, H, W, self.out_channels).permute(0, 4, 1, 2, 3)  # [B, C_out, D, H, W]

            result += out

        return result


class OurConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, n_grid, grid_min, dx, bias=True):
        super(OurConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer('n_grid', n_grid)
        self.register_buffer('grid_min', grid_min)
        self.register_buffer('dx', dx)
        self.has_bias = bias

        # TODO: initialize parameters
        self.kernel = nn.Parameter(torch.Tensor(3, 3, 3, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input, pos):
        # input: [B, C_in, D + 2, H + 2, W + 2]
        # pos: [B, N, 3]
        # result: [B, N, C_out]
        device = input.device
        B, N, _ = pos.shape

        input = input.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C_in]
        result = torch.zeros(B, N, self.out_channels, device=pos.device)

        Xp = (pos - self.grid_min) / self.dx  # [B, N, 3]
        base = Xp.int()  # [B, N, 3]
        fx = (Xp - base) - 0.5  # [B, N, 3]
        w = torch.stack([0.5 * (0.5 - fx) ** 2,
                         0.75 - fx ** 2,
                         0.5 * (0.5 + fx) ** 2], dim=-2)  # [B, N, 3(s), 3(xyz)]

        arange = torch.arange(B, device=device).view(-1, 1, 1).repeat(1, N, 1)  # [B, N, 1]

        # TODO: make parallel
        offsets = itertools.product(range(3), range(3), range(3))
        for offset in offsets:
            offset = torch.tensor(offset, dtype=torch.int, device=device)
            weight = (w[..., offset[0], 0] * w[..., offset[1], 1] * w[..., offset[2], 2]).unsqueeze(-1)  # [B, N, 1]

            indices = torch.cat([arange, base + offset.view(1, 1, 3) - 1], dim=-1)  # [B, N, 4]

            input_iter = input[indices[..., 0], indices[..., 1], indices[..., 2], indices[..., 3]]  # [B, N, C_in]
            if self.has_bias:
                result_iter = weight * (torch.matmul(input_iter, self.kernel[offset[0], offset[1], offset[2]])
                                        + self.bias.view(1, 1, -1))  # [B, N, C_out]
            else:
                result_iter = weight * torch.matmul(input_iter,
                                                    self.kernel[offset[0], offset[1], offset[2]])  # [B, N, C_out]

            result += result_iter

        # result += self.bias.view(1, 1, -1)
        return result


if __name__ == '__main__':
    from datasets.SplishSplash import SplishSplashDataset

    n_grid = [60, 105, 60]
    grid_min = [-30 * 0.05, -1 * 0.05, -30 * 0.05]
    dx = 0.05

    mpm_p2g = MpmP2G(n_grid, grid_min, dx)
    mpm_g2p = MpMG2P(n_grid, grid_min, dx)

    device = torch.device('cuda')
    mpm_p2g = mpm_p2g.to(device)
    mpm_g2p = mpm_g2p.to(device)

    dataset = SplishSplashDataset(train=True, shuffle=True, window=1)

    for i in range(100):
        data = dataset[i]['data0']

        for i in range(len(data)):
            data[i] = data[i].to(device)

        if len(data) == 5:
            pos, vel, acc, _, _ = data
        else:
            pos, vel, acc, _, _, _,  _ = data

        # test data
        # pos_1 = torch.tensor([[0, 1.01, 0]], dtype=torch.float, device=device)
        # acc_1 = torch.tensor([[1, 2, 3]], dtype=torch.float, device=device)
        pos_1 = torch.tensor([[0, 1.01, 0], [0, 1.07, 0]], dtype=torch.float, device=device)
        acc_1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float, device=device)

        grid_val = mpm_p2g(vel, pos)
        new_vel = mpm_g2p(grid_val, pos)

        loss = F.mse_loss(new_vel, vel)
        print("loss:", loss)

        sum_acc = torch.sum(vel, dim=0)
        sum_new_acc = torch.sum(new_vel, dim=0)
        diff = sum_acc - sum_new_acc
        print("diff:", diff)
        print()

