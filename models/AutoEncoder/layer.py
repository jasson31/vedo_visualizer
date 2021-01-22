import itertools
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import open3d.ml.torch as o3dml

from torch_scatter import scatter


class OurConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, n_grid, grid_min, dx, bias=True):
        super(OurConvTranspose3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        self.register_buffer('n_grid', n_grid)
        self.register_buffer('grid_min', grid_min)
        self.register_buffer('dx', dx)
        self.has_bias = bias

        self.kernel = nn.Parameter(torch.Tensor(3, 3, 3, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.kernel)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

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
        w = torch.stack([0.5 * (0.5 - fx)**2,
                         0.75 - fx**2,
                         0.5 * (0.5 + fx)**2], dim=-2)  # [B, N, 3(s), 3(xyz)]
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

            out = scatter(values, indices_1d, dim=1, dim_size=D*H*W, reduce="sum")  # [B, D*H*W, C_out]
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
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.kernel)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

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


class CConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_grid, grid_pos, dx):
        super(CConvEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer('grid_pos', grid_pos)  # [D*H*W, 3]
        self.register_buffer('n_grid', n_grid)
        self.register_buffer('dx', dx)

        self.cconv = o3dml.layers.ContinuousConv(in_channels=in_channels,
                                                 filters=out_channels,
                                                 kernel_size=[3, 3, 3],
                                                 activation=None,
                                                 use_bias=False,
                                                 coordinate_mapping='ball_to_cube_volume_preserving',
                                                 interpolation='linear')

    def forward(self, input, pos):
        # input: [B, N, C_in]
        # pos: [B, N, 3]
        # result: [B, C_out, D, H, W]
        B, N, _ = pos.shape
        search_radius = self.dx * 2.5

        grid_feat_1d = list()
        for b in range(B):
            grid_feat_1d.append(self.cconv(input[b], pos[b], self.grid_pos, search_radius))
        grid_feat_1d = torch.stack(grid_feat_1d, dim=0)  # [B, D*H*W, C_out]
        grid_feat = grid_feat_1d.view(B, self.n_grid[0], self.n_grid[1], self.n_grid[2], self.out_channels)
        return grid_feat.permute(0, 4, 1, 2, 3)


class CConvDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_grid, grid_pos, dx):
        super(CConvDecoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer('grid_pos', grid_pos)
        self.register_buffer('n_grid', n_grid)
        self.register_buffer('dx', dx)

        self.cconv = o3dml.layers.ContinuousConv(in_channels=in_channels,
                                                 filters=out_channels,
                                                 kernel_size=[3, 3, 3],
                                                 activation=None,
                                                 use_bias=False,
                                                 coordinate_mapping='ball_to_cube_volume_preserving',
                                                 interpolation='linear')

    def forward(self, input, pos):
        # input: [B, C_in, D, H, W]
        # pos: [B, N, 3]
        # result: [B, N, C_out]
        B, N, _ = pos.shape
        search_radius = self.dx * 2.5

        grid_feat = input.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C_in]
        grid_feat_1d = grid_feat.view(B, -1, self.in_channels)
        point_feat = list()
        for b in range(B):
            point_feat.append(self.cconv(grid_feat_1d[b], self.grid_pos, pos[b], search_radius))
        point_feat = torch.stack(point_feat, dim=0)  # [B, N, C_out]
        return point_feat
