import torch
from torch_geometric.nn import MessagePassing


def reset(_nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if _nn is not None:
        if hasattr(_nn, 'children') and len(list(_nn.children())) > 0:
            for item in _nn.children():
                _reset(item)
        else:
            _reset(_nn)


class OurGConv(MessagePassing):
    def __init__(self, nn, filter, eps: float = 0., train_eps: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(OurGConv, self).__init__(**kwargs)
        self.nn = nn
        self.filter = filter
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_attr, size=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)  # [sender, receiver]

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j, edge_attr):
        weight = self.filter(edge_attr)
        return x_j * weight

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)
