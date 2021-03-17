import torch
from torch_geometric.nn import GCNConv


class ChemGCLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,
        initlial_net_sizes,
        gc_out,
        feats_out,
        bias=True,
        linear_activation=None,
    ):
        super().__init__()
        initlial_net_sizes = [in_features] + initlial_net_sizes
        inital_net = []
        for i in range(1, len(initlial_net_sizes)):
            inital_net.append(
                torch.nn.Linear(
                    initlial_net_sizes[i - 1], initlial_net_sizes[i], bias=bias
                )
            )
            if linear_activation is not None:
                inital_net.append(linear_activation)

        self.fcnn = torch.nn.Sequential(*inital_net)

        self.gc = GCNConv(initlial_net_sizes[-1], gc_out, bias=bias)

        final_net = [
            torch.nn.Linear(initlial_net_sizes[-1] + gc_out, feats_out, bias=bias)
        ]
        if linear_activation is not None:
            final_net.append(linear_activation)
        self.combine = torch.nn.Sequential(*final_net)
        self.feats_out = feats_out

    def forward(self, feats_edges_batch):
        feats, edges, batch = feats_edges_batch
        nfeats = self.fcnn(feats)

        gc_feats = self.gc(nfeats, edges)

        return self.combine(torch.cat([nfeats, gc_feats], dim=1)), edges, batch
