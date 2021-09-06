import torch
from torch import nn
from torch_scatter import (
    scatter_add,
    scatter_max,
    scatter_min,
    scatter_mean,
    scatter_mul,
    # scatter_std,
    scatter_logsumexp,
    # scatter_log_softmax,
    # scatter_softmax,
)


class PoolWeightedSum(nn.Module):
    def __init__(self, n_in_feats, normalize=True, bias=True):
        super(PoolWeightedSum, self).__init__()
        if normalize:
            self.weighting_of_nodes = nn.Sequential(
                nn.Linear(n_in_feats, 1, bias=bias), nn.Sigmoid()
            )
        else:
            self.weighting_of_nodes = nn.Linear(n_in_feats, 1, bias=bias)

    def forward(self, feats, batch):
        # feats = nodes,last_gcn_feats
        weights = self.weighting_of_nodes(feats).squeeze()  # dims = nodes,
        weight_feats = feats.transpose(1, 0)  # dims = last_gcn_feats,nodes
        weight_feats = weights * weight_feats  # dims = last_gcn_feats,nodes
        weight_feats = weight_feats.transpose(1, 0)  # dims = nodes,last_gcn_feats

        summed_nodes = scatter_add(weight_feats, batch, dim=0)
        return summed_nodes


class PoolMax(nn.Module):
    def forward(self, feats, batch):
        maxed_nodes, _ = scatter_max(feats, batch, dim=0)
        return maxed_nodes


class PoolMin(nn.Module):
    def forward(self, feats, batch):
        maxed_nodes, _ = scatter_min(feats, batch, dim=0)
        return maxed_nodes


class PoolMean(nn.Module):
    def forward(self, feats, batch):
        maxed_nodes = scatter_mean(feats, batch, dim=0)
        return maxed_nodes


class PoolProd(nn.Module):
    def forward(self, feats, batch):
        maxed_nodes = scatter_mul(feats, batch, dim=0)
        return maxed_nodes


class PoolLogSumExp(nn.Module):
    def forward(self, feats, batch):
        maxed_nodes = scatter_logsumexp(feats, batch, dim=0)
        return maxed_nodes


class PoolSum(nn.Module):
    def forward(self, feats, batch):
        summed_nodes = scatter_add(
            feats, batch, dim=0
        )  # zeros.scatter_add(0, segment_ids, weight_feats) #dims = number of graphs,last_gcn_feats
        return summed_nodes


class MergedPooling(nn.Module):
    def __init__(self, pooling_layer_dict, pool_names=None):
        super().__init__()
        if isinstance(pooling_layer_dict, list):
            pooling_layer_dict = {
                pool_names[i] for i, pl in enumerate(pooling_layer_dict)
            }
        if pool_names is None:
            pool_names = list(pooling_layer_dict.keys())

        assert len(pool_names) == len(pooling_layer_dict)

        self.pool_names = pool_names
        self.pooling_layer = nn.ModuleDict(pooling_layer_dict)

    def forward(self, feats, batch):
        return torch.cat(
            [self.pooling_layer[pl](feats, batch) for pl in self.pool_names], dim=1
        )

    def __len__(self):
        return len(self.pool_names)
