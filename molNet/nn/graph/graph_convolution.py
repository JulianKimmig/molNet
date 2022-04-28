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
from torch_geometric.nn import Sequential as GCSequential, GCNConv


class PoolingBase(nn.Module):
    def prep_pool(self,n_in_feats):
        pass
    
class PoolWeightedSum(PoolingBase):
    def __init__(self, n_in_feats=None, normalize=True, bias=True):
        super(PoolWeightedSum, self).__init__()
        self._normalize=normalize
        self._bias=bias
        self._preped=False
        
        if n_in_feats is not None:
            self.prep_pool(n_in_feats)
    
    def prep_pool(self,n_in_feats):
        if self._preped:
            return
        
        if self._normalize:
            self.weighting_of_nodes = nn.Sequential(
                nn.Linear(n_in_feats, 1, bias=self._bias), nn.Sigmoid()
            )
        else:
            self.weighting_of_nodes = nn.Linear(n_in_feats, 1, bias=self._bias)
        self._preped=True
    
    def forward(self, feats, batch):
        # feats = nodes,last_gcn_feats
        weights = self.weighting_of_nodes(feats).squeeze()  # dims = nodes,
        weight_feats = feats.transpose(1, 0)  # dims = last_gcn_feats,nodes
        weight_feats = weights * weight_feats  # dims = last_gcn_feats,nodes
        weight_feats = weight_feats.transpose(1, 0)  # dims = nodes,last_gcn_feats

        summed_nodes = scatter_add(weight_feats, batch, dim=0)
        return summed_nodes


class PoolMax(PoolingBase):
    def forward(self, feats, batch):
        maxed_nodes, _ = scatter_max(feats, batch, dim=0)
        return maxed_nodes


class PoolMin(PoolingBase):
    def forward(self, feats, batch):
        maxed_nodes, _ = scatter_min(feats, batch, dim=0)
        return maxed_nodes


class PoolMean(PoolingBase):
    def forward(self, feats, batch):
        maxed_nodes = scatter_mean(feats, batch, dim=0)
        return maxed_nodes


class PoolProd(PoolingBase):
    def forward(self, feats, batch):
        maxed_nodes = scatter_mul(feats, batch, dim=0)
        return maxed_nodes


class PoolLogSumExp(PoolingBase):
    def forward(self, feats, batch):
        maxed_nodes = scatter_logsumexp(feats, batch, dim=0)
        return maxed_nodes


class PoolSum(PoolingBase):
    def forward(self, feats, batch):
        summed_nodes = scatter_add(
            feats, batch, dim=0
        )  # zeros.scatter_add(0, segment_ids, weight_feats) #dims = number of graphs,last_gcn_feats
        return summed_nodes


class MergedPooling(PoolingBase):
    def __init__(self, pooling_layer_dict):
        super().__init__()
        if isinstance(pooling_layer_dict, list):
            pooling_layer_dict = {
                str(i):pl for i, pl in enumerate(pooling_layer_dict)
            }
        
        pool_names = list(pooling_layer_dict.keys())

        assert len(pool_names) == len(pooling_layer_dict)
        
        self.pool_names = pool_names
        self.pooling_layer_dict=pooling_layer_dict
        
    
    def prep_pool(self,n_in_feats):
        for k,pl in self.pooling_layer_dict.items():
            pl.prep_pool(n_in_feats)
        
        self.pooling_layer = nn.ModuleDict(self.pooling_layer_dict)
        
    def forward(self, feats, batch):
        return torch.cat(
            [self.pooling_layer[pl](feats, batch) for pl in self.pool_names], dim=1
        )

    def __len__(self):
        return len(self.pool_names)


class VanillaGC(torch.nn.Module):
    def __init__(self, in_size,
                 out_size,
                 gc_feats_out,
                 n_gc_layer=6,
                 ):
        super().__init__()




        chem_layer=[
            GCNConv(in_size, gc_feats_out if n_gc_layer>1 else out_size, bias=True)
        ]
        for n in range(n_gc_layer-1-1):
            chem_layer.append(
                GCNConv(gc_feats_out, gc_feats_out, bias=True)
            )
        if n_gc_layer>1:
            chem_layer.append(
                GCNConv(gc_feats_out, out_size, bias=True)
            )

        self.chem_layer=GCSequential(
            'x, edge_index',
            [(cl, 'x, edge_index -> x') for cl in chem_layer]
        )


    def forward(self, feats, edges):
        feats = self.chem_layer(feats, edges)
        return feats