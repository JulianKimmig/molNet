from typing import Any

import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add, scatter_max, scatter_min
import pytorch_lightning as pl

class PoolWeightedSum(nn.Module):
    def __init__(self, n_in_feats):
        super(PoolWeightedSum, self).__init__()
        self.weighting_of_nodes = nn.Sequential(
            nn.Linear(n_in_feats, 1),
            nn.Sigmoid()
        )

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

class PoolSum(nn.Module):
    def forward(self, feats, batch):
        summed_nodes = scatter_add(feats, batch,
                                   dim=0)  # zeros.scatter_add(0, segment_ids, weight_feats) #dims = number of graphs,last_gcn_feats
        return summed_nodes


class MergedPooling(nn.Module):
    def __init__(self, pooling_layer_dict,pool_names):
        super().__init__()
        self.pool_names = pool_names
        self.pooling_layer = nn.ModuleDict(pooling_layer_dict)

    def forward(self, feats, batch):
        return torch.cat([self.pooling_layer[pl](feats, batch) for pl in self.pool_names], dim=1)


class GraphFingerprint(pl.LightningModule):
    def __init__(self, in_feats, hidden_feats=[None,None,None,None,None,None], fingerprint_size=1024, pooling=None):
        super().__init__()
        if pooling is None:
            pooling = ["weight_sum", "max"]
        self.fingerprint_size = fingerprint_size
        if len(hidden_feats) > 0:
            if hidden_feats[0] is None:
                hidden_feats[0] = in_feats

            for i in range(1, len(hidden_feats)):
                if hidden_feats[i] is None:
                    hidden_feats[i] = hidden_feats[i - 1]


        in_channels = in_feats

        gnn_l = {}
        self.gnn_l_names=[]
        for i,out_feats in enumerate(hidden_feats):
            n="{}_gc".format(i)
            gnn_l[n]=GCNConv(in_channels=in_channels,out_channels=out_feats)
            self.gnn_l_names.append(n)

            in_channels = out_feats

        self.gnn = nn.ModuleDict(gnn_l)

        pools = {}
        pool_names=[]
        last_out = 0

        for i,p in enumerate(pooling):
            if p == "sum":
                n="{}_pooling_{}".format(i,p)
                pool_names.append(n)
                pools[n]=PoolSum()
                last_out += in_channels
            elif p == "max":
                n="{}_pooling_{}".format(i,p)
                pool_names.append(n)
                pools[n]=PoolMax()
                last_out += in_channels
            elif p == "weight_sum":
                n="{}_pooling_{}".format(i,p)
                pool_names.append(n)
                pools[n]=PoolWeightedSum(in_channels)
                last_out += in_channels
            else:
                raise NotImplementedError("pooling '{}' not found".format(p))
        self.pooling = MergedPooling(pools,pool_names)


        graph_out_layer = {}
        self.graph_out_names = []


        if fingerprint_size is None:
            fingerprint_size = last_out
        else:
            n="final_finerprint"
            self.graph_out_names.append(n)
            graph_out_layer[n]=nn.Linear(last_out, fingerprint_size)

        self.fingerprint_size = fingerprint_size


        self.graph_out = nn.ModuleDict(graph_out_layer)

        self.loss = nn.MultiLabelSoftMarginLoss()

    def unpacked_forward(self, feats, edges, batch):
        for n in self.gnn_l_names:
            feats = self.gnn[n](feats, edges)
        feats = self.pooling(feats, batch)

        for n in self.graph_out_names:
            feats = self.graph_out[n](feats)
        return feats

    def forward(self, data):
        return self.unpacked_forward(data.x, data.edge_index, data.batch)

    def training_step(self,batch, *args, **kwargs):
        y_hat=self(batch)
        loss = torch.sum(torch.abs(y_hat-batch.y))
        result = pl.TrainResult(minimize=loss)#
        result.log('train_loss', loss, on_epoch=True)
        return result

    @staticmethod
    def batch_data_converter(data):
        return data, data.y

    @staticmethod
    def predict_function(model, data, device):
        data.to(device)
        pred = model(data)
        return pred

    def configure_optimizers(self):
        optimzer = torch.optim.AdamW(self.parameters(), lr=0.01,amsgrad=True)
        return optimzer

class ExtendedGrapFingerprint(GraphFingerprint):
    def __init__(self, in_feats,len_additional_data=0,fully_connected_layer=[],fingerprint_size=1024,*args,**kwargs):
        super().__init__(in_feats,*args,fingerprint_size=None,**kwargs)

        last_out = self.fingerprint_size

        graph_out_layer = {}
        self.graph_out_names=[]

        last_out = last_out + len_additional_data

        i=0
        if len(fully_connected_layer) > 0:
            if fully_connected_layer[0] is None:
                fully_connected_layer[0] = last_out
            n="0_post_linear"
            self.graph_out_names.append(n)
            graph_out_layer[n]=nn.Linear(last_out, fully_connected_layer[0])
            last_out = fully_connected_layer[0]


            for i in range(1, len(fully_connected_layer)):
                if fully_connected_layer[i] is None:
                    fully_connected_layer[i] = last_out
                n="{}_post_linear".format(i)
                self.graph_out_names.append(n)
                graph_out_layer[n]=nn.Linear(last_out, fully_connected_layer[i])
                last_out = fully_connected_layer[i]

        i+=1
        if fingerprint_size is None:
            fingerprint_size = last_out
        else:
            n="{}_final_finerprint".format(i)
            self.graph_out_names.append(n)
            graph_out_layer[n]=nn.Linear(last_out, fingerprint_size)

        self.fingerprint_size = fingerprint_size

        self.graph_out = nn.ModuleDict(graph_out_layer)

    def unpacked_forward(self, feats, edges, batch,graph_features):
        for n in self.gnn_l_names:
            feats = self.gnn[n](feats, edges)
        feats = self.pooling(feats, batch)

        feats = torch.cat([feats,graph_features],1)
        for n in self.graph_out_names:
            feats = self.graph_out[n](feats)
        return feats

    def forward(self, data):
        return self.unpacked_forward(data.x, data.edge_index, data.batch,data.graph_features)

class SubGraphFingerPrint(GraphFingerprint):
    def __init__(self, graph_fingerprint_module, hidden_feats, fingerprint_size):
        super(SubGraphFingerPrint, self).__init__(in_feats=graph_fingerprint_module.fingerprint_size,
                                                  hidden_feats=hidden_feats, fingerprint_size=fingerprint_size)

        self.graph_fingerprint_module = graph_fingerprint_module

    def forward(self, data):
        subgraph_feats = self.graph_fingerprint_module(data.subgraph_data)
        subgraph_indices = data.subgraph_indices

        feats = torch.cat([data.node_features, subgraph_feats[subgraph_indices]], dim=1)

        return self.unpacked_forward(feats, data.edge_index, data.batch)

