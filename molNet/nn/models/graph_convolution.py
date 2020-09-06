import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add, scatter_max, scatter_min


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
    def __init__(self, pooling_layer):
        super().__init__()
        self.pooling_layer = nn.ModuleList(pooling_layer)

    def forward(self, feats, batch):
        return torch.cat([pl(feats, batch) for pl in self.pooling_layer], dim=1)


class GraphFingerprint(nn.Module):
    def __init__(self, in_feats, hidden_feats, fingerprint_size, pooling=None, activation=None):
        super(GraphFingerprint, self).__init__()
        if pooling is None:
            pooling = ["weight_sum", "max"]
        self.fingerprint_size = fingerprint_size
        if len(hidden_feats) > 0:
            if hidden_feats[0] is None:
                hidden_feats[0] = in_feats

            for i in range(1, len(hidden_feats)):
                if hidden_feats[i] is None:
                    hidden_feats[i] = hidden_feats[i - 1]

        if isinstance(activation, (list, tuple)):
            assert len(activation) == hidden_feats + 1
        else:
            activation = [activation for _ in range(len(hidden_feats) + 1)]

        in_channels = in_feats

        gnn_l = []
        activation_n = 0
        gnn_activation=[]
        for out_feats in hidden_feats:
            gnn_activation.append(activation[activation_n]())
            activation_n+=1
            gnn_l.append(
                GCNConv(in_channels=in_channels,
                            out_channels=out_feats)
                )

            in_channels = out_feats

        self.gnn = nn.ModuleList(gnn_l)
        self.gnn_activations = nn.ModuleList(gnn_activation)

        pools = []
        last_out = 0

        for p in pooling:
            if p == "sum":
                pools.append(PoolSum())
                last_out += in_channels
            elif p == "max":
                pools.append(PoolMax())
                last_out += in_channels
            elif p == "weight_sum":
                pools.append(PoolWeightedSum(in_channels))
                last_out += in_channels
            else:
                raise NotImplementedError("pooling '{}' not found".format(p))
        self.pooling = MergedPooling(pools)

        graph_out_layer = [nn.Linear(last_out, fingerprint_size)]
        if activation[activation_n] is not None:
            graph_out_layer.append(activation[activation_n]())
        activation_n += 1

        self.graph_out = nn.Sequential(
            *graph_out_layer
        )

    def unpacked_forward(self, feats, edges, batch):
        for i,gnn in enumerate(self.gnn):
            feats = gnn(feats, edges)
            if self.gnn_activations[i] is not None:
                feats = self.gnn_activations[i](feats)
        feats = self.pooling(feats, batch)
        feats = self.graph_out(feats)

        return feats

    def forward(self, data):
        return self.unpacked_forward(data.node_features, data.edge_index, data.batch)

    @staticmethod
    def batch_data_converter(data):
        return data, data.y

    @staticmethod
    def predict_function(model, data, device):
        data.to(device)
        pred = model(data)
        return pred


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
