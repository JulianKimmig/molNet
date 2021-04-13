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
        linear_activation=torch.nn.ELU(),
        concat_input=False,
        inject_input_to_gc=True,
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
        
        self.inject_input_to_gc=inject_input_to_gc
        gc_ip_size=initlial_net_sizes[-1]
        
        if self.inject_input_to_gc:
            gc_ip_size+=in_features
            
        self.gc = GCNConv(gc_ip_size, gc_out, bias=bias)

        final_net = [
            torch.nn.Linear(initlial_net_sizes[-1] + gc_out, feats_out, bias=bias)
        ]
        if linear_activation is not None:
            final_net.append(linear_activation)
            
        self.combine = torch.nn.Sequential(*final_net)
        self.concat_input=concat_input
        self.feats_out = feats_out
        if self.concat_input:
            self.feats_out+= in_features

    def forward(self, feats_edges_batch):
        feats, edges, batch = feats_edges_batch
        nfeats = self.fcnn(feats)
        
        if self.inject_input_to_gc:
            gc_feats = self.gc(torch.cat([nfeats, feats], dim=1), edges)
        else:
             gc_feats = self.gc(nfeats, edges)
                
        out_feats=self.combine(torch.cat([nfeats, gc_feats], dim=1))
        if self.concat_input:
            out_feats=torch.cat([out_feats, feats], dim=1)
        return out_feats, edges, batch
