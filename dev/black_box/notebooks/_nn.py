import torch
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
import molNet

class ChemGCLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,
        initlial_net_sizes,
        gc_out,
        feats_out,
        bias=True,
        linear_activation=torch.nn.ELU(),
        concat_input=True,
        inject_input_to_gc=True,
        dropout=0.2,
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
            if dropout>0:
                inital_net.append(
                torch.nn.Dropout(p=min(0.99,dropout))
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
        if dropout>0:
                final_net.append(
                torch.nn.Dropout(p=min(0.99,dropout))
            )
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


class ChemGCModel(pl.LightningModule):
    def __init__(self,in_size,out_size,name="none",n_chem_layer=6,chem_layer_feats_out=40,chem_layer_gc_out=30,chem_layer_net_sizes=10,chem_layer_net_count=2,
                 collector_net_depth=3,collector_net_depth_red_fac=2,lr=1e-3,poolings=["pool_wsum"],dropout=0.2,lossf=None,linear_activation=torch.nn.ELU(),post_pool_norm=False):
        super().__init__()
        self.save_hyperparameters()
        if lossf is None:
            lossf=torch.nn.MSELoss()
        #print(self.hparams)
        
        chem_layer=[
            ChemGCLayer(
                in_size,
                [chem_layer_net_sizes if chem_layer_net_sizes >0 else in_size]*chem_layer_net_count,
                chem_layer_gc_out,
                chem_layer_feats_out,
                bias=True,
                linear_activation=linear_activation,
                concat_input=True,
                dropout=0)
        ]
        for n in range(n_chem_layer-1):
            chem_layer.append(
                ChemGCLayer(
                    chem_layer[-1].feats_out,
                    [chem_layer_net_sizes if chem_layer_net_sizes >0 else chem_layer[-1].feats_out]*chem_layer_net_count,
                    chem_layer_gc_out,
                    chem_layer_feats_out,
                    bias=True,
                    linear_activation=linear_activation,
                    concat_input=True,
                    dropout=dropout,
                )
            )
        
        self.chem_layer=torch.nn.Sequential(*chem_layer)
        
        
        self.lr = lr
        
        poolings_layer=[]
        for p in poolings:
            if p=="pool_max":
                poolings_layer.append(molNet.nn.models.graph_convolution.PoolMax())
            elif p=="pool_min":
                poolings_layer.append(molNet.nn.models.graph_convolution.PoolMin())
            elif p=="pool_mean":
                poolings_layer.append(molNet.nn.models.graph_convolution.PoolMean())
            elif p=="pool_sum":
                poolings_layer.append(molNet.nn.models.graph_convolution.PoolSum())
            elif p=="pool_wsum":
                poolings_layer.append(molNet.nn.models.graph_convolution.PoolWeightedSum(self.chem_layer[-1].feats_out,normalize=False))
            elif p=="pool_nwsum":
                poolings_layer.append(molNet.nn.models.graph_convolution.PoolWeightedSum(self.chem_layer[-1].feats_out,normalize=True))

            else:
                raise Exception("unknown pooling '{}'".format(p))
        
        if len(poolings_layer)==0:
            raise ValueError("No pooling")
    
        self.pooling=molNet.nn.models.graph_convolution.MergedPooling(
            {"pool_{}".format(i):poolings_layer[i] for i in range(len(poolings_layer))}
        )
        
        if post_pool_norm:
            self.post_pool_norm=torch.nn.Sigmoid()
        else:
            self.post_pool_norm=torch.nn.Identity()
        
        collector_net = [           torch.nn.Linear(len(self.pooling)*self.chem_layer[-1].feats_out,max(out_size,int(len(self.pooling)*self.chem_layer[-1].feats_out/collector_net_depth_red_fac))),
        ]
        if linear_activation is not None:
            collector_net.append(linear_activation)
        
        if linear_activation is not None:
            d=-2
        else:
            d=-1
            
        for i in range(collector_net_depth-1):
            if collector_net[d].out_features<=out_size:
                break
            collector_net.append(torch.nn.Linear(collector_net[d].out_features,max(out_size,int(collector_net[d].out_features/collector_net_depth_red_fac))))
            if collector_net[-1].out_features<=out_size:
                break
            if linear_activation is not None:
                collector_net.append(linear_activation)
        
        if collector_net[d].out_features!=out_size:
            collector_net.append(torch.nn.Linear(collector_net[d].out_features,out_size))
        
        #collector_net.append(torch.nn.Sigmoid())
        self.nn=torch.nn.Sequential(*collector_net)
        
        self.lossf=lossf

    def unpacked_forward(self, feats, edges, batch,graph_features):
        feats,_,_ = self.chem_layer((feats, edges, batch))

        y_hat = self.pooling(feats,batch)
        y_hat = self.post_pool_norm(y_hat)
        y_hat = self.nn(y_hat)
        return y_hat
      
    def forward(self,batch):
        return self.unpacked_forward(feats=batch.x,edges=batch.edge_index,batch=batch.batch,graph_features=batch.x_graph_features)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def batch_to_loss(self,batch):
        y_hat=self(batch)
        loss = self.lossf(y_hat, batch.y_graph_features)
        return loss

    def training_step(self,batch, *args, **kwargs):
        loss =  self.batch_to_loss(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self,batch, *args, **kwargs):
        loss =  self.batch_to_loss(batch)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self,batch, *args, **kwargs):
        loss =  self.batch_to_loss(batch)
        self.log('test_loss', loss)
        return loss

    
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.eps = 1e-6

    def forward(self,x,y):
        loss = torch.sqrt(self.criterion(x, y) + self.eps)
        return loss

torch.nn.RMSELoss = RMSELoss # monkey inject