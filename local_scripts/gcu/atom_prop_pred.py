import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping
from rdkit.Chem import GetAdjacencyMatrix
from tqdm import tqdm

import molNet
from molNet.dataloader.dataloader import DataLoader
from molNet.dataloader.lighning import InMemoryLoader
from molNet.dataloader.molecular.ESOL import ESOL
from molNet.dataloader.molecular.dataloader import MolDataLoader
from molNet.dataloader.molecular.prepmol import PreparedMolDataLoader, PreparedMolAdjacencyListDataLoader
from molNet.dataloader.streamer import NumpyStreamer
from molNet.featurizer import get_molecule_featurizer_info, get_atom_featurizer_info
from molNet.featurizer._autogen_rdkit_feats_numeric_atom_featurizer import Atom_AtomicNum_Featurizer
from molNet.featurizer._molecule_featurizer import prepare_mol_for_featurization
from molNet.featurizer.prefeaturizer import Prefeaturizer, UnknownFeaturizerError
from molNet.nn.graph.torch_geometric import graph_input_from_edgelist
from molNet.utils.mol import ATOMIC_SYMBOL_NUMBERS

from molNet.nn.graph.graph_convolution import VanillaGC
import pytorch_lightning as pl


symbs=[""]*120
for s,i in ATOMIC_SYMBOL_NUMBERS.items():
    symbs[i]=s

import torch
import torch_geometric
from torch_geometric.nn import Sequential as GCSequential
from torch.utils.data import random_split, Subset

class VanillaGCNodeModel(pl.LightningModule):
    def __init__(self, in_size,
                 out_size,
                 gc_feats_out,
                 n_gc_layer=6,
                 lr=1e-3, lossf=None,
                 final_activation=None,
                 ):
        super().__init__()
        self.save_hyperparameters()
        if lossf is None:
            lossf = torch.nn.MSELoss()

        gc = VanillaGC(
            in_size=in_size,
            out_size=out_size,
            gc_feats_out=gc_feats_out,
            n_gc_layer=n_gc_layer,
        )
        seqs = [(gc, 'x, edge_index -> x')]
        if final_activation is not None:
            seqs.append((final_activation, 'x -> x'))
        self.chem_layer = GCSequential(
            'x, edge_index', seqs
        )

        self.lr = lr
        self.lossf = lossf

    def unpacked_forward(self, feats, edges):
        feats = self.chem_layer(feats, edges)
        return feats

    def forward(self, batch):
        return self.unpacked_forward(feats=batch.x, edges=batch.edge_index)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def batch_to_loss(self, batch):
        y_hat = self(batch)
        loss = self.lossf(y_hat, batch.y)
        return loss

    def training_step(self, batch, *args, **kwargs):
        loss = self.batch_to_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        loss = self.batch_to_loss(batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, *args, **kwargs):
        loss = self.batch_to_loss(batch)
        self.log('test_loss', loss)
        return loss


class RelEarlyStopping(Callback):
    def __init__(self, patience=5, factor=0.99, metric='val_loss'):
        super().__init__()
        self.patience = patience
        self.factor = factor
        self.metric = metric
        self.best_val_loss = np.inf
        self.counter = 0
        self.best_score = torch.tensor(np.Inf)

    def on_validation_end(self, trainer,*args, **kwargs):
        if trainer.running_sanity_check:
            return

        self._run_early_stopping_check(trainer)

    def _run_early_stopping_check(self, trainer):
        """
        Checks whether the early stopping condition is met
        and if so tells the trainer to stop the training.
        """
        logs = trainer.callback_metrics

        current = logs.get(self.metric)

        # when in dev debugging
        trainer.dev_debugger.track_early_stopping_history(self, current)

        if torch.lt(current, self.best_score*self.factor):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1

            if self.wait_count >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True

        # stop every ddp process if any world process decides to stop
        trainer.should_stop = trainer.training_type_plugin.reduce_early_stopping_decision(trainer.should_stop)

def train_and_test(model, dataloader,max_epochs=1000):
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=False,
        checkpoint_callback=False,
        callbacks=[RelEarlyStopping(metric="val_loss",patience=3,factor=0.99)],
    )
    trainer.fit(model, dataloader)
    test_result = trainer.test(model,dataloader.test_dataloader())
    return test_result
    #trainer.(model, dataloader)




def main():
    all_feats=[str(d["instance"]) for i,d in get_atom_featurizer_info().iterrows()]
    try:
        res_df = pd.read_csv("all_feats.csv",index_col=0)
    except FileNotFoundError:
        res_df=pd.DataFrame(columns=all_feats,index=all_feats)
        res_df.to_csv("all_feats.csv")

    #molNet.MOLNET_LOGGER.info("Loading data...")
    dataset=ESOL(data_streamer_kwargs=dict(iter_None=True))

    mol_loader = PreparedMolDataLoader(dataset)

    adj_list_dl=PreparedMolAdjacencyListDataLoader(mol_loader)
    adj_list_dl.close()

    #for m in all:
    #    print(m)


    for i,d1 in get_atom_featurizer_info().iterrows():
        #reset loader
        adj_list_dl.close()
        mol_loader.close()

        featurizer1=d1["instance"]

        try:
            dp1 = Prefeaturizer(str(mol_loader),mol_loader,featurizer=featurizer1)
            dp1.prefeaturize()
        except UnknownFeaturizerError:
            continue
        for j,d2 in get_atom_featurizer_info().iterrows():
            featurizer2=d2["instance"]


            print(str(featurizer1),str(featurizer2))
            if not np.isnan(res_df.loc[str(featurizer1),str(featurizer2)]):
                print(res_df.loc[str(featurizer1),str(featurizer2)])
                continue
            try:
                dp2 = Prefeaturizer(str(mol_loader),mol_loader,featurizer=featurizer2)
                dp2.prefeaturize()
            except UnknownFeaturizerError:
                continue

            ouutput=np.concatenate([k for k in dp2])
            meanop=ouutput.mean(0).reshape(1,-1)
            unevenop=~((ouutput == meanop).all(0))
            if unevenop.sum()==0:
                print("all equal")
                continue
            ouutput=[k[:,unevenop] for k in dp2]




            vanilla_model=VanillaGCNodeModel(
                in_size=len(featurizer1),
                out_size=unevenop.sum(),
                gc_feats_out=12,
                n_gc_layer=3
            )


            dataloader=InMemoryLoader([graph_input_from_edgelist(adj.T,feats,y=y) for adj,feats,y in zip(adj_list_dl,dp1,ouutput)],batch_size=1024,dataloader = torch_geometric.loader.DataLoader)

            results = train_and_test(vanilla_model,dataloader)
            res_df.loc[str(featurizer1),str(featurizer2)]=results[0]["test_loss"]
            res_df.to_csv("all_feats.csv")





        #print(featurizer)


        #feats=[k for k in tqdm(dp,total=len(dp))]
        #print(feats)



if __name__ == "__main__":
    main()