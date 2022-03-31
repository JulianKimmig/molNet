# import relative if not in sys path
import sys
import os

import numpy as np
from rdkit.Chem import GetAdjacencyMatrix
from tqdm import tqdm

from molNet.dataloader.dataloader import DataLoader
from molNet.dataloader.streamer import NumpyStreamer

if __name__ == "__main__":
    modp = os.path.dirname(os.path.abspath(__file__))
    
    while not "molNet" in os.listdir(modp):
        modp=os.path.dirname(modp)
        if os.path.dirname(modp) == modp:
            raise ValueError("connot determine local molNet")
    if modp not in sys.path:
        sys.path.insert(0,modp)
        sys.path.append(modp)

import molNet
from molNet.dataloader.molecular.dataloader import MolDataLoader
from molNet.dataloader.molecular.streamer import PickledMolStreamer
from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization
import pickle

class PreparedMolDataLoader(MolDataLoader):
        raw_file = "mols"
        data_streamer_generator = PickledMolStreamer.generator(
        folder_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )
        
        def __init__(self,mdl:MolDataLoader,parent_dir=None,**kwargs):
            assert not isinstance(mdl,PreparedMolDataLoader)
            assert isinstance(mdl,MolDataLoader)
            
            if parent_dir is None:
                parent_dir = os.path.join(
                    molNet.get_user_folder(), "dataloader", f"{mdl}_prepared"
                )
                
            super().__init__(parent_dir=parent_dir,**kwargs)
            self.expected_data_size=mdl.expected_data_size
            self.expected_mol_count=mdl.expected_mol_count
            self._mdl=mdl

        def __str__(self):
            return str(self._mdl)

        def _needs_raw(self):
            if not os.path.exists(self.raw_file_path):
                os.makedirs(self.raw_file_path,exist_ok=True)
            molfiles=[f for f in os.listdir(self.raw_file_path) if f.endswith(".mol")]
            if len(molfiles)<self.expected_mol_count:
                self._mdl.close()
                for i,mol in enumerate(tqdm(self._mdl,total=self.expected_data_size,desc="generate prepared mols")):
                    if mol is None:
                        continue
                    pmol=prepare_mol_for_featurization(mol)
                    with open(os.path.join(self.raw_file_path,f"{i}.mol"),"w+b") as f:
                        pickle.dump(pmol,f)



class PreparedMolAdjacencyListDataLoader(PreparedMolDataLoader):
    raw_file = "adjacency_list"
    data_streamer_generator = NumpyStreamer.generator(
        folder_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )

    def __str__(self):
        return f"PreparedMolAdjacencyListDataLoader_{self._mdl}"



    def _needs_raw(self):
        if not os.path.exists(self.raw_file_path):
            os.makedirs(self.raw_file_path,exist_ok=True)
        adj_files=[f for f in os.listdir(self.raw_file_path) if f.endswith(".npy")]
        if len(adj_files)<self.expected_mol_count:
            for i,mol in enumerate(tqdm(self._mdl,total=self.expected_mol_count,desc="generate prepared mols adjecency list")):
                if mol is None:
                    continue
                adj_matrix=GetAdjacencyMatrix(mol)
                row, col = np.where(adj_matrix)
                adj_list =  np.unique(np.sort(np.vstack((row, col)).T,axis=1),axis=0)
                np.save(os.path.join(self.raw_file_path,f"{i}.npy"),adj_list)



def main():
    from molNet.dataloader.molecular.ESOL import ESOL
    l = PreparedMolDataLoader(ESOL())
    print(l.raw_file_path)
    for m in tqdm(l):
        pass
    
if __name__ == "__main__":
    main()
    
    