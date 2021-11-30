# import relative if not in sys path
import sys
import os
from tqdm import tqdm
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
                    molNet.get_user_folder(), "dataloader", f"{mdl.__class__.__name__}_prepared"
                )
                
            super().__init__(parent_dir=parent_dir,**kwargs)
            self.expected_data_size=mdl.expected_data_size
            self._mdl=mdl
        
        def _needs_raw(self):
            if not os.path.exists(self.raw_file_path):
                os.makedirs(self.raw_file_path,exist_ok=True)
            molfiles=[f for f in os.listdir(self.raw_file_path) if f.endswith(".mol")]
            if len(molfiles)<self.expected_data_size:
                for i,mol in enumerate(tqdm(self._mdl,total=self.expected_data_size,desc="generate prepared mols")):
                    pmol=prepare_mol_for_featurization(mol)
                    with open(os.path.join(self.raw_file_path,f"{i}.mol"),"w+b") as f:
                        pickle.dump(pmol,f)
                    
                
def main():
    from molNet.dataloader.molecular.ESOL import ESOL
    l = PreparedMolDataLoader(ESOL())
    print(l.raw_file_path)
    for m in tqdm(l):
        pass
    
if __name__ == "__main__":
    main()
    
    