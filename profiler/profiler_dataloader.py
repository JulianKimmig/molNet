import cProfile,pstats
import pandas as pd

from molNet.dataloader.molecule_loader import PytorchGeomMolDfLoader
from molNet.featurizer.atom_featurizer import atom_partial_charge, atom_formal_charge

profile = cProfile.Profile()


data = {
    "smiles": ["C", "CC", "CCC", "CCCC"],
    "mass": [12, 24, 36, 48],
    "charge": [0, 1, -2, -1],
    "n": [1, 2, 3, 4],
    "prop": [1.0, 2, 17, 0.32],
    "prop2": [0.0, 2.2, 170, 10],
    "prop3": [True, False, True, False],
}
df = pd.DataFrame(data)


def profile_PytorchGeomMolDfLoader():
    for i in range(100):
        dfl = PytorchGeomMolDfLoader(
            df,
            y_columns=["prop", "prop2", "prop3"],
            atom_featurizer=atom_partial_charge,
            y_atom_featurizer=atom_formal_charge,
            inplace=True,
            to_graph_input_kwargs=dict(
                keep_string_data=True, include_graph_features_titles=True
            ),
        )
        dfl.setup()
    for batch in dfl.train_dataloader():
        pass


profile.runcall(profile_PytorchGeomMolDfLoader)
ps = pstats.Stats(profile)
ps.sort_stats('cumtime')
ps.print_stats(20)


