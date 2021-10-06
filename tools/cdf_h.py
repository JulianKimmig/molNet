import numpy as np
from molNet.mol.molgraph import mol_graph_from_smiles
from molNet import ConformerError

def func(d):
    f = d[0][3]()
    r = np.zeros((len(d), len(f))) * np.nan
    for i, data in enumerate(d):
        mg = mol_graph_from_smiles(data[0], *data[1], **data[2])
        try:
            mg.featurize_mol(f, name="para_feats")
            r[i] = mg.as_arrays()["graph_features"]["para_feats"]
        except ConformerError:
            pass
    return r
