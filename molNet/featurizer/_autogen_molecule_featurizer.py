from molNet.featurizer._autogen_ochem_alerts_molecule_featurizer import *
from molNet.featurizer._autogen_ochem_alerts_molecule_featurizer import (
    _available_featurizer as _autogen_ochem_alerts_molecule_featurizer_available_featurizer,
)
from molNet.featurizer._autogen_rdkit_feats_list_molecule_featurizer import *
from molNet.featurizer._autogen_rdkit_feats_list_molecule_featurizer import (
    _available_featurizer as _autogen_rdkit_feats_list_molecule_featurizer_available_featurizer,
)
from molNet.featurizer._autogen_rdkit_feats_numeric_molecule_featurizer import *
from molNet.featurizer._autogen_rdkit_feats_numeric_molecule_featurizer import (
    _available_featurizer as _autogen_rdkit_feats_numeric_molecule_featurizer_available_featurizer,
)
from molNet.featurizer._autogen_rdkit_feats_numpy_arrays_molecule_featurizer import *
from molNet.featurizer._autogen_rdkit_feats_numpy_arrays_molecule_featurizer import (
    _available_featurizer as _autogen_rdkit_feats_numpy_arrays_molecule_featurizer_available_featurizer,
)
from molNet.featurizer._autogen_rdkit_feats_rdkit_vec_molecule_featurizer import *
from molNet.featurizer._autogen_rdkit_feats_rdkit_vec_molecule_featurizer import (
    _available_featurizer as _autogen_rdkit_feats_rdkit_vec_molecule_featurizer_available_featurizer,
)

_available_featurizer = [
    *_autogen_ochem_alerts_molecule_featurizer_available_featurizer,
    *_autogen_rdkit_feats_list_molecule_featurizer_available_featurizer,
    *_autogen_rdkit_feats_numeric_molecule_featurizer_available_featurizer,
    *_autogen_rdkit_feats_numpy_arrays_molecule_featurizer_available_featurizer,
    *_autogen_rdkit_feats_rdkit_vec_molecule_featurizer_available_featurizer,
]


def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1")

    for f in _available_featurizer:
        print(f, f(testmol))


if __name__ == "__main__":
    main()
