import numpy as np

from molNet import MOLNET_LOGGER
from molNet.featurizer._atom_featurizer import SingleValueAtomFeaturizer
from molNet.featurizer.featurizer import FeaturizerList

_available_featurizer = {}
__all__ = []
try:
    from molNet.featurizer import _manual_atom_featurizer
    from molNet.featurizer._manual_molecule_featurizer import *

    for n, f in _manual_atom_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(f"encoutered duplicate while collecting moelcule featurizer: {n}")
            continue
        _available_featurizer[n] = f

    __all__ += _manual_atom_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)

try:
    from molNet.featurizer import _autogen_atom_featurizer
    from molNet.featurizer._autogen_atom_featurizer import *

    for n, f in _autogen_atom_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            n = f"autogen_molecule_featurizer_{n}"
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(f"encoutered duplicate while collecting moelcule featurizer: {n}")
            continue
        _available_featurizer[n] = f

    __all__ += _autogen_atom_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)


class AllSingleValueAtomFeaturizer(FeaturizerList):
    dtype = np.float32

    def __init__(self, *args, **kwargs):
        super().__init__(
            [
                f
                for n, f in _available_featurizer.items()
                if isinstance(f, SingleValueAtomFeaturizer)
            ],
            *args,
            **kwargs
        )


atom_all_single_val_feats = AllSingleValueAtomFeaturizer(name="atom_all_single_val_feats")
__all__.extend(["atom_all_single_val_feats", "AllSingleValueAtomFeaturizer"])
_available_featurizer["atom_all_single_val_feats"] = atom_all_single_val_feats


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem
    from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization

    testmol = prepare_mol_for_featurization(Chem.MolFromSmiles("c1ccccc1")).GetAtoms()[
        -1
    ]
    for n, f in get_available_featurizer().items():
        print(n, end=" ")
        print(f(testmol))
    print(len(get_available_featurizer()))


if __name__ == "__main__":
    main()

