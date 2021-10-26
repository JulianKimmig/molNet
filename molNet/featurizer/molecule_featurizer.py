import numpy as np

from molNet import MOLNET_LOGGER
from molNet.featurizer._molecule_featurizer import SingleValueMoleculeFeaturizer
from molNet.featurizer.featurizer import FeaturizerList

_available_featurizer = {}
__all__ = []
try:
    from molNet.featurizer import _manual_molecule_featurizer
    from molNet.featurizer._manual_molecule_featurizer import *

    for n, f in _manual_molecule_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(f"encoutered duplicate while collecting moelcule featurizer: {n}")
            continue
        _available_featurizer[n] = f

    __all__ += _manual_molecule_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)

try:
    from molNet.featurizer import _autogen_molecule_featurizer
    from molNet.featurizer._autogen_molecule_featurizer import *

    for n, f in _autogen_molecule_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            n = f"autogen_molecule_featurizer_{n}"
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(f"encoutered duplicate while collecting moelcule featurizer: {n}")
            continue
        _available_featurizer[n] = f

    __all__ += _autogen_molecule_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)


class AllSingleValueMoleculeFeaturizer(FeaturizerList):
    dtype = np.float32

    def __init__(self, *args, **kwargs):
        super().__init__(
            [
                f
                for n, f in _available_featurizer.items()
                if isinstance(f, SingleValueMoleculeFeaturizer)
            ],
            *args,
            **kwargs
        )


molecule_all_single_val_feats = AllSingleValueMoleculeFeaturizer(name="molecule_all_single_val_feats")

__all__.extend(["molecule_all_single_val_feats", "AllSingleValueMoleculeFeaturizer"])
_available_featurizer["molecule_all_single_val_feats"] = molecule_all_single_val_feats


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1")

    for n, f in get_available_featurizer().items():
        print(f, f(testmol))


if __name__ == "__main__":
    main()

default_molecule_featurizer = FeaturizerList(
    [],
    name="default_molecule_featurizer",
)
