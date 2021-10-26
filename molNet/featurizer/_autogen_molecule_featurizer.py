from molNet import MOLNET_LOGGER

_available_featurizer = {}

__all__ = []
try:
    from molNet.featurizer import _autogen_ochem_alerts_molecule_featurizer
    from molNet.featurizer._autogen_ochem_alerts_molecule_featurizer import *

    for n, f in _autogen_ochem_alerts_molecule_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            n = f"ochem_alerts_{n}"
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(f"encoutered duplicate while collecting moelcule featurizer: {n}")
            continue
        _available_featurizer[n] = f
    __all__ += _autogen_ochem_alerts_molecule_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)

try:
    from molNet.featurizer import _autogen_rdkit_feats_list_molecule_featurizer
    from molNet.featurizer._autogen_rdkit_feats_list_molecule_featurizer import *

    for n, f in _autogen_rdkit_feats_list_molecule_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            n = f"rdkit_feats_list_{n}"
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(f"encoutered duplicate while collecting moelcule featurizer: {n}")
            continue
        _available_featurizer[n] = f
    __all__ += _autogen_rdkit_feats_list_molecule_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)

try:
    from molNet.featurizer import _autogen_rdkit_feats_numeric_molecule_featurizer
    from molNet.featurizer._autogen_rdkit_feats_numeric_molecule_featurizer import *

    for n, f in _autogen_rdkit_feats_numeric_molecule_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            n = f"rdkit_feats_numeric_{n}"
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(f"encoutered duplicate while collecting moelcule featurizer: {n}")
            continue
        _available_featurizer[n] = f
    __all__ += _autogen_rdkit_feats_numeric_molecule_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)

try:
    from molNet.featurizer import _autogen_rdkit_feats_numpy_arrays_molecule_featurizer
    from molNet.featurizer._autogen_rdkit_feats_numpy_arrays_molecule_featurizer import *

    for n, f in _autogen_rdkit_feats_numpy_arrays_molecule_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            n = f"rdkit_feats_numpy_{n}"
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(f"encoutered duplicate while collecting moelcule featurizer: {n}")
            continue
        _available_featurizer[n] = f
    __all__ += _autogen_rdkit_feats_numpy_arrays_molecule_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)

try:
    from molNet.featurizer import _autogen_rdkit_feats_rdkit_vec_molecule_featurizer
    from molNet.featurizer._autogen_rdkit_feats_rdkit_vec_molecule_featurizer import *

    for n, f in _autogen_rdkit_feats_rdkit_vec_molecule_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            n = f"rdkit_feats_rdkit_vec_{n}"
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(f"encoutered duplicate while collecting moelcule featurizer: {n}")
            continue
        _available_featurizer[n] = f
    __all__ += _autogen_rdkit_feats_rdkit_vec_molecule_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1")

    for n, f in get_available_featurizer().items():
        print(f, f(testmol))


if __name__ == "__main__":
    main()

