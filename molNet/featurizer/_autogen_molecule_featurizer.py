from molNet import MOLNET_LOGGER

_available_featurizer = {}
__all__ = []

try:
    from molNet.featurizer import _autogen_ochem_alerts_molecule_featurizer
    from molNet.featurizer._autogen_ochem_alerts_molecule_featurizer import *

    for (
            n,
            f,
    ) in _autogen_ochem_alerts_molecule_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(
                f"encoutered duplicate while collecting moelcule featurizer: {n}"
            )
            continue
        _available_featurizer[n] = f

    __all__ += _autogen_ochem_alerts_molecule_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)

try:
    from molNet.featurizer import _autogen_rdkit_atomtype_molecule_featurizer
    from molNet.featurizer._autogen_rdkit_atomtype_molecule_featurizer import *

    for (
            n,
            f,
    ) in _autogen_rdkit_atomtype_molecule_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(
                f"encoutered duplicate while collecting moelcule featurizer: {n}"
            )
            continue
        _available_featurizer[n] = f

    __all__ += _autogen_rdkit_atomtype_molecule_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)

try:
    from molNet.featurizer import _autogen_rdkit_feats_array_molecule_featurizer
    from molNet.featurizer._autogen_rdkit_feats_array_molecule_featurizer import *

    for (
            n,
            f,
    ) in (
            _autogen_rdkit_feats_array_molecule_featurizer.get_available_featurizer().items()
    ):
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(
                f"encoutered duplicate while collecting moelcule featurizer: {n}"
            )
            continue
        _available_featurizer[n] = f

    __all__ += _autogen_rdkit_feats_array_molecule_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)

try:
    from molNet.featurizer import _autogen_rdkit_feats_numeric_molecule_featurizer
    from molNet.featurizer._autogen_rdkit_feats_numeric_molecule_featurizer import *

    for (
            n,
            f,
    ) in (
            _autogen_rdkit_feats_numeric_molecule_featurizer.get_available_featurizer().items()
    ):
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(
                f"encoutered duplicate while collecting moelcule featurizer: {n}"
            )
            continue
        _available_featurizer[n] = f

    __all__ += _autogen_rdkit_feats_numeric_molecule_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)

try:
    from molNet.featurizer import _autogen_rdkit_feats_str_molecule_featurizer
    from molNet.featurizer._autogen_rdkit_feats_str_molecule_featurizer import *

    for (
            n,
            f,
    ) in (
            _autogen_rdkit_feats_str_molecule_featurizer.get_available_featurizer().items()
    ):
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(
                f"encoutered duplicate while collecting moelcule featurizer: {n}"
            )
            continue
        _available_featurizer[n] = f

    __all__ += _autogen_rdkit_feats_str_molecule_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)

try:
    from molNet.featurizer import _autogen_rdkit_feats_vec_molecule_featurizer
    from molNet.featurizer._autogen_rdkit_feats_vec_molecule_featurizer import *

    for (
            n,
            f,
    ) in (
            _autogen_rdkit_feats_vec_molecule_featurizer.get_available_featurizer().items()
    ):
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(
                f"encoutered duplicate while collecting moelcule featurizer: {n}"
            )
            continue
        _available_featurizer[n] = f

    __all__ += _autogen_rdkit_feats_vec_molecule_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem
    from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization

    testmol = prepare_mol_for_featurization(Chem.MolFromSmiles("c1ccccc1"))
    for n, f in get_available_featurizer().items():
        print(n, end=" ")
        print(f(testmol))
    print(len(get_available_featurizer()))


if __name__ == "__main__":
    main()
