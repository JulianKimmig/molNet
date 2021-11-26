from rdkit.Chem.rdchem import Atom

from molNet import MOLNET_LOGGER
from molNet.featurizer._molecule_featurizer import prepare_mol_for_featurization, check_mol_is_prepared
from molNet.featurizer.featurizer import Featurizer, FixedSizeFeaturizer, OneHotFeaturizer, StringFeaturizer


class _AtomFeaturizer(Featurizer):
    def pre_featurize(self, atom: Atom):
        mol = atom.GetOwningMol()

        if not check_mol_is_prepared(mol):
            if not self._unprepared_logged:
                MOLNET_LOGGER.warning("you tried to featurize an atom without previous preparation of the molecule. "
                                      "I will do this for you, but please try to implement this, "
                                      "otherwise you might end uo with differences in yout molecules and the featurized,"
                                      " since the preparation creates an copy of the molecule, "
                                      "adds hydrogens, conformerst etc."
                                      "")
                self._unprepared_logged = True

            from rdkit.Chem.rdchem import Mol
            mol: Mol = prepare_mol_for_featurization(mol, renumber=False)
            atom = mol.GetAtomWithIdx(atom.GetIdx())
        if self._add_prefeat:
            atom = self._add_prefeat(atom)
        return atom

    def __init__(self, *args, **kwargs):
        self._add_prefeat = kwargs.get("pre_featurize", None)
        self._unprepared_logged = False
        kwargs["pre_featurize"] = None

        super().__init__(*args, **kwargs)


class VarSizeAtomFeaturizer(_AtomFeaturizer, Featurizer):
    pass


AtomFeaturizer = VarSizeAtomFeaturizer


class FixedSizeAtomFeaturizer(_AtomFeaturizer, FixedSizeFeaturizer):
    pass


class SingleValueAtomFeaturizer(FixedSizeAtomFeaturizer):
    LENGTH = 1


class OneHotAtomFeaturizer(_AtomFeaturizer, OneHotFeaturizer):
    pass


class StringAtomFeaturizer(_AtomFeaturizer, StringFeaturizer):
    pass
