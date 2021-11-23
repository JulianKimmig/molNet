import numpy as np
from rdkit.Chem.Descriptors import (Chi1)

from molNet.featurizer._atom_featurizer import (SingleValueAtomFeaturizer)
from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization


class Atom_AtomicNum_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=GetAtomicNum
    dtype = np.int32

    def featurize(self, atom):
        return atom.GetAtomicNum()


class Atom_Degree_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=GetDegree
    dtype = np.int32

    def featurize(self, atom):
        return atom.GetDegree()


class Atom_ExplicitValence_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=GetExplicitValence
    dtype = np.int32

    def featurize(self, atom):
        return atom.GetExplicitValence()


class Atom_FormalCharge_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=GetFormalCharge
    dtype = np.int32

    def featurize(self, atom):
        return atom.GetFormalCharge()


class Atom_ImplicitValence_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=GetImplicitValence
    dtype = np.int32

    def featurize(self, atom):
        return atom.GetImplicitValence()


class Atom_IsAromatic_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=GetIsAromatic
    dtype = bool

    def featurize(self, atom):
        return atom.GetIsAromatic()


class Atom_Isotope_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=GetIsotope
    dtype = np.int32

    def featurize(self, atom):
        return atom.GetIsotope()


class Atom_Mass_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=GetMass
    dtype = np.float32

    def featurize(self, atom):
        return atom.GetMass()


class Atom_NoImplicit_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=GetNoImplicit
    dtype = bool

    def featurize(self, atom):
        return atom.GetNoImplicit()


class Atom_NumExplicitHs_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=GetNumExplicitHs
    dtype = np.int32

    def featurize(self, atom):
        return atom.GetNumExplicitHs()


class Atom_NumImplicitHs_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=GetNumImplicitHs
    dtype = np.int32

    def featurize(self, atom):
        return atom.GetNumImplicitHs()


class Atom_NumRadicalElectrons_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=GetNumRadicalElectrons
    dtype = np.int32

    def featurize(self, atom):
        return atom.GetNumRadicalElectrons()


class Atom_TotalDegree_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=GetTotalDegree
    dtype = np.int32

    def featurize(self, atom):
        return atom.GetTotalDegree()


class Atom_TotalNumHs_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=GetTotalNumHs
    dtype = np.int32

    def featurize(self, atom):
        return atom.GetTotalNumHs()


class Atom_TotalValence_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=GetTotalValence
    dtype = np.int32

    def featurize(self, atom):
        return atom.GetTotalValence()


class Atom_IsInRing_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=IsInRing
    dtype = bool

    def featurize(self, atom):
        return atom.IsInRing()


class Atom_Chi1_Featurizer(SingleValueAtomFeaturizer):
    # _rdfunc=rdkit.Chem.Descriptors.Chi1
    dtype = np.float32
    featurize = staticmethod(Chi1)


atom_AtomicNum_featurizer = Atom_AtomicNum_Featurizer()
atom_Degree_featurizer = Atom_Degree_Featurizer()
atom_ExplicitValence_featurizer = Atom_ExplicitValence_Featurizer()
atom_FormalCharge_featurizer = Atom_FormalCharge_Featurizer()
atom_ImplicitValence_featurizer = Atom_ImplicitValence_Featurizer()
atom_IsAromatic_featurizer = Atom_IsAromatic_Featurizer()
atom_Isotope_featurizer = Atom_Isotope_Featurizer()
atom_Mass_featurizer = Atom_Mass_Featurizer()
atom_NoImplicit_featurizer = Atom_NoImplicit_Featurizer()
atom_NumExplicitHs_featurizer = Atom_NumExplicitHs_Featurizer()
atom_NumImplicitHs_featurizer = Atom_NumImplicitHs_Featurizer()
atom_NumRadicalElectrons_featurizer = Atom_NumRadicalElectrons_Featurizer()
atom_TotalDegree_featurizer = Atom_TotalDegree_Featurizer()
atom_TotalNumHs_featurizer = Atom_TotalNumHs_Featurizer()
atom_TotalValence_featurizer = Atom_TotalValence_Featurizer()
atom_IsInRing_featurizer = Atom_IsInRing_Featurizer()
atom_Chi1_featurizer = Atom_Chi1_Featurizer()
_available_featurizer = {
    'atom_AtomicNum_featurizer': atom_AtomicNum_featurizer,
    'atom_Degree_featurizer': atom_Degree_featurizer,
    'atom_ExplicitValence_featurizer': atom_ExplicitValence_featurizer,
    'atom_FormalCharge_featurizer': atom_FormalCharge_featurizer,
    'atom_ImplicitValence_featurizer': atom_ImplicitValence_featurizer,
    'atom_IsAromatic_featurizer': atom_IsAromatic_featurizer,
    'atom_Isotope_featurizer': atom_Isotope_featurizer,
    'atom_Mass_featurizer': atom_Mass_featurizer,
    'atom_NoImplicit_featurizer': atom_NoImplicit_featurizer,
    'atom_NumExplicitHs_featurizer': atom_NumExplicitHs_featurizer,
    'atom_NumImplicitHs_featurizer': atom_NumImplicitHs_featurizer,
    'atom_NumRadicalElectrons_featurizer': atom_NumRadicalElectrons_featurizer,
    'atom_TotalDegree_featurizer': atom_TotalDegree_featurizer,
    'atom_TotalNumHs_featurizer': atom_TotalNumHs_featurizer,
    'atom_TotalValence_featurizer': atom_TotalValence_featurizer,
    'atom_IsInRing_featurizer': atom_IsInRing_featurizer,
    'atom_Chi1_featurizer': atom_Chi1_featurizer,
}
__all__ = [
    'Atom_AtomicNum_Featurizer',
    'atom_AtomicNum_featurizer',
    'Atom_Degree_Featurizer',
    'atom_Degree_featurizer',
    'Atom_ExplicitValence_Featurizer',
    'atom_ExplicitValence_featurizer',
    'Atom_FormalCharge_Featurizer',
    'atom_FormalCharge_featurizer',
    'Atom_ImplicitValence_Featurizer',
    'atom_ImplicitValence_featurizer',
    'Atom_IsAromatic_Featurizer',
    'atom_IsAromatic_featurizer',
    'Atom_Isotope_Featurizer',
    'atom_Isotope_featurizer',
    'Atom_Mass_Featurizer',
    'atom_Mass_featurizer',
    'Atom_NoImplicit_Featurizer',
    'atom_NoImplicit_featurizer',
    'Atom_NumExplicitHs_Featurizer',
    'atom_NumExplicitHs_featurizer',
    'Atom_NumImplicitHs_Featurizer',
    'atom_NumImplicitHs_featurizer',
    'Atom_NumRadicalElectrons_Featurizer',
    'atom_NumRadicalElectrons_featurizer',
    'Atom_TotalDegree_Featurizer',
    'atom_TotalDegree_featurizer',
    'Atom_TotalNumHs_Featurizer',
    'atom_TotalNumHs_featurizer',
    'Atom_TotalValence_Featurizer',
    'atom_TotalValence_featurizer',
    'Atom_IsInRing_Featurizer',
    'atom_IsInRing_featurizer',
    'Atom_Chi1_Featurizer',
    'atom_Chi1_featurizer',
]


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem
    testdata = prepare_mol_for_featurization(Chem.MolFromSmiles('c1ccccc1')).GetAtoms()[-1]
    for n, f in get_available_featurizer().items():
        print(n, end=' ')
        print(f(testdata))
    print(len(get_available_featurizer()))


if __name__ == '__main__':
    main()
