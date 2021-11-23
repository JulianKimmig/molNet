from rdkit.Chem.AllChem import (CalcMolFormula)

from molNet.featurizer._molecule_featurizer import (StringMoleculeFeaturizer)
from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization


class Molecule_MolToSmiles_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToSmiles
    dtype = str
    featurize = staticmethod(MolToSmiles)


class Molecule_MolToInchiKey_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToInchiKey
    dtype = str
    featurize = staticmethod(MolToInchiKey)


class Molecule_MolToCXSmiles_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToCXSmiles
    dtype = str
    featurize = staticmethod(MolToCXSmiles)


class Molecule_MolToInchiAndAuxInfo_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToInchiAndAuxInfo
    dtype = str
    featurize = staticmethod(MolToInchiAndAuxInfo)


class Molecule_MolToJSON_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToJSON
    dtype = str
    featurize = staticmethod(MolToJSON)


class Molecule_MolToCMLBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToCMLBlock
    dtype = str
    featurize = staticmethod(MolToCMLBlock)


class Molecule_MolToXYZBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToXYZBlock
    dtype = str
    featurize = staticmethod(MolToXYZBlock)


class Molecule_MolToTPLBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToTPLBlock
    dtype = str
    featurize = staticmethod(MolToTPLBlock)


class Molecule_MolToV3KMolBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToV3KMolBlock
    dtype = str
    featurize = staticmethod(MolToV3KMolBlock)


class Molecule_MolToHELM_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToHELM
    dtype = str
    featurize = staticmethod(MolToHELM)


class Molecule_MolToCXSmarts_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToCXSmarts
    dtype = str
    featurize = staticmethod(MolToCXSmarts)


class Molecule_MolToInchi_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToInchi
    dtype = str
    featurize = staticmethod(MolToInchi)


class Molecule_MolFormula_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcMolFormula
    dtype = str
    featurize = staticmethod(CalcMolFormula)


class Molecule_MolToMolBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToMolBlock
    dtype = str
    featurize = staticmethod(MolToMolBlock)


class Molecule_MolToSmarts_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToSmarts
    dtype = str
    featurize = staticmethod(MolToSmarts)


class Molecule_MolToSequence_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToSequence
    dtype = str
    featurize = staticmethod(MolToSequence)


class Molecule_MolToPDBBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToPDBBlock
    dtype = str
    featurize = staticmethod(MolToPDBBlock)


class Molecule_MolToFASTA_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToFASTA
    dtype = str
    featurize = staticmethod(MolToFASTA)


molecule_MolToSmiles_featurizer = Molecule_MolToSmiles_Featurizer()
molecule_MolToInchiKey_featurizer = Molecule_MolToInchiKey_Featurizer()
molecule_MolToCXSmiles_featurizer = Molecule_MolToCXSmiles_Featurizer()
molecule_MolToInchiAndAuxInfo_featurizer = Molecule_MolToInchiAndAuxInfo_Featurizer()
molecule_MolToJSON_featurizer = Molecule_MolToJSON_Featurizer()
molecule_MolToCMLBlock_featurizer = Molecule_MolToCMLBlock_Featurizer()
molecule_MolToXYZBlock_featurizer = Molecule_MolToXYZBlock_Featurizer()
molecule_MolToTPLBlock_featurizer = Molecule_MolToTPLBlock_Featurizer()
molecule_MolToV3KMolBlock_featurizer = Molecule_MolToV3KMolBlock_Featurizer()
molecule_MolToHELM_featurizer = Molecule_MolToHELM_Featurizer()
molecule_MolToCXSmarts_featurizer = Molecule_MolToCXSmarts_Featurizer()
molecule_MolToInchi_featurizer = Molecule_MolToInchi_Featurizer()
molecule_MolFormula_featurizer = Molecule_MolFormula_Featurizer()
molecule_MolToMolBlock_featurizer = Molecule_MolToMolBlock_Featurizer()
molecule_MolToSmarts_featurizer = Molecule_MolToSmarts_Featurizer()
molecule_MolToSequence_featurizer = Molecule_MolToSequence_Featurizer()
molecule_MolToPDBBlock_featurizer = Molecule_MolToPDBBlock_Featurizer()
molecule_MolToFASTA_featurizer = Molecule_MolToFASTA_Featurizer()
_available_featurizer = {
    'molecule_MolToSmiles_featurizer': molecule_MolToSmiles_featurizer,
    'molecule_MolToInchiKey_featurizer': molecule_MolToInchiKey_featurizer,
    'molecule_MolToCXSmiles_featurizer': molecule_MolToCXSmiles_featurizer,
    'molecule_MolToInchiAndAuxInfo_featurizer': molecule_MolToInchiAndAuxInfo_featurizer,
    'molecule_MolToJSON_featurizer': molecule_MolToJSON_featurizer,
    'molecule_MolToCMLBlock_featurizer': molecule_MolToCMLBlock_featurizer,
    'molecule_MolToXYZBlock_featurizer': molecule_MolToXYZBlock_featurizer,
    'molecule_MolToTPLBlock_featurizer': molecule_MolToTPLBlock_featurizer,
    'molecule_MolToV3KMolBlock_featurizer': molecule_MolToV3KMolBlock_featurizer,
    'molecule_MolToHELM_featurizer': molecule_MolToHELM_featurizer,
    'molecule_MolToCXSmarts_featurizer': molecule_MolToCXSmarts_featurizer,
    'molecule_MolToInchi_featurizer': molecule_MolToInchi_featurizer,
    'molecule_MolFormula_featurizer': molecule_MolFormula_featurizer,
    'molecule_MolToMolBlock_featurizer': molecule_MolToMolBlock_featurizer,
    'molecule_MolToSmarts_featurizer': molecule_MolToSmarts_featurizer,
    'molecule_MolToSequence_featurizer': molecule_MolToSequence_featurizer,
    'molecule_MolToPDBBlock_featurizer': molecule_MolToPDBBlock_featurizer,
    'molecule_MolToFASTA_featurizer': molecule_MolToFASTA_featurizer,
}
__all__ = [
    'Molecule_MolToSmiles_Featurizer',
    'molecule_MolToSmiles_featurizer',
    'Molecule_MolToInchiKey_Featurizer',
    'molecule_MolToInchiKey_featurizer',
    'Molecule_MolToCXSmiles_Featurizer',
    'molecule_MolToCXSmiles_featurizer',
    'Molecule_MolToInchiAndAuxInfo_Featurizer',
    'molecule_MolToInchiAndAuxInfo_featurizer',
    'Molecule_MolToJSON_Featurizer',
    'molecule_MolToJSON_featurizer',
    'Molecule_MolToCMLBlock_Featurizer',
    'molecule_MolToCMLBlock_featurizer',
    'Molecule_MolToXYZBlock_Featurizer',
    'molecule_MolToXYZBlock_featurizer',
    'Molecule_MolToTPLBlock_Featurizer',
    'molecule_MolToTPLBlock_featurizer',
    'Molecule_MolToV3KMolBlock_Featurizer',
    'molecule_MolToV3KMolBlock_featurizer',
    'Molecule_MolToHELM_Featurizer',
    'molecule_MolToHELM_featurizer',
    'Molecule_MolToCXSmarts_Featurizer',
    'molecule_MolToCXSmarts_featurizer',
    'Molecule_MolToInchi_Featurizer',
    'molecule_MolToInchi_featurizer',
    'Molecule_MolFormula_Featurizer',
    'molecule_MolFormula_featurizer',
    'Molecule_MolToMolBlock_Featurizer',
    'molecule_MolToMolBlock_featurizer',
    'Molecule_MolToSmarts_Featurizer',
    'molecule_MolToSmarts_featurizer',
    'Molecule_MolToSequence_Featurizer',
    'molecule_MolToSequence_featurizer',
    'Molecule_MolToPDBBlock_Featurizer',
    'molecule_MolToPDBBlock_featurizer',
    'Molecule_MolToFASTA_Featurizer',
    'molecule_MolToFASTA_featurizer',
]


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem
    testdata = prepare_mol_for_featurization(Chem.MolFromSmiles('c1ccccc1'))
    for n, f in get_available_featurizer().items():
        print(n, end=' ')
        print(f(testdata))
    print(len(get_available_featurizer()))


if __name__ == '__main__':
    main()
