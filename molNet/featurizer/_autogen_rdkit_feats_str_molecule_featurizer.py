import numpy as np
from molNet.featurizer._molecule_featurizer import (
    FixedSizeMoleculeFeaturizer,
    StringMoleculeFeaturizer,
    VarSizeMoleculeFeaturizer,
)
from rdkit.Chem.AllChem import (
    MolToCMLBlock,
    MolToSmarts,
    MolToSequence,
    MolToCXSmarts,
    MolToCXSmiles,
    CalcMolFormula,
    MolToPDBBlock,
    MolToFASTA,
    MolToHELM,
    MolToSmiles,
)
from rdkit.Chem import (
    MolToCMLBlock,
    MolToSmarts,
    MolToSequence,
    MolToTPLBlock,
    MolToXYZBlock,
    MolToCXSmarts,
    MolToCXSmiles,
    MolToInchiKey,
    MolToInchi,
    MolToMolBlock,
    MolToV3KMolBlock,
    MolToJSON,
    MolToPDBBlock,
    MolToInchiAndAuxInfo,
    MolToFASTA,
    MolToHELM,
    MolToSmiles,
)
from rdkit.Chem.inchi import MolToInchiKey, MolToInchi, MolToInchiAndAuxInfo
from rdkit.Chem.rdMolInterchange import MolToJSON
from rdkit.Chem.rdinchi import MolToInchiKey
from rdkit.Chem.rdmolfiles import (
    MolToMolBlock,
    MolToTPLBlock,
    MolToV3KMolBlock,
    MolToXYZBlock,
)


class Molecule_AllChem_MolFormula_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcMolFormula
    dtype = str
    featurize = staticmethod(CalcMolFormula)


class Molecule_AllChem_MolToCMLBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.MolToCMLBlock
    dtype = str
    featurize = staticmethod(MolToCMLBlock)


class Molecule_AllChem_MolToCXSmarts_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.MolToCXSmarts
    dtype = str
    featurize = staticmethod(MolToCXSmarts)


class Molecule_AllChem_MolToCXSmiles_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.MolToCXSmiles
    dtype = str
    featurize = staticmethod(MolToCXSmiles)


class Molecule_AllChem_MolToFASTA_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.MolToFASTA
    dtype = str
    featurize = staticmethod(MolToFASTA)


class Molecule_AllChem_MolToHELM_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.MolToHELM
    dtype = str
    featurize = staticmethod(MolToHELM)


class Molecule_AllChem_MolToPDBBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.MolToPDBBlock
    dtype = str
    featurize = staticmethod(MolToPDBBlock)


class Molecule_AllChem_MolToSequence_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.MolToSequence
    dtype = str
    featurize = staticmethod(MolToSequence)


class Molecule_AllChem_MolToSmarts_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.MolToSmarts
    dtype = str
    featurize = staticmethod(MolToSmarts)


class Molecule_AllChem_MolToSmiles_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.MolToSmiles
    dtype = str
    featurize = staticmethod(MolToSmiles)


class Molecule_Chem_MolToCMLBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToCMLBlock
    dtype = str
    featurize = staticmethod(MolToCMLBlock)


class Molecule_Chem_MolToCXSmarts_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToCXSmarts
    dtype = str
    featurize = staticmethod(MolToCXSmarts)


class Molecule_Chem_MolToCXSmiles_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToCXSmiles
    dtype = str
    featurize = staticmethod(MolToCXSmiles)


class Molecule_Chem_MolToFASTA_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToFASTA
    dtype = str
    featurize = staticmethod(MolToFASTA)


class Molecule_Chem_MolToHELM_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToHELM
    dtype = str
    featurize = staticmethod(MolToHELM)


class Molecule_Chem_MolToInchiAndAuxInfo_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToInchiAndAuxInfo
    dtype = str
    featurize = staticmethod(MolToInchiAndAuxInfo)


class Molecule_Chem_MolToInchiKey_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToInchiKey
    dtype = str
    featurize = staticmethod(MolToInchiKey)


class Molecule_Chem_MolToInchi_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToInchi
    dtype = str
    featurize = staticmethod(MolToInchi)


class Molecule_Chem_MolToJSON_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToJSON
    dtype = str
    featurize = staticmethod(MolToJSON)


class Molecule_Chem_MolToMolBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToMolBlock
    dtype = str
    featurize = staticmethod(MolToMolBlock)


class Molecule_Chem_MolToPDBBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToPDBBlock
    dtype = str
    featurize = staticmethod(MolToPDBBlock)


class Molecule_Chem_MolToSequence_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToSequence
    dtype = str
    featurize = staticmethod(MolToSequence)


class Molecule_Chem_MolToSmarts_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToSmarts
    dtype = str
    featurize = staticmethod(MolToSmarts)


class Molecule_Chem_MolToSmiles_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToSmiles
    dtype = str
    featurize = staticmethod(MolToSmiles)


class Molecule_Chem_MolToTPLBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToTPLBlock
    dtype = str
    featurize = staticmethod(MolToTPLBlock)


class Molecule_Chem_MolToV3KMolBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToV3KMolBlock
    dtype = str
    featurize = staticmethod(MolToV3KMolBlock)


class Molecule_Chem_MolToXYZBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.MolToXYZBlock
    dtype = str
    featurize = staticmethod(MolToXYZBlock)


class Molecule_inchi_MolToInchiAndAuxInfo_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.inchi.MolToInchiAndAuxInfo
    dtype = str
    featurize = staticmethod(MolToInchiAndAuxInfo)


class Molecule_inchi_MolToInchiKey_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.inchi.MolToInchiKey
    dtype = str
    featurize = staticmethod(MolToInchiKey)


class Molecule_inchi_MolToInchi_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.inchi.MolToInchi
    dtype = str
    featurize = staticmethod(MolToInchi)


class Molecule_rdMolInterchange_MolToJSON_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.rdMolInterchange.MolToJSON
    dtype = str
    featurize = staticmethod(MolToJSON)


class Molecule_rdinchi_MolToInchiKey_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.rdinchi.MolToInchiKey
    dtype = str
    featurize = staticmethod(MolToInchiKey)


class Molecule_rdmolfiles_MolToMolBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.rdmolfiles.MolToMolBlock
    dtype = str
    featurize = staticmethod(MolToMolBlock)


class Molecule_rdmolfiles_MolToTPLBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.rdmolfiles.MolToTPLBlock
    dtype = str
    featurize = staticmethod(MolToTPLBlock)


class Molecule_rdmolfiles_MolToV3KMolBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.rdmolfiles.MolToV3KMolBlock
    dtype = str
    featurize = staticmethod(MolToV3KMolBlock)


class Molecule_rdmolfiles_MolToXYZBlock_Featurizer(StringMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.rdmolfiles.MolToXYZBlock
    dtype = str
    featurize = staticmethod(MolToXYZBlock)


molecule_AllChem_MolFormula_featurizer = Molecule_AllChem_MolFormula_Featurizer()
molecule_AllChem_MolToCMLBlock_featurizer = Molecule_AllChem_MolToCMLBlock_Featurizer()
molecule_AllChem_MolToCXSmarts_featurizer = Molecule_AllChem_MolToCXSmarts_Featurizer()
molecule_AllChem_MolToCXSmiles_featurizer = Molecule_AllChem_MolToCXSmiles_Featurizer()
molecule_AllChem_MolToFASTA_featurizer = Molecule_AllChem_MolToFASTA_Featurizer()
molecule_AllChem_MolToHELM_featurizer = Molecule_AllChem_MolToHELM_Featurizer()
molecule_AllChem_MolToPDBBlock_featurizer = Molecule_AllChem_MolToPDBBlock_Featurizer()
molecule_AllChem_MolToSequence_featurizer = Molecule_AllChem_MolToSequence_Featurizer()
molecule_AllChem_MolToSmarts_featurizer = Molecule_AllChem_MolToSmarts_Featurizer()
molecule_AllChem_MolToSmiles_featurizer = Molecule_AllChem_MolToSmiles_Featurizer()
molecule_Chem_MolToCMLBlock_featurizer = Molecule_Chem_MolToCMLBlock_Featurizer()
molecule_Chem_MolToCXSmarts_featurizer = Molecule_Chem_MolToCXSmarts_Featurizer()
molecule_Chem_MolToCXSmiles_featurizer = Molecule_Chem_MolToCXSmiles_Featurizer()
molecule_Chem_MolToFASTA_featurizer = Molecule_Chem_MolToFASTA_Featurizer()
molecule_Chem_MolToHELM_featurizer = Molecule_Chem_MolToHELM_Featurizer()
molecule_Chem_MolToInchiAndAuxInfo_featurizer = (
    Molecule_Chem_MolToInchiAndAuxInfo_Featurizer()
)
molecule_Chem_MolToInchiKey_featurizer = Molecule_Chem_MolToInchiKey_Featurizer()
molecule_Chem_MolToInchi_featurizer = Molecule_Chem_MolToInchi_Featurizer()
molecule_Chem_MolToJSON_featurizer = Molecule_Chem_MolToJSON_Featurizer()
molecule_Chem_MolToMolBlock_featurizer = Molecule_Chem_MolToMolBlock_Featurizer()
molecule_Chem_MolToPDBBlock_featurizer = Molecule_Chem_MolToPDBBlock_Featurizer()
molecule_Chem_MolToSequence_featurizer = Molecule_Chem_MolToSequence_Featurizer()
molecule_Chem_MolToSmarts_featurizer = Molecule_Chem_MolToSmarts_Featurizer()
molecule_Chem_MolToSmiles_featurizer = Molecule_Chem_MolToSmiles_Featurizer()
molecule_Chem_MolToTPLBlock_featurizer = Molecule_Chem_MolToTPLBlock_Featurizer()
molecule_Chem_MolToV3KMolBlock_featurizer = Molecule_Chem_MolToV3KMolBlock_Featurizer()
molecule_Chem_MolToXYZBlock_featurizer = Molecule_Chem_MolToXYZBlock_Featurizer()
molecule_inchi_MolToInchiAndAuxInfo_featurizer = (
    Molecule_inchi_MolToInchiAndAuxInfo_Featurizer()
)
molecule_inchi_MolToInchiKey_featurizer = Molecule_inchi_MolToInchiKey_Featurizer()
molecule_inchi_MolToInchi_featurizer = Molecule_inchi_MolToInchi_Featurizer()
molecule_rdMolInterchange_MolToJSON_featurizer = (
    Molecule_rdMolInterchange_MolToJSON_Featurizer()
)
molecule_rdinchi_MolToInchiKey_featurizer = Molecule_rdinchi_MolToInchiKey_Featurizer()
molecule_rdmolfiles_MolToMolBlock_featurizer = (
    Molecule_rdmolfiles_MolToMolBlock_Featurizer()
)
molecule_rdmolfiles_MolToTPLBlock_featurizer = (
    Molecule_rdmolfiles_MolToTPLBlock_Featurizer()
)
molecule_rdmolfiles_MolToV3KMolBlock_featurizer = (
    Molecule_rdmolfiles_MolToV3KMolBlock_Featurizer()
)
molecule_rdmolfiles_MolToXYZBlock_featurizer = (
    Molecule_rdmolfiles_MolToXYZBlock_Featurizer()
)
_available_featurizer = {
    "molecule_AllChem_MolFormula_featurizer": molecule_AllChem_MolFormula_featurizer,
    "molecule_AllChem_MolToCMLBlock_featurizer": molecule_AllChem_MolToCMLBlock_featurizer,
    "molecule_AllChem_MolToCXSmarts_featurizer": molecule_AllChem_MolToCXSmarts_featurizer,
    "molecule_AllChem_MolToCXSmiles_featurizer": molecule_AllChem_MolToCXSmiles_featurizer,
    "molecule_AllChem_MolToFASTA_featurizer": molecule_AllChem_MolToFASTA_featurizer,
    "molecule_AllChem_MolToHELM_featurizer": molecule_AllChem_MolToHELM_featurizer,
    "molecule_AllChem_MolToPDBBlock_featurizer": molecule_AllChem_MolToPDBBlock_featurizer,
    "molecule_AllChem_MolToSequence_featurizer": molecule_AllChem_MolToSequence_featurizer,
    "molecule_AllChem_MolToSmarts_featurizer": molecule_AllChem_MolToSmarts_featurizer,
    "molecule_AllChem_MolToSmiles_featurizer": molecule_AllChem_MolToSmiles_featurizer,
    "molecule_Chem_MolToCMLBlock_featurizer": molecule_Chem_MolToCMLBlock_featurizer,
    "molecule_Chem_MolToCXSmarts_featurizer": molecule_Chem_MolToCXSmarts_featurizer,
    "molecule_Chem_MolToCXSmiles_featurizer": molecule_Chem_MolToCXSmiles_featurizer,
    "molecule_Chem_MolToFASTA_featurizer": molecule_Chem_MolToFASTA_featurizer,
    "molecule_Chem_MolToHELM_featurizer": molecule_Chem_MolToHELM_featurizer,
    "molecule_Chem_MolToInchiAndAuxInfo_featurizer": molecule_Chem_MolToInchiAndAuxInfo_featurizer,
    "molecule_Chem_MolToInchiKey_featurizer": molecule_Chem_MolToInchiKey_featurizer,
    "molecule_Chem_MolToInchi_featurizer": molecule_Chem_MolToInchi_featurizer,
    "molecule_Chem_MolToJSON_featurizer": molecule_Chem_MolToJSON_featurizer,
    "molecule_Chem_MolToMolBlock_featurizer": molecule_Chem_MolToMolBlock_featurizer,
    "molecule_Chem_MolToPDBBlock_featurizer": molecule_Chem_MolToPDBBlock_featurizer,
    "molecule_Chem_MolToSequence_featurizer": molecule_Chem_MolToSequence_featurizer,
    "molecule_Chem_MolToSmarts_featurizer": molecule_Chem_MolToSmarts_featurizer,
    "molecule_Chem_MolToSmiles_featurizer": molecule_Chem_MolToSmiles_featurizer,
    "molecule_Chem_MolToTPLBlock_featurizer": molecule_Chem_MolToTPLBlock_featurizer,
    "molecule_Chem_MolToV3KMolBlock_featurizer": molecule_Chem_MolToV3KMolBlock_featurizer,
    "molecule_Chem_MolToXYZBlock_featurizer": molecule_Chem_MolToXYZBlock_featurizer,
    "molecule_inchi_MolToInchiAndAuxInfo_featurizer": molecule_inchi_MolToInchiAndAuxInfo_featurizer,
    "molecule_inchi_MolToInchiKey_featurizer": molecule_inchi_MolToInchiKey_featurizer,
    "molecule_inchi_MolToInchi_featurizer": molecule_inchi_MolToInchi_featurizer,
    "molecule_rdMolInterchange_MolToJSON_featurizer": molecule_rdMolInterchange_MolToJSON_featurizer,
    "molecule_rdinchi_MolToInchiKey_featurizer": molecule_rdinchi_MolToInchiKey_featurizer,
    "molecule_rdmolfiles_MolToMolBlock_featurizer": molecule_rdmolfiles_MolToMolBlock_featurizer,
    "molecule_rdmolfiles_MolToTPLBlock_featurizer": molecule_rdmolfiles_MolToTPLBlock_featurizer,
    "molecule_rdmolfiles_MolToV3KMolBlock_featurizer": molecule_rdmolfiles_MolToV3KMolBlock_featurizer,
    "molecule_rdmolfiles_MolToXYZBlock_featurizer": molecule_rdmolfiles_MolToXYZBlock_featurizer,
}
__all__ = [
    "Molecule_AllChem_MolFormula_Featurizer",
    "molecule_AllChem_MolFormula_featurizer",
    "Molecule_AllChem_MolToCMLBlock_Featurizer",
    "molecule_AllChem_MolToCMLBlock_featurizer",
    "Molecule_AllChem_MolToCXSmarts_Featurizer",
    "molecule_AllChem_MolToCXSmarts_featurizer",
    "Molecule_AllChem_MolToCXSmiles_Featurizer",
    "molecule_AllChem_MolToCXSmiles_featurizer",
    "Molecule_AllChem_MolToFASTA_Featurizer",
    "molecule_AllChem_MolToFASTA_featurizer",
    "Molecule_AllChem_MolToHELM_Featurizer",
    "molecule_AllChem_MolToHELM_featurizer",
    "Molecule_AllChem_MolToPDBBlock_Featurizer",
    "molecule_AllChem_MolToPDBBlock_featurizer",
    "Molecule_AllChem_MolToSequence_Featurizer",
    "molecule_AllChem_MolToSequence_featurizer",
    "Molecule_AllChem_MolToSmarts_Featurizer",
    "molecule_AllChem_MolToSmarts_featurizer",
    "Molecule_AllChem_MolToSmiles_Featurizer",
    "molecule_AllChem_MolToSmiles_featurizer",
    "Molecule_Chem_MolToCMLBlock_Featurizer",
    "molecule_Chem_MolToCMLBlock_featurizer",
    "Molecule_Chem_MolToCXSmarts_Featurizer",
    "molecule_Chem_MolToCXSmarts_featurizer",
    "Molecule_Chem_MolToCXSmiles_Featurizer",
    "molecule_Chem_MolToCXSmiles_featurizer",
    "Molecule_Chem_MolToFASTA_Featurizer",
    "molecule_Chem_MolToFASTA_featurizer",
    "Molecule_Chem_MolToHELM_Featurizer",
    "molecule_Chem_MolToHELM_featurizer",
    "Molecule_Chem_MolToInchiAndAuxInfo_Featurizer",
    "molecule_Chem_MolToInchiAndAuxInfo_featurizer",
    "Molecule_Chem_MolToInchiKey_Featurizer",
    "molecule_Chem_MolToInchiKey_featurizer",
    "Molecule_Chem_MolToInchi_Featurizer",
    "molecule_Chem_MolToInchi_featurizer",
    "Molecule_Chem_MolToJSON_Featurizer",
    "molecule_Chem_MolToJSON_featurizer",
    "Molecule_Chem_MolToMolBlock_Featurizer",
    "molecule_Chem_MolToMolBlock_featurizer",
    "Molecule_Chem_MolToPDBBlock_Featurizer",
    "molecule_Chem_MolToPDBBlock_featurizer",
    "Molecule_Chem_MolToSequence_Featurizer",
    "molecule_Chem_MolToSequence_featurizer",
    "Molecule_Chem_MolToSmarts_Featurizer",
    "molecule_Chem_MolToSmarts_featurizer",
    "Molecule_Chem_MolToSmiles_Featurizer",
    "molecule_Chem_MolToSmiles_featurizer",
    "Molecule_Chem_MolToTPLBlock_Featurizer",
    "molecule_Chem_MolToTPLBlock_featurizer",
    "Molecule_Chem_MolToV3KMolBlock_Featurizer",
    "molecule_Chem_MolToV3KMolBlock_featurizer",
    "Molecule_Chem_MolToXYZBlock_Featurizer",
    "molecule_Chem_MolToXYZBlock_featurizer",
    "Molecule_inchi_MolToInchiAndAuxInfo_Featurizer",
    "molecule_inchi_MolToInchiAndAuxInfo_featurizer",
    "Molecule_inchi_MolToInchiKey_Featurizer",
    "molecule_inchi_MolToInchiKey_featurizer",
    "Molecule_inchi_MolToInchi_Featurizer",
    "molecule_inchi_MolToInchi_featurizer",
    "Molecule_rdMolInterchange_MolToJSON_Featurizer",
    "molecule_rdMolInterchange_MolToJSON_featurizer",
    "Molecule_rdinchi_MolToInchiKey_Featurizer",
    "molecule_rdinchi_MolToInchiKey_featurizer",
    "Molecule_rdmolfiles_MolToMolBlock_Featurizer",
    "molecule_rdmolfiles_MolToMolBlock_featurizer",
    "Molecule_rdmolfiles_MolToTPLBlock_Featurizer",
    "molecule_rdmolfiles_MolToTPLBlock_featurizer",
    "Molecule_rdmolfiles_MolToV3KMolBlock_Featurizer",
    "molecule_rdmolfiles_MolToV3KMolBlock_featurizer",
    "Molecule_rdmolfiles_MolToXYZBlock_Featurizer",
    "molecule_rdmolfiles_MolToXYZBlock_featurizer",
]


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem
    from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization

    testdata = prepare_mol_for_featurization(Chem.MolFromSmiles("c1ccccc1"))
    for n, f in get_available_featurizer().items():
        print(n, end=" ")
        print(f(testdata))
    print(len(get_available_featurizer()))


if __name__ == "__main__":
    main()
