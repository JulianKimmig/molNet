from rdkit import Chem
from rdkit.Chem import AllChem


def name_polymer_check(identifier):
    if identifier.lower().startswith("poly(") and identifier.endswith(")"):
        identifier = identifier.replace("Poly(", "").replace("poly(", "")
        identifier = identifier[:-1]
        return identifier, True

    return identifier, False


def detect_polymer_type_by_name(name):
    for polymer in AVAILABLE_POLYMER_TYPES:
        for n in polymer.names:
            if n in name:
                return polymer
    return None


def detect_polymer_type_by_monomer_smiles(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    for pi in AVAILABLE_POLYMER_TYPES:
        if pi.matches_monomer(mol):
            return pi
    return None


def monomer_to_repeating_unit_smiles(smiles, polymer_type=None, deep=False):
    if polymer_type is None:
        polymer_type = detect_polymer_type_by_monomer_smiles(smiles)

    if polymer_type is None:
        return None

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    smiles = Chem.MolToSmiles(polymer_type.monomer_to_polymer(mol))

    if deep:
        n_smiles = True
        while n_smiles is not None:
            n_smiles = monomer_to_repeating_unit_smiles(smiles)
            if n_smiles is not None:
                smiles = n_smiles
    # return type
    return smiles


class Polymerizer:
    def __init__(self, polymer_smarts, monomer_smarts, names):
        self.names = names
        self.monomer_smarts = monomer_smarts
        self.polymer_smarts = polymer_smarts
        self.polymer_mol = Chem.MolFromSmarts(polymer_smarts)
        self.monomer_mol = Chem.MolFromSmarts(monomer_smarts)

        for atom in self.monomer_mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                if int(atom.GetProp('molAtomMapNumber')) >= 1000:
                    atom.ClearProp('molAtomMapNumber')

        for atom in self.polymer_mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                if int(atom.GetProp('molAtomMapNumber')) >= 1000:
                    atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() + 1)
                    atom.ClearProp('molAtomMapNumber')

    def matches_monomer(self, mol):
        return len(mol.GetSubstructMatches(self.monomer_mol)) > 0

    def matches_polymer(self, mol):
        return len(mol.GetSubstructMatches(self.polymer_mol)) > 0

    def monomer_to_polymer(self, mol):
        mod_mol = Chem.ReplaceSubstructs(Chem.AddHs(mol),
                                         self.monomer_mol,
                                         self.polymer_mol,
                                         replacementConnectionPoint=0,
                                         replaceAll=True)
        return mod_mol[0]


METHACRYLATE = Polymerizer(polymer_smarts="OC(=O)[#6:1000]([#6]([#1])([#1])([#1]))[#6:1000]([#1])([#1])",
                           monomer_smarts="OC(=O)[#6:1000]([#6]([#1])([#1])([#1]))=[#6:1000]([#1])([#1])",
                           names=["methacrylat", "methacryl"],
                           )

ACRYLATE = Polymerizer(polymer_smarts="OC(=O)[#6:1000]([#1])[#6:1000]([#1])([#1])",
                       monomer_smarts="OC(=O)[#6:1000]([#1])=[#6:1000]([#1])([#1])",
                       names=["acrylat", "acryl"],
                       )

VINYL = Polymerizer(polymer_smarts="[#6:1000]([#1])[#6:1000]([#1])([#1])",
                    monomer_smarts="[#6:1000]([#1])=[#6:1000]([#1])([#1])",
                    names=["vinyl"],
                    )

AVAILABLE_POLYMER_TYPES = [  # reactivity order important
    ACRYLATE, METHACRYLATE, VINYL,
]
