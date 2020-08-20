from IPython.core.display import display
from rdkit import Chem
from rdkit.Chem import AllChem


class Polymerizer:
    def __init__(self, polymer_smarts, monomer_smarts, names):
        self.names = names
        self.monomer_smarts = monomer_smarts
        self.polymer_smarts = polymer_smarts
        self.polymer_mol = Chem.MolFromSmarts(polymer_smarts)
        self.monomer_mol = Chem.MolFromSmarts(monomer_smarts)
        from IPython.core.display import display

        self.monomer_to_polymer_rxn = AllChem.ReactionFromSmarts(Chem.MolToSmarts(self.monomer_mol)+">>"+Chem.MolToSmarts(self.polymer_mol))

    def matches_monomer(self, mol):
        return len(mol.GetSubstructMatches(self.monomer_mol)) > 0

    def matches_polymer(self, mol):
        return len(mol.GetSubstructMatches(self.polymer_mol)) > 0

    def monomer_to_polymer(self,mol):
        from IPython.core.display import display
        for atom in self.polymer_mol.GetAtoms():
            atom.SetProp('atomLabel',str(atom.GetIdx()))

        display(self.polymer_mol)
        display(mol)
        mod_mol = Chem.ReplaceSubstructs(mol,
                                         self.monomer_mol,
                                         self.polymer_mol,
                                         replacementConnectionPoint=3,
                                         replaceAll=True)
        display(mod_mol[0])
        return self.monomer_to_polymer_rxn.RunReactants((mol,))[0][0]

METHACRYLATE = Polymerizer(polymer_smarts="[CH2][C](C)(C(=O)O)",
                           monomer_smarts="[CH2]=[C](C)(C(=O)O)",
                           names=["methacrylat", "methacryl"],
                           )

ACRYLATE = Polymerizer(polymer_smarts="[CH2][CH](C(=O)O)",
                       monomer_smarts="[C]([H])([H])=[C]([H])(C(=O)O)",
                       names=["acrylat", "sacryl"],
                       )

VINYL = Polymerizer(polymer_smarts="[CH2:1][CH:2]",
                    monomer_smarts="[CH2:1]=[CH:2]",
                    names=["vinyl"],
                    )

AVAILABLE_POLYMER_TYPES = [
    METHACRYLATE, ACRYLATE, VINYL,
]


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
    mol =Chem.AddHs(Chem.MolFromSmiles(smiles))
    for pi in AVAILABLE_POLYMER_TYPES:
        if pi.matches_monomer(mol):
            return pi
    return None


def monomer_to_repeating_unit_smiles(smiles, polymer_type=None):
    display(polymer_type)
    if polymer_type is None:
        polymer_type = detect_polymer_type_by_monomer_smiles(smiles)

    if polymer_type is None:
        return None

    mol =Chem.AddHs(Chem.MolFromSmiles(smiles))
    # return type
    return Chem.MolToSmiles(polymer_type.monomer_to_polymer(mol))
