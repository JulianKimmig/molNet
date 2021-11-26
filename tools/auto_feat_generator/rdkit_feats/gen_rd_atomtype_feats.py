import black

from molNet.utils.mol import ATOMIC_SYMBOL_NUMBERS


def genrate_mol_feats():
    absnumcode = """
class Molecule_NumberAtoms{atom_symbol}_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.uint32
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom({atom_number})))
    
"""
    relnumcode = """
class Molecule_RelativeContent{atom_symbol}_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = np.float32
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom({atom_number})))/ mol.GetNumAtoms()

"""
    hascode = """
class Molecule_Has{atom_symbol}_Featurizer(SingleValueMoleculeFeaturizer):
    dtype = bool
    def featurize(self, mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom({atom_number})))>0

"""

    classes = []
    instances = []
    avail = []
    alls = []
    for atom_symbol, atom_number in ATOMIC_SYMBOL_NUMBERS.items():
        if atom_symbol == "*":
            atom_symbol = "Rgroup"
        classes.append(absnumcode.format(atom_symbol=atom_symbol, atom_number=atom_number))
        classes.append(relnumcode.format(atom_symbol=atom_symbol, atom_number=atom_number))
        classes.append(hascode.format(atom_symbol=atom_symbol, atom_number=atom_number))
        instances.append(
            f"molecule_NumberAtoms{atom_symbol}_featurizer = Molecule_NumberAtoms{atom_symbol}_Featurizer()")
        instances.append(
            f"molecule_RelativeContent{atom_symbol}_featurizer = Molecule_RelativeContent{atom_symbol}_Featurizer()")
        instances.append(f"molecule_Has{atom_symbol}_featurizer = Molecule_Has{atom_symbol}_Featurizer()")
        avail.append(
            f"    'molecule_NumberAtoms{atom_symbol}_featurizer': molecule_NumberAtoms{atom_symbol}_featurizer,")
        avail.append(
            f"    'molecule_RelativeContent{atom_symbol}_featurizer': molecule_RelativeContent{atom_symbol}_featurizer,")
        avail.append(f"    'molecule_Has{atom_symbol}_featurizer': molecule_Has{atom_symbol}_featurizer,")
        alls.append(f"    'Molecule_NumberAtoms{atom_symbol}_Featurizer',")
        alls.append(f"    'molecule_NumberAtoms{atom_symbol}_featurizer',")
        alls.append(f"    'Molecule_RelativeContent{atom_symbol}_Featurizer',")
        alls.append(f"    'molecule_RelativeContent{atom_symbol}_featurizer',")
        alls.append(f"    'Molecule_Has{atom_symbol}_Featurizer',")
        alls.append(f"    'molecule_Has{atom_symbol}_featurizer',")

    code = "import numpy as np\n" \
           "from rdkit.Chem import rdqueries\n" \
           "from molNet.featurizer._molecule_featurizer import SingleValueMoleculeFeaturizer\n"

    for c in classes:
        code += c
    code += "\n".join(instances)

    code += "\n_available_featurizer = {\n"
    code += "\n".join(avail)
    code += "\n}\n"

    code += "\n__all__ = [\n"
    code += "\n".join(alls)
    code += "\n]\n"

    code += """
def get_available_featurizer():
    return _available_featurizer
def main():
    from rdkit import Chem
    from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization
    testdata = prepare_mol_for_featurization(Chem.MolFromSmiles('c1ccccc1'))
    for n, f in get_available_featurizer().items():
        print(n, f(testdata))
    print(len(get_available_featurizer()))
if __name__ == '__main__':
    main()
    """

    code = black.format_str(code, mode=black.FileMode())
    with open("_autogen_rdkit_atomtype_molecule_featurizer.py", "w+") as f:
        f.write(code)


def genrate_atom_feats():
    isscode = """
class Atom_Is{atom_symbol}_Featurizer(SingleValueAtomFeaturizer):
    dtype = bool
    def featurize(self, atom):
        return atom.GetAtomicNum() == {atom_number}

"""

    classes = []
    instances = []
    avail = []
    alls = []
    for atom_symbol, atom_number in ATOMIC_SYMBOL_NUMBERS.items():
        if atom_symbol == "*":
            atom_symbol = "Rgroup"
        classes.append(isscode.format(atom_symbol=atom_symbol, atom_number=atom_number))
        instances.append(f"atom_Is{atom_symbol}_featurizer = Atom_Is{atom_symbol}_Featurizer()")
        avail.append(f"    'atom_Is{atom_symbol}_featurizer': atom_Is{atom_symbol}_featurizer,")
        alls.append(f"    'Atom_Is{atom_symbol}_Featurizer',")
        alls.append(f"    'atom_Is{atom_symbol}_featurizer',")

    classes.append(f"""
class Atom_AllSymbolOneHot_Featurizer(OneHotAtomFeaturizer):
    POSSIBLE_VALUES = ['{"','".join(ATOMIC_SYMBOL_NUMBERS.keys())}']
    def featurize(self,atom):
        return atom.GetSymbol()

""")
    instances.append(f"atom_AllSymbolOneHot_featurizer = Atom_AllSymbolOneHot_Featurizer()")
    avail.append(f"    'atom_AllSymbolOneHot_featurizer': atom_AllSymbolOneHot_featurizer,")
    alls.append(f"    'Atom_AllSymbolOneHot_Featurizer',")
    alls.append(f"    'atom_AllSymbolOneHot_featurizer',")

    code = "import numpy as np\n" \
           "from rdkit.Chem import rdqueries\n" \
           "from molNet.featurizer._atom_featurizer import SingleValueAtomFeaturizer, OneHotAtomFeaturizer\n"

    for c in classes:
        code += c
    code += "\n".join(instances)

    code += "\n_available_featurizer = {\n"
    code += "\n".join(avail)
    code += "\n}\n"

    code += "\n__all__ = [\n"
    code += "\n".join(alls)
    code += "\n]\n"

    code += """
def get_available_featurizer():
    return _available_featurizer
def main():
    from rdkit import Chem
    from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization
    testdata = prepare_mol_for_featurization(Chem.MolFromSmiles('c1ccccc1').GetAtoms()[0])
    for n, f in get_available_featurizer().items():
        print(n, f(testdata))
    print(len(get_available_featurizer()))
if __name__ == '__main__':
    main()
    """
    code = black.format_str(code, mode=black.FileMode())
    with open("_autogen_rdkit_atomtype_atom_featurizer.py", "w+") as f:
        f.write(code)


def main():
    genrate_mol_feats()
    genrate_atom_feats()
    print(ATOMIC_SYMBOL_NUMBERS)


if __name__ == '__main__':
    main()
