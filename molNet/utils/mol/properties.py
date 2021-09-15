from rdkit.Chem import AllChem, AddHs
from rdkit.Chem.rdchem import Mol

from molNet import ConformerError


def create_conformers(mol: Mol, iterations: int = 10) -> Mol:
    if iterations <= 0:
        return mol

    if mol.GetNumConformers() == 0:
        mol = AddHs(mol)
        AllChem.EmbedMolecule(mol, useRandomCoords=False, maxAttempts=iterations)
    else:
        return mol

    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=iterations)
    else:
        return mol

    if mol.GetNumConformers() == 0:
        ps = AllChem.ETKDGv2()
        ps.maxIterations = iterations
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
    else:
        return mol

    if mol.GetNumConformers() == 0:
        ps = AllChem.ETKDGv3()
        ps.maxIterations = iterations
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
    else:
        return mol

    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(
            mol,
            useRandomCoords=True,
            maxAttempts=iterations,
            ignoreSmoothingFailures=True,
        )
    else:
        return mol

    return mol


def create_conformers_if_needed(mol: Mol, iterations: int = 10) -> Mol:
    if mol.GetNumConformers() > 0:
        return mol
    return create_conformers(mol, iterations=iterations)


def assert_conformers(mol: Mol, iterations: int = 10) -> Mol:
    mol = create_conformers_if_needed(mol, iterations=iterations)

    if mol.GetNumConformers() == 0:
        raise ConformerError("could not generate conformer for molecule")

    return mol


def has_confomers(mol, create_if_not=True, *args, **kwargs):
    if mol is None:
        return False
    try:
        if create_if_not:
            assert_conformers(mol, **kwargs)
        return mol.GetNumConformers() > 0

    except:
        return False
