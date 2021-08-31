from rdkit.Chem import AllChem, AddHs
from rdkit.Chem.rdchem import Mol

from molNet import ConformerError


def assert_confomers(mol: Mol, iterations: int = 10) -> Mol:
    if iterations <= 0:
        return mol

    if mol.GetNumConformers() == 0:
        mol = AddHs(mol)
        AllChem.EmbedMolecule(mol, useRandomCoords=False, maxAttempts=iterations)
    else:
        return mol

    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=iterations)

    if mol.GetNumConformers() == 0:
        ps = AllChem.ETKDGv2()
        ps.maxIterations = iterations
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)

    if mol.GetNumConformers() == 0:
        ps = AllChem.ETKDGv3()
        ps.maxIterations = iterations
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)

    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(
            mol,
            useRandomCoords=True,
            maxAttempts=iterations,
            ignoreSmoothingFailures=True,
        )

    if mol.GetNumConformers() == 0:
        raise ConformerError("could not generate conformer for molecule")

    if mol.GetNumConformers() > 0:
        AllChem.MMFFOptimizeMolecule(mol)

    return mol


def has_confomers(mol, create_if_not=True, *args, **kwargs):
    if mol is None:
        return False
    try:
        if create_if_not:
            assert_confomers(mol, *args, **kwargs)
        return mol.GetNumConformers() > 0

    except:
        return False
