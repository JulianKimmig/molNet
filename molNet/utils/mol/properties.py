import random

from rdkit.Chem import AllChem, AddHs
from rdkit.Chem.rdchem import Mol

from molNet import ConformerError


def create_conformers(mol: Mol, iterations: int = 100, seed=None) -> Mol:
    if iterations <= 0:
        return mol
    in_iterations = 1
    for i in range(iterations):
        if seed is None:
            randomSeed = random.randint(1, 10 ** 9)
        else:
            randomSeed = seed
        if mol.GetNumConformers() == 0:
            mol = AddHs(mol)
            AllChem.EmbedMolecule(
                mol,
                useRandomCoords=False,
                maxAttempts=in_iterations,
                randomSeed=randomSeed,
            )
        else:
            return mol

        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(
                mol,
                useRandomCoords=True,
                maxAttempts=in_iterations,
                randomSeed=randomSeed,
            )
        else:
            return mol

        if mol.GetNumConformers() == 0:
            ps = AllChem.ETKDGv2()
            ps.maxIterations = in_iterations
            ps.useRandomCoords = True
            ps.randomSeed = randomSeed
            AllChem.EmbedMolecule(mol, ps)
        else:
            return mol

        if mol.GetNumConformers() == 0:
            ps = AllChem.ETKDGv3()
            ps.maxIterations = in_iterations
            ps.useRandomCoords = True
            ps.randomSeed = randomSeed
            AllChem.EmbedMolecule(mol, ps)
        else:
            return mol

        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(
                mol,
                useRandomCoords=True,
                maxAttempts=in_iterations,
                ignoreSmoothingFailures=True,
                randomSeed=randomSeed,
            )
        else:
            return mol

    return mol


def create_conformers_if_needed(mol: Mol, iterations: int = 100) -> Mol:
    if mol.GetNumConformers() > 0:
        return mol
    return create_conformers(mol, iterations=iterations)


def assert_conformers(mol: Mol, iterations: int = 100) -> Mol:
    mol = create_conformers_if_needed(mol, iterations=iterations)

    if mol.GetNumConformers() == 0:
        raise ConformerError("could not generate conformer for molecule")

    return mol


def has_confomers(mol):
    if mol is None:
        return False
    try:
        return mol.GetNumConformers() > 0

    except:
        return False
