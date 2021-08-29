import numpy as np
from rdkit import Chem

from molNet import MolGenerationError


def get_random_smiles(smiles, max_smiles=1000):
    mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
    if mol is None:
        raise MolGenerationError(
            "cannot convert smiles '{}' to molecule".format(smiles)
        )
    s = []
    possibilities = []
    for m in [mol, Chem.AddHs(mol), Chem.RemoveHs(mol)]:
        for allBondsExplicit in [False, True]:
            for allHsExplicit in [False, True]:
                # for kekuleSmiles in [False, True]: # TODO why keklulization not worlking? -> produces bad SMILES
                for isomericSmiles in [True, False]:
                    for canonical in [True, False]:
                        for atom in m.GetAtoms():
                            possibilities.append(
                                dict(
                                    mol=m,
                                    canonical=canonical,
                                    allHsExplicit=allHsExplicit,
                                    rootedAtAtom=atom.GetIdx(),
                                    allBondsExplicit=allBondsExplicit,
                                    # kekuleSmiles=kekuleSmiles,
                                    isomericSmiles=isomericSmiles,
                                )
                            )

    indices = np.arange(len(possibilities))
    np.random.shuffle(indices)
    i = 0
    for idx in indices:
        try:
            ns = Chem.MolToSmiles(**possibilities[idx])
            if ns not in s:
                i += 1
                s.append(ns)
                if i >= max_smiles:
                    return s[:max_smiles]
        except:
            pass
    return s[:max_smiles]
