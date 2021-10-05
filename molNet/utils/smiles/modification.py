from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from tqdm import tqdm

from molNet.utils.parallelization.multiprocessing import parallelize
from molNet.utils.smiles import mol_from_smiles
from molNet import MolGenerationError, MOLNET_LOGGER


def get_random_smiles(smiles, max_smiles=1000):
    mol = mol_from_smiles(smiles)
    s = []
    possibilities = []
    for m in [mol, Chem.RemoveHs(mol), Chem.AddHs(mol)]:
        for allBondsExplicit in [True, False]:
            for allHsExplicit in [True, False]:
                # for kekuleSmiles in [True,False ]: # TODO why keklulization not worlking? -> produces bad SMILES
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


def multiple_mol_from_smiles_generator(
    smiles: List[str], raise_error: bool = True, progess_bar=True
):
    gen = (mol_from_smiles(s, raise_error=raise_error) for s in smiles)
    if progess_bar:
        gen = tqdm(
            gen, total=len(smiles), bar_format="{l_bar}{bar}{r_bar}", unit=" mol"
        )
    for g in gen:
        yield g


def multiple_mol_from_smiles(
    smiles: List[str], raise_error: bool = True, progess_bar=True
) -> List[Mol]:
    return [
        m
        for m in multiple_mol_from_smiles_generator(
            smiles=smiles, raise_error=raise_error, progess_bar=progess_bar
        )
    ]


def _mfs(smiles):
    return [mol_from_smiles(s, raise_error=False) for s in smiles]


def parallel_mol_from_smiles(smiles, cores="all-1", progess_bar=True):
    return parallelize(
        _mfs,
        smiles,
        cores=cores,
        progess_bar=progess_bar,
        progress_bar_kwargs=dict(unit=" mol"),
    )
