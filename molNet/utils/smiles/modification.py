from multiprocessing import cpu_count, Pool
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm

from molNet import MolGenerationError, MOLNET_LOGGER


def get_random_smiles(smiles, max_smiles=1000):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise MolGenerationError(
            "cannot convert smiles '{}' to molecule".format(smiles)
        )
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


def mol_from_smiles(smiles: str, raise_error: bool = True) -> Mol:
    m = MolFromSmiles(smiles)
    if m is None and raise_error:
        raise MolGenerationError(
            "cannot convert smiles '{}' to molecule".format(smiles)
        )
    return m


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
    n_cores = cpu_count()
    if "all" in cores:
        if "all-" in cores:
            n_cores = n_cores - int(cores.replace("all-", ""))
        elif cores == "all":
            pass
        else:
            raise ValueError("Cannot get core number from '{cores}'")
        cores = n_cores

    cores = max(1, min(n_cores, int(cores)))

    MOLNET_LOGGER.debug(f"using {cores} cores")

    smiles = np.array(smiles)
    sub_smiles = np.array_split(smiles, min(1000, int(np.ceil(len(smiles) / cores))))
    with Pool(cores) as p:
        r = []
        if progess_bar:
            with tqdm(total=len(smiles), unit=" mol") as pbar:
                for ri in p.imap(_mfs, sub_smiles):
                    r.extend(ri)
                    pbar.update(len(ri))
        else:
            for ri in p.imap(_mfs, sub_smiles):
                r.extend(ri)
    return r
