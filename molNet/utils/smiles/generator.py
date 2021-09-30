import random

import numpy as np
from rdkit import Chem
from rdkit.Chem import MolToSmiles, MolFromSmiles
from tqdm import tqdm

from molNet.utils.mol.generator import (
    generate_random_carbon_lattice as mol_generate_random_carbon_lattice,
)


def generate_random_carbon_lattice(n=20):
    return MolToSmiles(mol_generate_random_carbon_lattice(n=n))


def generate_n_random_carbon_lattice_generator(
    min_c: int, max_c: int = 60, rounds: int = 1000
):
    if max_c is None:
        max_c = 60
    already_yielded = set()
    for k in range(rounds):
        temp_subs = set()
        i_to_lattice = []

        for i in range(min_c, max_c + 1):
            for _ in range(i):
                i_to_lattice.append(i)
        random.shuffle(i_to_lattice)

        for i in i_to_lattice:
            nl = generate_random_carbon_lattice(i)
            if nl not in already_yielded:
                temp_subs.add(nl)
                already_yielded.add(nl)
                yield nl

        if len(temp_subs) == 0:
            continue

        for smiles in temp_subs.copy():
            _n_subs = set()

            s = MolFromSmiles(smiles)
            bonds = s.GetBonds()
            for bond in bonds:
                aidx1 = bond.GetBeginAtomIdx()
                aidx2 = bond.GetEndAtomIdx()
                em = Chem.EditableMol(s)
                em.RemoveBond(aidx1, aidx2)
                m = em.GetMol()
                _n_subs.update(
                    [n for n in MolToSmiles(m).split(".") if n.count("C") >= min_c]
                )

            _n_subs = _n_subs - already_yielded
            temp_subs.update(_n_subs)
            temp_subs = temp_subs - already_yielded
            for s in temp_subs:
                yield s
            already_yielded.update(temp_subs)


def generate_n_random_carbon_lattice(
    n, min_c=1, max_c=None, rounds=1000, progess_bar=True, raise_error=True
):
    if max_c is None:
        max_c = int(np.ceil(max(np.sqrt(n) / 20, np.log(n) * 2 + 2)))
    print(max_c)
    d = []
    if progess_bar:
        g = tqdm(
            generate_n_random_carbon_lattice_generator(
                min_c=min_c, max_c=max_c, rounds=rounds
            ),
            total=n,
            unit=" C_lattice",
        )
    else:
        g = generate_n_random_carbon_lattice_generator(
            min_c=min_c, max_c=max_c, rounds=rounds
        )
    for r in g:
        if len(d) >= n:
            break
        d.append(r)

    if raise_error and len(d) < n:
        raise ValueError("not enough structures generated, consider increasing 'max_c'")

    return d
