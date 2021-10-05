from random import shuffle
from typing import Dict, Callable, List, Tuple, Any

import numpy as np

# from numpy.random import rand
from random import random as rand
from rdkit import Chem
from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles
from rdkit.Chem.rdchem import (
    AtomSanitizeException,
    KekulizeException,
    MolSanitizeException,
)
from molNet.utils.mol import ATOMIC_SYMBOL_NUMBERS
from tqdm import tqdm


def generate_random_carbon_lattice(n=20, cross_rate=0.1) -> Mol:
    m = Chem.RWMol()
    i = 0
    indices = np.arange(n)
    atoms = np.array([m.AddAtom(Chem.Atom(6)) for _ in indices])
    free_val = np.array([4 for _ in indices])
    connected = np.array([False for _ in indices])
    bonds_free = ~np.diag(np.ones(n, dtype=bool))

    def bond(idx1, idx2):
        idx1, idx2 = int(idx1), int(idx2)
        m.AddBond(idx1, idx2, Chem.BondType.SINGLE)
        connected[idx1] = True
        connected[idx2] = True
        free_val[idx1] -= 1
        free_val[idx2] -= 1
        bonds_free[idx1, idx2] = False
        bonds_free[idx2, idx1] = False

    if n > 1:
        bond(0, 1)
    else:
        connected[0] = True

    def _r(a, p):
        x = rand()
        cs = p.cumsum()
        cs = cs >= x
        i = cs.argmax()
        return a[i]

    while not all(connected):
        idx1 = connected.argmin()

        c_fv = free_val * connected * bonds_free[idx1]
        idx2 = _r(indices, p=c_fv / c_fv.sum())
        bond(idx1, idx2)

        if rand() <= cross_rate:
            c_fv = free_val * connected
            # print()
            # print(bonds_free)
            # print(c_fv)
            idx1 = _r(indices, p=c_fv / c_fv.sum())
            c_fv *= bonds_free[idx1]
            if any(c_fv > 0):
                idx2 = _r(indices, p=c_fv / c_fv.sum())
                bond(idx1, idx2)

    m = m.GetMol()
    Chem.SanitizeMol(m)
    return m


def generate_random_carbon_lattice2(n=20, cross_rate=0.1) -> Mol:
    m = Chem.RWMol()
    i = 0

    def random_bond(m):
        if rand() <= cross_rate:
            _pre_free_atoms = [[a] * a.GetImplicitValence() for a in m.GetAtoms()]
            _pre_free_atoms = [pfe for pfe in _pre_free_atoms if len(pfe) > 0]
            if len(_pre_free_atoms) < 2:
                return m

            pre_free_atoms = []
            for al in _pre_free_atoms:
                pre_free_atoms.extend(al)
            if len(pre_free_atoms) >= 2:
                shuffle(pre_free_atoms)
                idx1 = pre_free_atoms[0].GetIdx()
                idx2 = pre_free_atoms[1].GetIdx()
                try:
                    bondIdx = m.AddBond(idx1, idx2, Chem.BondType.SINGLE)
                except RuntimeError:
                    pass
        return m

    def end_round(em):
        Chem.SanitizeMol(em)
        m = random_bond(em)
        return m

    while len(m.GetAtoms()) < n and i < n ** 2:
        Chem.SanitizeMol(m)
        i += 1
        _pre_free_atoms = [[a] * a.GetImplicitValence() for a in m.GetAtoms()]
        pre_free_atoms = []
        for al in _pre_free_atoms:
            pre_free_atoms.extend(al)

        idx1 = m.AddAtom(Chem.Atom(6))

        if len(pre_free_atoms) == 0:
            m = end_round(m)
            continue

        idx2 = pre_free_atoms[int(rand() * len(pre_free_atoms))].GetIdx()
        bondIdx = m.AddBond(idx1, idx2, Chem.BondType.SINGLE)

        m = end_round(m)

    m = m.GetMol()
    Chem.SanitizeMol(m)
    return m


def generate_random_unsaturated_carbon_lattice(
    n=20,
    cross_rate=0.1,
    double_bond_rate=0.2,
    triple_bond_rate=0.05,
) -> Mol:
    m = generate_random_carbon_lattice(n=n, cross_rate=cross_rate)
    em = Chem.RWMol(m)

    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in em.GetBonds()]
    for aidx1, aidx2 in bonds:
        Chem.SanitizeMol(em)
        a1 = em.GetAtomWithIdx(aidx1)
        a2 = em.GetAtomWithIdx(aidx2)

        v1 = a1.GetImplicitValence()
        v2 = a2.GetImplicitValence()
        if v1 > 1 and v2 > 1:
            if rand() <= triple_bond_rate:
                em.RemoveBond(aidx1, aidx2)
                em.AddBond(aidx1, aidx2, Chem.BondType.TRIPLE)
                continue
        if v1 > 0 and v2 > 0:
            if rand() <= double_bond_rate:
                em.RemoveBond(aidx1, aidx2)
                em.AddBond(aidx1, aidx2, Chem.BondType.DOUBLE)
                continue

    m = em.GetMol()
    Chem.SanitizeMol(m)
    return m


def generate_random_hetero_carbon_lattice(
    n: int = 20,
    cross_rate: float = 0.1,
    double_bond_rate: float = 0.2,
    triple_bond_rate: float = 0.05,
    hetero_rates: Dict[str, float] = None,
) -> Mol:
    if hetero_rates is None:
        hetero_rates = {"N": 0.2, "O": 0.2}
    hetero_rates = {ATOMIC_SYMBOL_NUMBERS[k]: v for k, v in hetero_rates.items()}

    m = generate_random_unsaturated_carbon_lattice(
        n=n,
        cross_rate=cross_rate,
        double_bond_rate=double_bond_rate,
        triple_bond_rate=triple_bond_rate,
    )
    em = Chem.RWMol(m)
    Chem.SanitizeMol(em)
    for a in em.GetAtoms():
        for n, r in hetero_rates.items():
            try:
                if rand() <= r:
                    a.SetAtomicNum(n)
                    Chem.SanitizeMol(em)
                    break
            except (AtomSanitizeException, MolSanitizeException):
                a.SetAtomicNum(6)
                Chem.SanitizeMol(em)

    m = em.GetMol()
    Chem.SanitizeMol(m)
    return m


def fragment_random_molecule_generator(
    random_mol_generator,
    rounds: int = 1000,
    fragment_checker_mol: Callable = None,
    fragment_checker_smiles: Callable = None,
    assert_conformer=False,
    list_of_args_kwargs: List[Tuple[List, Dict[str, Any]]] = None,
):
    if list_of_args_kwargs is None:
        list_of_args_kwargs = [([], {})]

    already_yielded = set()
    for k in range(rounds):
        temp_subs = set()
        for args, kwargs in list_of_args_kwargs:
            nl = random_mol_generator(*args, **kwargs)
            if nl:
                nls = MolToSmiles(nl)
                if nls not in already_yielded:
                    temp_subs.add(nls)
                    already_yielded.add(nls)
                    yield nl
        if len(temp_subs) == 0:
            continue

        for smiles in temp_subs.copy():
            _n_subs = set()

            s = MolFromSmiles(smiles)
            bonds = s.GetBonds()
            for i in range(len(bonds)):
                mfs = Chem.FragmentOnBonds(s, (i,), addDummies=False)
                n = MolToSmiles(mfs).split(".")
                _n_subs.update(n)

            _n_subs = _n_subs - already_yielded
            temp_subs.update(_n_subs)
            temp_subs = temp_subs - already_yielded
            for s in temp_subs:  #
                m = MolFromSmiles(s)
                if m:
                    yield m
            already_yielded.update(temp_subs)


def shuffle_buffer_mol_gen(buffer_size=100, *args, **kwargs):
    buffer = []
    for k in fragment_random_molecule_generator(**kwargs):
        buffer.append(k)
        if len(buffer) > buffer_size:
            yield buffer.pop(int(rand() * buffer_size))
    shuffle(buffer)
    while buffer:
        yield buffer.pop()


def random_carbon_lattice_generator(
    min_c=1, max_c=60, rounds=1000, buffer=100, **kwargs
):
    i_to_lattice = []

    for i in range(min_c, max_c + 1):
        for _ in range(i):
            i_to_lattice.append(i)

    shuffle(i_to_lattice)
    if buffer > 1:
        g = shuffle_buffer_mol_gen(
            random_mol_generator=generate_random_carbon_lattice,
            rounds=rounds,
            list_of_args_kwargs=[([], dict(n=n, **kwargs)) for n in i_to_lattice],
        )
    else:
        g = fragment_random_molecule_generator(
            random_mol_generator=generate_random_carbon_lattice,
            rounds=rounds,
            list_of_args_kwargs=[([], dict(n=n, **kwargs)) for n in i_to_lattice],
        )

    for m in g:
        if m is not None:
            yield m


def random_unsaturated_carbon_lattice_generator(
    min_c=1, max_c=60, rounds=1000, buffer=100, **kwargs
):
    i_to_lattice = []

    for i in range(min_c, max_c + 1):
        for _ in range(i):
            i_to_lattice.append(i)
    shuffle(i_to_lattice)
    if buffer > 1:
        g = shuffle_buffer_mol_gen(
            random_mol_generator=generate_random_unsaturated_carbon_lattice,
            rounds=rounds,
            list_of_args_kwargs=[([], dict(n=n, **kwargs)) for n in i_to_lattice],
        )
    else:
        g = fragment_random_molecule_generator(
            random_mol_generator=generate_random_unsaturated_carbon_lattice,
            rounds=rounds,
            list_of_args_kwargs=[([], dict(n=n, **kwargs)) for n in i_to_lattice],
        )

    for m in g:
        if m is not None:
            yield m


def random_hetero_carbon_lattice_generator(
    min_c=1, max_c=60, rounds=1000, buffer=100, **kwargs
):
    i_to_lattice = []

    for i in range(min_c, max_c + 1):
        for _ in range(i):
            i_to_lattice.append(i)
    shuffle(i_to_lattice)
    if buffer > 1:
        g = shuffle_buffer_mol_gen(
            random_mol_generator=generate_random_hetero_carbon_lattice,
            rounds=rounds,
            list_of_args_kwargs=[([], dict(n=n, **kwargs)) for n in i_to_lattice],
        )
    else:
        g = fragment_random_molecule_generator(
            random_mol_generator=generate_random_hetero_carbon_lattice,
            rounds=rounds,
            list_of_args_kwargs=[([], dict(n=n, **kwargs)) for n in i_to_lattice],
        )

    for m in g:
        if m is not None:
            yield m


def generate_n_random_carbon_lattice(n, progess_bar=True, **kwargs):
    kwargs["buffer"] = n
    d = []
    if "max_c" not in kwargs:
        kwargs["max_c"] = int(np.ceil(max(np.sqrt(n) / 20, np.log(n) * 2 + 2)))

    g = random_carbon_lattice_generator(**kwargs)
    if progess_bar:
        g = tqdm(
            g,
            total=n,
            unit=" mol",
        )
    for k in g:
        if len(d) >= n:
            break
        d.append(k)

    return d


def generate_n_random_unsaturated_carbon_lattice(n, progess_bar=True, **kwargs):
    kwargs["buffer"] = n
    d = []
    g = random_unsaturated_carbon_lattice_generator(**kwargs)
    if progess_bar:
        g = tqdm(
            g,
            total=n,
            unit=" mol",
        )
    for k in g:
        if len(d) >= n:
            break
        d.append(k)

    return d


def generate_n_random_hetero_carbon_lattice(n, progess_bar=True, **kwargs):
    kwargs["buffer"] = n
    d = []
    g = random_hetero_carbon_lattice_generator(**kwargs)
    if progess_bar:
        g = tqdm(
            g,
            total=n,
            unit=" mol",
        )
    for k in g:
        if len(d) >= n:
            break
        d.append(k)

    return d
