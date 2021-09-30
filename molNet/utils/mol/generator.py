import random

from rdkit import Chem


def generate_random_carbon_lattice(n=20, cross_rate=0.1):
    m = Chem.Mol()
    i = 0

    def random_bond(m):
        if random.random() <= cross_rate:
            pre_free_atoms = [a for a in m.GetAtoms() if len(a.GetBonds()) < 4]
            if len(pre_free_atoms) >= 2:
                em = Chem.EditableMol(m)
                random.shuffle(pre_free_atoms)
                idx1 = pre_free_atoms[0].GetIdx()
                idx2 = pre_free_atoms[1].GetIdx()
                try:
                    bondIdx = em.AddBond(idx1, idx2, Chem.BondType.SINGLE)
                except RuntimeError:
                    pass
                m = em.GetMol()
        return m

    def end_round(em):
        m = em.GetMol()
        m = random_bond(m)
        return m

    while len(m.GetAtoms()) < n and i < n ** 2:
        i += 1
        em = Chem.EditableMol(m)
        pre_free_atoms = [a for a in m.GetAtoms() if len(a.GetBonds()) < 4]
        idx1 = em.AddAtom(Chem.Atom(6))

        if len(pre_free_atoms) == 0:
            m = end_round(em)
            continue

        random.shuffle(pre_free_atoms)
        idx2 = pre_free_atoms[0].GetIdx()
        bondIdx = em.AddBond(idx1, idx2, Chem.BondType.SINGLE)

        m = end_round(em)

    Chem.SanitizeMol(m)
    return m
