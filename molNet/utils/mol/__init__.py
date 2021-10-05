from rdkit import Chem


def _gen_atom_n():
    d = {}
    em = Chem.RWMol()
    em.AddAtom(Chem.Atom(6))
    i = 0
    while True:
        try:
            em.GetAtoms()[0].SetAtomicNum(i)
            d[em.GetAtoms()[0].GetSymbol()] = i
        except RuntimeError:
            break
        i += 1
    return d


ATOMIC_SYMBOL_NUMBERS = _gen_atom_n()
