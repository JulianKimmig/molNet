import os

import black

from molNet.featurizer import molecule_featurizer, atom_featurizer


def gen_molecule_feats():
    mdir = os.path.dirname(molecule_featurizer.__file__)
    feat_base_class = "molecule_featurizer"

    code = "from molNet import MOLNET_LOGGER\n" \
           "from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization\n" \
           "_available_featurizer = {}\n__all__ = []\n"

    imp_code = """
try:
    from molNet.featurizer import {mod}
    from molNet.featurizer.{mod} import *

    for n, f in {mod}.get_available_featurizer().items():
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(f"encoutered duplicate while collecting moelcule featurizer: {{n}}")
            continue
        _available_featurizer[n] = f

    __all__ += {mod}.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)\n"""

    for f in os.listdir(mdir):
        if f == f"_autogen_{feat_base_class}.py":
            continue
        print(f)
        if f.startswith("_autogen") and f.endswith(f"{feat_base_class}.py"):
            mod = f.replace(".py", "")
            code += imp_code.format(mod=mod)
            print(f)
    code += "def get_available_featurizer():\n" \
            "    return _available_featurizer\n"
    code += "def main():\n" \
            "    from rdkit import Chem\n" \
            "    testmol = prepare_mol_for_featurization(Chem.MolFromSmiles('c1ccccc1'))\n"
    code += "    for n,f in get_available_featurizer().items():\n" \
            "        print(n,end=" ")\n" \
            "        print(f(testmol))\n" \
            "    print(len(get_available_featurizer()))"
    code += "if __name__ == '__main__':\n" \
            "    main()"
    code = black.format_str(code, mode=black.FileMode())

    print(mdir)
    with open(
            os.path.join(mdir, f"_autogen_{feat_base_class}.py"), "w+b"
    ) as f:
        f.write(code.encode("utf8"))


def gen_atom_feats():
    mdir = os.path.dirname(atom_featurizer.__file__)
    feat_base_class = "atom_featurizer"

    code = "from molNet import MOLNET_LOGGER\n" \
           "from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization\n" \
           "_available_featurizer = {}\n__all__ = []\n"

    imp_code = """
try:
    from molNet.featurizer import {mod}
    from molNet.featurizer.{mod} import *

    for n, f in {mod}.get_available_featurizer().items():
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(f"encoutered duplicate while collecting atom featurizer: {{n}}")
            continue
        _available_featurizer[n] = f

    __all__ += {mod}.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)\n"""

    for f in os.listdir(mdir):
        if f == f"_autogen_{feat_base_class}.py":
            continue
        print(f)
        if f.startswith("_autogen") and f.endswith(f"{feat_base_class}.py"):
            mod = f.replace(".py", "")
            code += imp_code.format(mod=mod)
            print(f)
    code += "def get_available_featurizer():\n" \
            "    return _available_featurizer\n"
    code += "def main():\n" \
            "    from rdkit import Chem\n" \
            "    testmol = prepare_mol_for_featurization(Chem.MolFromSmiles('c1ccccc1')).GetAtoms()[-1]\n"
    code += "    for n,f in get_available_featurizer().items():\n" \
            "        print(n,end=" ")\n" \
            "        print(f(testmol))\n" \
            "    print(len(get_available_featurizer()))"
    code += "if __name__ == '__main__':\n" \
            "    main()"
    code = black.format_str(code, mode=black.FileMode())

    print(mdir)
    with open(
            os.path.join(mdir, f"_autogen_{feat_base_class}.py"), "w+b"
    ) as f:
        f.write(code.encode("utf8"))


def main():
    gen_molecule_feats()
    gen_atom_feats()


if __name__ == '__main__':
    main()
