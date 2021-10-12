import os

import black

from molNet.featurizer import molecule_featurizer

dirs = {"molecule_featurizer": os.path.dirname(molecule_featurizer.__file__)}

for feat_base_class in ["molecule_featurizer"]:

    mdir = os.path.dirname(__file__)
    code = ""
    imp_code = "from molNet.featurizer.{mod}  import * \n"
    imp_code += "from molNet.featurizer.{mod}  import _available_featurizer as {mod}_available_featurizer\n"

    av_code = "    *{mod}_available_featurizer,\n"
    availabless_code = ""

    for f in os.listdir(mdir):
        if f == f"_autogen_{feat_base_class}.py":
            continue
        if f.startswith("_autogen") and f.endswith(f"{feat_base_class}.py"):
            mod = f.replace(".py", "")
            code += imp_code.format(mod=mod)
            availabless_code += av_code.format(mod=mod)
            print(f)

    code += f"_available_featurizer = [\n{availabless_code}]\n"

    code += """
def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1")
    """
    if feat_base_class == "molecule_featurizer":
        code += """
    for f in _available_featurizer:
        print(f, f(testmol))
    """
    code += """
if __name__ == "__main__":
    main()

"""
    code = black.format_str(code, mode=black.FileMode())
    with open(
        os.path.join(dirs[feat_base_class], f"_autogen_{feat_base_class}.py"), "w+b"
    ) as f:
        f.write(code.encode("utf8"))
