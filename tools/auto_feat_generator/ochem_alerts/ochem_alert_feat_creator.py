import os
import sys

modp=os.path.abspath(os.path.dirname(__file__))
while not "molNet" in os.listdir(modp) and not "setup.py" in os.listdir(modp):
    modp=os.path.dirname(modp) 
sys.path.append(modp)
sys.path.insert(0,modp)

import pickle
from re import sub

import black
import unicodedata
from rdkit import RDLogger
from rdkit.Chem import MolFromSmarts

from molNet.featurizer import molecule_featurizer

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

with open(os.path.join(os.path.dirname(__file__),"ochem_alerts.pickle"), "rb") as f:
    data = pickle.load(f)


classstring = """
molecule_functional_group_{classname}_featurizer = MoleculeHasSubstructureFeaturizer(smarts="{smarts}",feature_descriptions={description},name={name})

"""

code = """
from molNet.featurizer._molecule_featurizer import MoleculeHasSubstructureFeaturizer
"""

to_check = {}


def to_cls_name(oname):
    # name=name\
    #    .replace("(","")\
    #    .replace(")","")\
    #    .strip()
    #   .replace(",","_")
    backgroup = ""
    name = oname
    name = name.title().strip()
    name = sub("^[0-9][0-9]* -", "", name)
    # if name!=sub("[0-9][0-9]* -","",name):
    #    to_check[oname]="regex"
    #    name=sub("[0-9][0-9]* -","",name)
    name = name.strip()
    for pos_smile in name.split():
        if len(pos_smile) < 3:
            continue
    #  m=MolFromSmiles(pos_smile,sanitize=False)
    # if m is not None:
    # print(pos_smile,name)
    # if pos_smile.isalnum():
    #    continue
    #       name = name.replace(pos_smile,"")
    #      to_check[oname]="smiles"+pos_smile
    # else:
    # m=MolFromSmiles(sub("-*H[0-9]*","",pos_smile),sanitize=False)
    # if m is not None:
    # print(pos_smile,name)
    # if pos_smile.isalnum():
    #    continue
    # name = name.replace(pos_smile,"")
    # to_check[oname]="smiles"+pos_smile
    name = name.replace("(", "_")
    name = name.replace(")", "_")
    name = name.strip()
    name = name.strip("_")
    if name != sub("^[0-9][0-9]* -", "", name):
        to_check[oname] = "regex"
        name = sub("^[0-9][0-9]* -", "", name)

    name = name.replace("-", "_")
    name = name.replace("–", "_")
    name = name.replace("/", "_")
    name = name.replace("'", "_")
    name = name.replace("&", " ")
    name = name.replace("*", "")
    name = name.replace("+", "")
    name = name.replace(";", "_")
    name = name.replace("#", "")
    name = name.replace("=", "")

    name = name.replace(".", "_")
    name = name.replace(", ", " ")
    name = name.replace(",", "_")
    name = name.replace("⋯", "_")

    if ">" in name:
        name = name.replace(">", " gt ")
        # to_check[oname]=">"
    if "<" in name:
        name = name.replace("<", " st ")
        # to_check[oname]="<"
    if ":" in name:
        name = name.replace(":", "_")
        # to_check[oname]=":"

    back_info = ""
    #    name=name.strip("-")

    # while len(name)>0 and (name[0].isnumeric() or name[0]=="-" or name[0]==","):
    #    back_info+=name[0]
    #    name=name[1:].strip()

    if len(back_info) > 0:
        name = name + "_" + back_info

    name = name.title()
    name = sub("\s", "", name)
    name = sub("_+", "_", name)
    name = name.strip("_")

    if len(name) == 0:
        to_check[oname] = "None"
        return None
    return name


isin = {}
data = sorted(data, key=lambda d: d["name"])
for d in data:
    d["ignore"] = False
    if MolFromSmarts(d["smart"]) is None:
        d["ignore"] = True
        continue
    description = d.get("description", None)

    if description:
        description = '"{}"'.format(description.replace('"', "'"))
    else:
        description = "None"
    d["description"] = description

    name = d["name"]
    d["oname"] = name
    classname = to_cls_name(name)
    if classname is None:
        # if description=="None":
        d["ignore"] = True
        continue

    name = '"{}"'.format(d["name"].replace('"', "'"))
    d["name"] = name
    d["classname"] = classname
    i = 1
    while d["classname"] in isin:
        od = isin[d["classname"]]
        if d["smart"] != od["smart"]:
            d["classname"] = sub(f"__{i}", "", d["classname"])
            i += 1
            d["classname"] += f"__{i}"
            continue
        else:
            if d["description"] == "None":
                d["ignore"] = True
            elif od["description"] == "None":
                od["ignore"] = True
                del isin[d["classname"]]
            else:
                if len(od["description"]) >= len(d["description"]):
                    d["ignore"] = True
                else:
                    od["ignore"] = True
                    del isin[d["classname"]]

            break
    if d["ignore"]:
        continue

    isin[d["classname"]] = d

error = False
data = sorted([d for d in data if not d["ignore"]], key=lambda d: d["classname"])
for d in data:
    cls = classstring.format(
        smarts=d["smart"],
        description=d["description"],
        name=d["name"],
        classname=d["classname"],
    )
    # if d["classname"].startswith("2"):
    #    pprint(d)

    #    break
    if d["oname"] in to_check:
        print(to_check[d["oname"]], "|" + d["classname"] + "|", d["oname"], d["smart"])
    if not f"FunctionalGroup_{d['classname']}_Featurizer".isidentifier():
        print(f"FunctionalGroup_{d['classname']}_Featurizer", d["name"], d["smart"])
        error = True
    code += cls
if error:
    exit(0)
# print(code)
code += "_available_featurizer = {\n"
for d in data:
    vn=f"molecule_functional_group_{d['classname']}_featurizer"
    
    code += f"   '{vn}':{vn},\n"
code += "}"

code += """
def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1")
    for n,f in _available_featurizer.items():
        print(n, f(testmol))

if __name__ == "__main__":
    main()


"""
code = unicodedata.normalize("NFD", code)

code = black.format_str(code, mode=black.FileMode())


with open(
    os.path.join(
        os.path.dirname(molecule_featurizer.__file__),
        f"_autogen_{os.path.basename(os.path.dirname(os.path.abspath(__file__)))}_molecule_featurizer.py",
    ),
    "w+b",
) as f:
    f.write(code.encode("utf8"))
