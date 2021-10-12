import logging
import pickle
from pprint import pprint
from re import sub

from rdkit.Chem import MolFromSmiles
from  rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

with open("ochem_alerts.pickle","rb") as f:
    data=pickle.load(f)

classstring="""
class FunctionalGroup_{classname}_Featurizer(MoleculeFunctionalGroupFeaturizer):
    smarts="{smarts}"
    description={description}
    name={name}
    
    def featurize(mol):
        return mol.HasSubstructMatch(self.pattern)
"""

code = ""

to_check={}
def to_cls_name(oname):
    #name=name\
    #    .replace("(","")\
    #    .replace(")","")\
    #    .strip()
     #   .replace(",","_")
    backgroup=""
    name=oname
    name=sub("^[0-9][0-9]* -","",name)
    if name!=sub("[0-9][0-9]* -","",name):
        to_check[oname]="regex"
        name=sub("[0-9][0-9]* -","",name)
    name = name.strip()
    for pos_smile in name.split():
        if len(pos_smile)<3:
            continue
        m=MolFromSmiles(pos_smile,sanitize=False)
        if m is not None:
            #print(pos_smile,name)
            #if pos_smile.isalnum():
            #    continue
            name = name.replace(pos_smile,"")
            to_check[oname]="smiles"+pos_smile
        else:
            m=MolFromSmiles(sub("-*H[0-9]*","",pos_smile),sanitize=False)
            if m is not None:
                #print(pos_smile,name)
                #if pos_smile.isalnum():
                #    continue
                name = name.replace(pos_smile,"")
                to_check[oname]="smiles"+pos_smile
    name = name.replace("(","_")
    name = name.replace(")","_")
    name = name.replace("-","_")
    name = name.replace("–","_")
    name = name.replace("/","_")
    name = name.replace("'","_")
    name = name.replace("&"," ")
    name = name.replace("*","")
    name = name.replace("+","")
    name = name.replace(";","_")
    name = name.replace("#","")
    name = name.replace("=","")

    name = name.replace(".","_")
    name = name.replace(", "," ")
    name = name.replace(",","_")
    if ">" in name:
        name = name.replace(">"," gt ")
        #to_check[oname]=">"
    if "<" in name:
        name = name.replace("<"," st ")
        #to_check[oname]="<"
    if ":" in name:
        name = name.replace(":","_")
        #to_check[oname]=":"


    back_info=""
#    name=name.strip("-")

    #while len(name)>0 and (name[0].isnumeric() or name[0]=="-" or name[0]==","):
    #    back_info+=name[0]
    #    name=name[1:].strip()

    if len(back_info)>0:
        name=name+"_"+back_info





    name = name.title()
    name = sub("\s","",name)
    name = sub("_+","_",name)
    name=name.strip("_")

    if len(name)==0:
        to_check[oname]="None"
        return None
    return name

for d in sorted(data,key=lambda d:d["name"]):
    d["ignore"]=False

    description=d.get("description",None)
    if description:
        description="'{}'".format(description)
    else:
        description="None"
    d["description"]=description

    name = d["name"]
    d["oname"]=name
    classname=to_cls_name(name)
    if classname is None:
       # if description=="None":
        d["ignore"]=True
        continue

    name="'{}'".format(d["name"])
    d["name"]=name
    d["classname"]=classname



for d in sorted([d for d  in data if not d["ignore"]],key=lambda d:d["classname"]):
    cls = classstring.format(smarts=d["smart"],description=d["description"],name=d["name"],classname=d["classname"])
    #if d["classname"].startswith("2"):
    #    pprint(d)

    #    break
    if d["oname"] in to_check:
        print(to_check[d["oname"]],d['classname'],d["oname"])
    if not f"FunctionalGroup_{d['classname']}_Featurizer".isidentifier():
        print(f"FunctionalGroup_{d['classname']}_Featurizer",d["name"],d["smart"])

    code+=cls
    code +="\n\n"

#print(code)