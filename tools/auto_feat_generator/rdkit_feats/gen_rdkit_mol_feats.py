import os
import pickle
from warnings import warn

import black
import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    rdMolDescriptors,
    Descriptors3D,
    GraphDescriptors,
    Descriptors,
    rdmolops,
    rdForceFieldHelpers,
)

from molNet import ConformerError
from molNet.featurizer import molecule_featurizer
from molNet.utils.mol.properties import assert_conformers
from molNet.utils.smiles.generator import generate_n_random_hetero_carbon_lattice

modules = [Chem, rdmolops, rdMolDescriptors, Descriptors3D, GraphDescriptors, Descriptors]
modules = {mod.__name__: mod for mod in modules}

data_to_work_pckl = "rdkit_feat_ex_data_to_work.pckl"
unsucsess_mods_pckl = "rdkit_feat_ex_unsucsess_mods.pckl"
sucsess_mods_pckl = "rdkit_feat_ex_sucsess_mods.pckl"

MANUAL_PRECLASSCODES = {
    "CalcGETAWAY": """
from rdkit.Chem.rdMolDescriptors import CalcGETAWAY as _oCalcGETAWAY
from rdkit.Chem import GetMolFrags
def CalcGETAWAY(mol):
    frags=GetMolFrags(mol,asMols=True)
    if len(frags)>1:
        frags=sorted(frags,key=lambda m: mol.GetNumAtoms(),reverse=True)
        return  _oCalcGETAWAY(frags[0])
    return  _oCalcGETAWAY(mol)"""
}

BAD_LIST = [
    "^SplitMolByPDBResidues$",
    "^SplitMolByPDBChainId$",
    "SanitizeMol",
    "AUTOCORR2D_[0-9]",
    "^_",
    "_$",  # internal use,
]
BLACK_LIST_MODULES = [rdForceFieldHelpers]
SELF_IGNORES = [
    "Compute2DCoords",
    "GetConformer",
    "GetConformers",
    "GetNumConformers",
    "GetPropNames",
    "GetPropsAsDict",
    "NeedsUpdatePropertyCache",
    "ToBinary",
    "__class__",
    "__copy__",
    "__dir__",
    "__getinitargs__",
    "__hash__",
    "__reduce__",
    "__repr__",
    "__sizeof__",
    "__str__",
    "__subclasshook__",
    "_repr_png_",
]

testmols = []
for smile in [
                 "C1=CC=C(C=C1)C2=CC(=CC(=C2)C3=CC(=CC4=C3SC5=CC=CC=C54)N(C6=CC=CC=C6)C7=CC=CC=C7)C8=CC(=CC9=C8SC1=CC=CC=C19)N(C1=CC=CC=C1)C1=CC=CC=C1",
                 "C" * 20,
                 "CC(C(=O)NCCN1C(=O)C=CC(=N1)N2C=CC=N2)OC3=CC(=CC=C3)Cl",
                 "CC1=C(C(NC(=S)N1)C2=CC=C(C=C2)OCC3=CC=CC=C3)C(=O)C4=CC=CC=C4",
                 "CC1=CC(=CC=C1)OC(=O)C23CC4CC(C2)CC(C4)(C3)Cl",
                 "COC(=O)CN1C2C(NC(=O)N2)NC1=O",
                 "CC1=C(SC2=CC=CC=C12)C(=O)C[NH+]3CCN(CC3)C4=[NH+]C=C(C=C4)C(F)(F)F",
                 # 'CCCC=CNC1=C(C=C(C=N1)C2=CC(=CC=C2)C(=O)C(C)C3CCCCN(CC3)C(=O)OC(C)(C)C)NC4=[N+](C(=C(C=C4)OC)OC)C',
                 # 'CN(C1CCNC1)C(=O)/C=C/C2=CC=CC=C2Br',
                 # 'CN1CN(C2=CC=CC(=C21)C(=O)NC[C@H]([C@@H]3CC4=CC=CC=C4CN3)O)C5CCOCC5',
                 # 'CNC(=O)C1(C=C(C(=C(N1)O)O)C(=O)NCC2=CC=C(C=C2)F)CC3=CC=CC=C3',
                 # 'CC(C)(CCN)CNC1=CC2=C(C=C1)NC(=O)O2',
                 # 'C[C@H](C1=NN=C(N1C2=CC=CC=C2)SCC(=O)NC3=NC(=CS3)C4=CC=C(C=C4)F)[NH+](C)C',
                 # 'C1=CC(=C(C=C1[N+](=O)[O-])[N+](=O)[O-])NC(=O)C2=C(C=CN=C2)[N+](=O)[O-]',
                 # #'C[C@@H](CC[C@H]1[C@@H](OC(O1)(C)C)CCC=C)C[C@H]([C@@H](C)C(=O)O[C@@H](C[C@H](C)CC=C)CO[Si](C2=CC=CC=C2)(C3=CC=CC=C3)C(C)(C)C)OCOC',
                 # 'CC(C)(COCC=C)N=C=O',
                 # 'CC(C)(CNC1=CC(=CC=C1)OC2=CC=CC=C2)N.Cl',
                 # 'C[C@@]1([C@H]([C@@H](O[C@]1(C#N)N2C=CC(=NC2=O)N)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O',
                 # 'CCC(C)CCC(=O)C1CCCC2(C1)CCC(CC2)C(C)C',
                 # 'C(CCCN)CCC(=O)NCC(C(F)(F)F)O',
                 # 'CC(C)(C)C1=CC=C(C=C1)CN2C=NC(=N2)CNC',
                 # 'CC1=CC=C(C=C1)CSC2=NN=C(N2CC=C)C3=CC=C(C=C3)Br',
                 # 'CCOC1=C2C=CC=C(C2=CC3=CC=CC=C31)C(C)O',
                 # 'CC(=O)NC1=NC(=C(S1)C2=CC=CC=C2)CCC3=CC=C(C=C3)N=C(N)N',
                 # 'CC(C)(CCC(=O)NCC1=C(N=CC=C1)OC)N',
                 # 'CCOC1=CC=CC(=C1OCC2=CC=CC=N2)CO',
                 # 'C1=CSC(=C1)C(=O)NC2=C(C=C(C=C2)I)C(=O)NCCCCO',
                 # 'CN(C)CC=CC(=O)N1CC[C@H](C1)NC2=NC=NC3=C2C=C(N3)C#CCOC4=CC=CC=C4Cl',
                 # 'CC(=CCC1=CC=C(C=C1)OC)C2=CC3=CC=CC=C3C=C2',
                 # 'CC[C@@H](C)[I-]C1CCC(CC1)C',
                 # 'CC1(CC(C(N1[O])(C)C)NC(=O)C=C)C',
                 # 'CN(C)C1CN(CCC1O)SC',
                 # 'CCCC[C@]1(CS(=O)(=O)C2=C(C=C(C=C2)N(C)C)[C@@H]([C@@H]1O)C3=CC(=CC=C3)NC(=O)CCCCN(C(=O)C)C(=O)[C@@H]([C@H]([C@@H]([C@@H](COC(=O)C)OC(=O)C)OC(=O)C)OC(=O)C)OC(=O)C)CC',
                 # 'CN(C)CCCN(CC1=CC(=C(C=C1)Cl)Cl)S(=O)(=O)C2=CNC(=C2)C(=O)N3CCCC3',
                 # 'CCC1=CC=C(C=C1)CCC(=O)NC2=CC=NN2C(C)C',
                 # 'C1[C@@H](CC(=O)O[C@H]1C2=CC=CC=C2)C3=CC=CC=C3',
                 # 'CCC1=CC=C(C=C1)C(C)N[C@H](C)C2=CC=CC=C2Br',
                 # 'C1C=CC=CC1(C2=CC=CC=C2)C3=NN(N=N3)N=[N+]=[N-]',
                 # 'CCNCC1CCCN(C1)C2=NC3=CC=CC=C3N=C2C',
                 # 'C1=CC=C2C(=C1)C3=C(N2CC4=CC(=C(C(=C4)F)F)F)C=CC(=C3)N(C5=CC=C(C=C5)C#N)C6=CC7=C(C=C6)N(C8C7C=CC=C8)CC9=CC(=C(C(=C9)F)F)F',
                 # 'CC1=NC(=C(C=C1)N2C=C(C=N2)CCO)N',
                 # 'CCCNCC(C)(C)CCC1=CC=C(S1)Br',
                 # 'CNC(=O)C1[C@H]2[C@H](C(C(C1C(=O)NCC3=CC=CC=C3)O2)OC(=O)CCl)O',
                 # 'C1C(N(C(=O)C1I)C2=CC=C(C=C2)Cl)C3=CC(=CC=C3)OC4=NC=CN=C4',
                 # 'CCC1=CC2=C(C=C1)N=C(S2)N(CC3CCCO3)C(=O)C4=CC(=CC=C4)S(=O)(=O)CC',
                 # 'C1CCC(C1)(CCCOCC(F)(F)F)CO',
                 # 'CC1=CC(=C(C=C1)F)NC(=O)NC2=CC=C(C=CC2)C3=C4C(=CC=C3)NN=C4N',
                 # 'C1=CSC(=C1)C2=NOC(=N2)CNC(=S)NC3=CC=C(C=C3)CC#N',
                 # 'CC.CC.CC.CCC1CN(CCN1CC2=CC=CC=C2)C3=CC=CC(=C3C(=O)N(C)CC4=CC(=C(C=C4)OC)OC)C=O',
                 # 'CCCCN1CCN(C(=O)C1=O)CC2=NN=C(O2)C(C)(CC3=CC=CC=C3)N',
                 # 'C[C@@H](C1=CC=CC=C1)C(=O)N[C@H]2COC3=CC=CC=C3[C@@H]2OC',
                 # 'CCCC1=NC=CN1CC2=CC=C(C=C2)C(=NO)N',
                 # 'CN1C=C(N=N1)CC(C2=COC=C2)NN',
                 # 'CC(=O)OCC(=O)[C@]1(CC[C@@H]2[C@@]1(C[C@@H]([C@H]3[C@H]2CC(C4=CC(=O)CC[C@]34C)CN(C)C5=CC=CC=C5)OC(=O)C(F)(F)F)C)O',
                 # 'CCCNC(CCC1=CN=CC=C1)C2=CN(N=C2)CCC',
                 # 'C1=CC2=C3C(=C1NCCNCCO)C(=O)C4=C(C3=NN2CCCN)C=CN=C4',
                 # 'C1=CC(=C(C(=C1)Cl)NC2=NC=C(N=C2)C(=O)O)Cl',
                 # 'C=CCC1=CC=CC=C1OCC2=CC=C(O2)C(=O)NC3=C(C=C(C=C3)[N+](=O)[O-])C#N',
                 # 'CC1=C(C(=CC=C1)N(C)N)OCC2=CC=CC=N2',
                 # 'CCN1C=C(C(=N1)C(=O)NCC(C)C)NC(=O)C2=NN(C=C2)COC3=C(C(=CC=C3)Cl)Cl',
                 # 'B(C)(C1=CC=CC=C1C2=NC(=CC=C2)N)F.CC',
                 # 'CC(C1=C(C2=CC(=C(C=C2N=C1)OC)OC)C3=CC(=C(C=C3)OC)OC)O',
                 # 'CC(=C)C(=O)OCCP12(C3=CC=CC4=C3C(=CC=C4)O1)C5=CC=CC6=C5C(=CC=C6)O2',
                 # 'CN(C)CC1=C(C2CCC1C2)C3=CC4=C(C=CC=C4S3)OC',
                 # 'CC(C)C(CC1=CSC(=N1)CCOCC(F)F)N',
                 # 'C1CN(CCN1)CC2=CC3=C(O2)C=C(C=C3)CN',
                 # 'CN1C=CC=C1CN(CC2=CC=CO2)C(=O)CN(CCOC)C(=O)C=CC3=CC=CC=C3',
                 # 'C=C/C=C\\C=C\\C1=CC(=CC(=C1)N2C3=CC=CC=C3C4=C2C=C(C=C4)C5=NN=C(O5)C6=CC7=C(C=C6)N(C8=CC=CC=C87)C9=CC(=CC(=C9)C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1',
                 # 'CC(=O)C1=CC(=C(C=C1)OCCCC(=O)NC2=CC=C(C=C2)NC(=O)COC)OC',
                 # 'C1=CC=C(C=C1)N[C@@H](C2=CC=CC3=CC=CC=C32)[C@@H](C#N)O',
                 # 'CC1=CC(=C(S1)NC(=O)CSC2=NN=C(N2C)C3CCCC3)C(=O)OC',
                 # 'CC1=CC(=C(C=C1)O)/C=N/NC(=O)[C@H](C2=CC=CC=C2)O',
                 # 'C[C@H](C1=NC=C(C=C1)Cl)NC(=O)C2=CN(N=C2COC)C',
                 # 'C[C@H](C(=O)NC1=CC=C(C=C1)C2=CN3[C@@H](C2)C=NC4=CC(=C(C=C4C3=O)OC)OCCCCCOC5=C(C=C6C(=C5)N=C[C@@H]7CC(=CN7C6=O)C8=CC=C(C=C8)OC)OC)NC(=O)[C@@H](C(C)C)NC(=O)CCCCCN9C(=O)C=C(C9=O)OC1=CC=C(C=C1)C#N',
                 # 'C1=CC(=CC(=C1)C(F)(F)F)C(C2=CC(=C(C=C2)F)Cl)O',
                 # 'CC1=CC(=C(C=C1)C)OC2=NN=C(S2)CNCCOC',
                 # 'CC(C)C1=CC(=CC=C1)OC(CN)C2=CC=C(C=C2)OC',
                 # 'CCC1=C(N=C2C=C(C=CC2=C1)F)C/C=C\\C=C(/C=C)\\F',
                 # 'CCCCOC(=O)CNC(CC)(CC)CO',
                 # 'CC1=NC2=C(N1C3CCCC3)C=C(C=C2)C4=NC(=NCC4F)NC5=NC=C(C=C5)C6CCNCC6',
                 # 'CCOC(=O)COCCN1CCSC(C1C)C',
                 # 'C1=CC=C(C(=C1)C2=CC=C(O2)/C=C\\3/C(=NC4=CC=C(C=C4)Cl)NC(=O)S3)[N+](=O)[O-]',
                 # 'CCS(=O)(=O)N[C@@H](CC1=CC=CC=C1)C(=O)N2CCC[C@H]2C(=O)C(=O)[C@](C)(CCCN=C(N)N)N',
                 # 'C1CC(=O)NCC1NC(=O)[C@H](CCC2=CC=CC=C2)N',
                 # 'CC1=CC(=CC=C1)S(=O)(=O)NCC(=O)NC2=C(C=C(C(=C2)C)Cl)OC',
                 # 'CC(C(=O)O)N(C)C/C=C/C1=CC(=C(C=C1)F)F',
                 # 'CCCC(C(=O)O)NC(=O)CCN1C(=O)C(=CC2=CC=C(C=C2)OC)SC1=S',
                 # 'CCN(C)C(C1=CC=CC=C1)C(=O)NCCC(=O)O',
                 # 'CCC(CC)N1C=CC(=N1)CN2CCC(CC2)OCCN',
                 # 'CCNC(=S)NCCCNC1=C(C=C2C=C(C=CC2=[NH+]1)OCC)C#N',
                 # 'CC1CCC2=NC3=CC=CC=C3C(=C2C1)C(=O)NCC4=CN=C(C=C4)N5CCOCC5',
                 # 'CC1=CC=C(C=C1)N2C(=O)C3=C(C=CC(=C3)Br)N=C2NN',
                 # 'CC1=CC2=NC(=NN2C=C1NC)NC(=O)CCC3=CC=CC=C3',
                 # 'CNC1CCN(CC1)C(=O)C2=CC(=C(C(=C2)F)F)F.Cl',
                 # 'CC1CN(C2=C(N1C(=O)CC3=C(C=C(C=C3)F)F)C=CC(=C2)C4=CN(N=C4)C5CCNCC5)C(=O)O',
                 # 'COC=NC(C#N)C(=O)N',
                 # 'CC[C@H](C(=O)O)SC1=NN=NN1C2=CC=CC=C2',
                 # 'CN=C(NCCC1=CC=CC2=CC=CC=C21)NC3CCN(C3)C(=O)C4CCCC4.I',
                 # 'CC(CC1CCCCCN1C2CCOCC2)O',
                 # 'CCCN1C2=C(C=C(C=N2)C(=O)NC3=CC=CC=C3N4CCC5=CC=CC=C54)C(=O)NC1=O',
                 # 'CC1=CC(=CC(=C1)OCC(=O)N(CC2=CC=C(C=C2)F)[C@@H](C)C(=O)NC(C)(C)C)C',
                 # 'C1CN=C(N1)NN=C(C=CC2=CC=CC=C2)C=CC3=CC=C(C=C3)Cl',
                 # 'CN(CCCN)C1=CC(=C(C=C1)C(=O)N)Br',
                 # 'CN(CC1CCNCC1)C2CCCC(C2)C(F)(F)F',
                 # 'CCCCN1C(C(CCC1=O)C(=O)NCC(C)O)C2=CC=CC=C2OC',
                 # 'CC1=C(C(=NO1)C)CCNC(=NC)NCC2(CCCO2)C',
                 # 'CS(=O)(=O)N1CC2(C1)CN(C2)C3=CC=C(C=C3)NC4=NC=C5C(=N4)N6C=CN=C6N(C5=O)C7=C(C=CC=C7Cl)Cl',
                 # 'CC1CCC(N(C1)CC(=O)C2=C(C=CS2)Br)C',
                 # 'C1=CC=C(C=C1)S(=O)(=O)N(CC(=O)NN=CC2=C(C=CC3=CC=CC=C32)O)C4=CC(=CC=C4)Cl',
                 # 'CC(CNC(=NC)N1CCC(CC1)OCCCOC)COCC2=CC=CC=C2.I',
                 # 'CC(C)(C)C1=CC(=CC(=C1)C2=CSC(=N2)CNCC(=O)O)C(C)(C)C',
                 # 'C1=CC=C(C=C1)C2=CC(=CC(=N2)C3=CC=CC=C3)C4=CC=CC(=C4)C5=CC(=CC=C5)C6=CC=CC7=C6SC8=C7C9=CC=CC=C9C=C8',
                 # 'CCC(CC#C)NC(C/C(=N/O)/N)C1=CC=CC=C1',
                 # 'CC(=O)SC\\1CCNC/C1=C\\C2=NN(C=C2)CC(=O)OC(C)(C)C.C(=O)(C(F)(F)F)O',
                 # 'C1=C(C(=CC(=C1N(CCN=[N+]=[N-])CCCl)[N+](=O)[O-])[N+](=O)[O-])C(=O)N',
                 # 'CCOC1=CC=CC=C1NC(=O)N(CC(=O)N2C(C3=CC=CN3C4=CC=CC=C42)C5=CC=C(C=C5)C)C(C)C',
                 # 'CCN1CCC(CC1)CNS(=O)(=O)C2=CC=CC=C2CO',
                 # 'CCCCCCCCC(=O)N(CCCCCl)CCCO',
                 # 'CCNC(=NCC1CCN(C1)CC)NC2CCC(CC2)C(C)C',
                 # 'CCC1=NNC(=S)N1NCC2=CC(=C(C=C2Br)OCC3=CC=CC=C3F)OCC',
                 # 'C1C=C2C(CC3C(C2C4=COC5=C(C4)C=C(C=C5)O)C(=O)N(C3=O)C6=CC=C(C=C6)[N+](=O)[O-])C7C1C(=O)N(C7=O)C8=CC=C(C=C8)[N+](=O)[O-]',
                 # 'CC1=CC(=CC=C1)CCNC(=O)C2=CC(=CC(=C2)Br)N3CCN(CC3)C4=CC=NC=C4.C(=O)(C(F)(F)F)O',
                 # 'CC1=CC(=O)C(=C2N1C3=C(C=C(C=C3)Cl)SC(C2)C4=CSC=C4)C(=O)NCC5=CN=CC=C5',
                 # 'CC12C(CC3C4C(CC=C3C1C5=CC(=C(C(=C5)Br)O)OC)C(=O)N(C4=O)N(C)C6=C(C=CC(=N6)C(F)(F)F)Cl)C(=O)N(C2=O)C7=CC=CC=C7',
                 # 'CC1=CC=C(C=C1)CC(=O)NCC2=CC=C(C=C2)N3C=CN=C3C',
                 # 'CCS(=O)CCNC1CCCCC1C2CCCCC2',
                 # 'CC1=C(C(=NO1)C)CS(=O)(=O)CC2=NN(C=C2)C3CCCC3',
                 # 'CC1=CC(=C(C=C1)O)C(=O)NCCN',
                 # 'CC(=O)N(C)C1=CC=C(C=C1)NS(=O)(=O)C2=CC=CC=C2Cl',
                 # 'CCC(=O)C(C1CCCCC1)C(C)(C)O[Si](C)(C)C',
                 # 'CC1=C(C=CC(=C1)C(C(=O)O)NC(=O)C2=CSC(=C2C)C)F',
                 # 'CN(CC1=CC=CC=C1)C(=O)C2=CC(=CC=C2)NCC(=O)N(C)C3CCCCC3',
                 # 'CC1=CN=C(O1)CNC2=C(C=CC(=C2)S(=O)(=O)N)N',
                 # 'CC(C)(C(=O)O)ON=C(C1=CSC(=N1)N)C(=O)NC2[C@H]3N(C2=O)C(=C(CS3)N4C=[N+](C(CC4=N)N)C5CC5)C(=O)O',
                 # 'CCN(CC)CCCN1CC(=O)N[C@@H]2[C@H]1CCCC2',
                 # 'COC1=C(C=C(C=C1)F)S(=O)(=O)NCCN2C(=O)C=CC=N2',
             ] + generate_n_random_hetero_carbon_lattice(n=60, max_c=15):
    try:
        testmols.append(assert_conformers(Chem.MolFromSmiles(smile)))
    except ConformerError:
        pass

import re

BAD_LIST = [re.compile(s) for s in BAD_LIST]


def reduce_name(n):
    red_name = n
    if red_name.startswith("Calc") and red_name[4:].strip("_")[0].isalpha():
        red_name = red_name[4:]

    if red_name.startswith("Get") and red_name[3:].strip("_")[0].isalpha():
        red_name = red_name[3:]

    return red_name


if not os.path.exists(data_to_work_pckl):
    data_to_work = [None]
    for mod_name, mod in modules.items():
        for n in vars(mod).keys():
            data_to_work.append((mod_name, n))
    with open(data_to_work_pckl, "w+b") as f:
        pickle.dump(data_to_work, f)

with open(data_to_work_pckl, "rb") as f:
    data_to_work = pickle.load(f)

if os.path.exists(unsucsess_mods_pckl):
    with open(unsucsess_mods_pckl, "rb") as f:
        unsucsess_mods = pickle.load(f)
else:
    unsucsess_mods = set()

if os.path.exists(sucsess_mods_pckl):
    with open(sucsess_mods_pckl, "rb") as f:
        sucsess_mods = pickle.load(f)
else:
    sucsess_mods = set()

unsucsess_mods.add(data_to_work[0])
print("bad", data_to_work[0])
data_to_work = data_to_work[1:]
with open(unsucsess_mods_pckl, "w+b") as f:
    pickle.dump(unsucsess_mods, f)


def error_res(n, f, e):
    # pass
    print(n, e)


for i, d in enumerate(data_to_work):
    with open(data_to_work_pckl, "w+b") as f:
        pickle.dump(data_to_work[i:], f)
    mod = modules[d[0]]
    n = d[1]
    f = vars(mod)[n]
    if any([r.search(n) is not None for r in BAD_LIST]):
        continue
    if n.startswith("_") or isinstance(f, str):
        continue
    try:
        f(Chem.Mol(testmols[0]))
        # succs[n]=f
        sucsess_mods.add(d)
        with open(sucsess_mods_pckl, "w+b") as f:
            pickle.dump(sucsess_mods, f)
    except Exception as e:
        unsucsess_mods.add(d)
        with open(unsucsess_mods_pckl, "w+b") as f:
            pickle.dump(unsucsess_mods, f)
        error_res(n, f, e)

try:
    os.remove(
        data_to_work_pckl,
    )
except FileNotFoundError:
    pass
try:
    os.remove(sucsess_mods_pckl)
except FileNotFoundError:
    pass
try:
    os.remove(unsucsess_mods_pckl)
except FileNotFoundError:
    pass
len(sucsess_mods)

succs = []
for mod_name, func_name in sucsess_mods:
    mod = modules[mod_name]
    f = getattr(mod, func_name)
    if any(
            [f == getattr(black_mod, func_name, None) for black_mod in BLACK_LIST_MODULES]
    ):
        continue
    s = {
        "func_name": func_name,
        "func": f,
        "module": mod_name,
    }
    s["red_name"] = reduce_name(s["func_name"])
    s["classname"] = f"{s['red_name']}_Featurizer"

    already_in = False
    for ss in succs:
        if s["func_name"] == ss["func_name"] and (s["module"] == ss["module"] or s["func"] == ss["func"]):
            already_in = True
            break
    if already_in:
        continue
    succs.append(s)

def check_memberf(func_name):
    try:

        def f(_mol):
            return getattr(_mol, func_name)()

        res = f(Chem.Mol(testmols[0]))
        if res is None:
            return
        succs.append(
            {
                "func_name": func_name,
                "func": f,
                "module": "self",
            }
        )
        # print(func_name)
        # print(res)
    # print("-"*20)
    except Exception:
        pass


for func_name in dir(testmols[0]):
    if func_name in SELF_IGNORES:
        continue
    check_memberf(func_name)
    # print(func_name)
len(succs)

MAX_LENGTH = 4096

loded_funcs = []
for s in succs[::-1]:
    n = s["func_name"]
    f = s["func"]
    # if f in loded_funcs:
    #    continue
    loded_funcs.append(f)


    # print(n)
    def try_f(mol):
        try:
            # print("Call",n,end=" ")
            return f(Chem.Mol(mol))

        except Exception as e:
            print(Chem.MolToSmiles(mol))
            # raise e


    # print("done")
    # f=try_f
    r = try_f(Chem.Mol(testmols[0]))
    # print(n,f,r)
    # break
    data = [r]
    s["sample_data"] = data
    if r is None:
        s["type"] = "none"
        continue
    elif isinstance(r, (Chem.Mol, Chem.EditableMol)):
        s["type"] = "mol"
        continue
    elif isinstance(r, (Exception)):
        s["type"] = "exception"
        continue
    elif isinstance(r, (int, float)):
        s["type"] = "numeric"
        s["length_type"] = "independend_length"
        s["length"] = 1
    elif isinstance(r, (str)):
        s["type"] = "string"
        continue
    elif isinstance(r, (list, tuple)):

        l1 = len(r)
        if l1 == 0:
            s["type"] = "unknown"
            continue

        if isinstance(r[0], (int, float)):
            pass
        else:
            s["type"] = "unknown"
            continue
        s["type"] = "list"
        l1 = len(r)
        s["length_type"] = None
        for m in testmols:
            try:
                r2 = f(m)
            except:
                continue
            l2 = len(r2)
            if l1 != l2:
                s["length_type"] = "dependend_length"
                break
            data.append(r2)

        if s["length_type"] == None:
            s["length_type"] = "independend_length"
            s["length"] = l1

    elif isinstance(r, (np.ndarray)):
        s["type"] = "numpy_arrays"
        s["dtype"] = "np." + str(r.dtype)
        s["length_type"] = None
        if len(r.flatten()) > MAX_LENGTH:
            s["length_type"] = "too_long"
            continue
        else:
            for m in testmols:
                r2 = f(m)
                if r.shape != r2.shape:
                    s["length_type"] = "dependend_length"
                    break
                data.append(r2)

            if s["length_type"] == None:
                s["length_type"] = "independend_length"
                s["length"] = r.flatten().shape[0]
            else:
                continue

    elif r.__class__.__name__.endswith("Vect"):
        # print(n,dir(r))
        s["type"] = "rdkit_vec"
        s["length_type"] = None
        try:
            l = r.GetLength()
        except:
            l = len(r)
        if l > MAX_LENGTH:
            # print(l, "lengt")
            s["length_type"] = "too_long"
            continue
        else:
            for m in testmols:
                r2 = f(m)
                try:
                    l2 = r2.GetLength()
                except:
                    l2 = len(r2)

                if l != l2:
                    s["length_type"] = "dependend_length"
                    break
                data.append(r2)

            if s["length_type"] == None:
                s["length_type"] = "independend_length"
                s["length"] = l
                if r.__class__.__name__.endswith("IntVect"):
                    if "long" in r.__class__.__name__.lower():
                        s["dtype"] = "np.int64"
                    else:
                        s["dtype"] = "np.int32"
                    continue
                elif r.__class__.__name__.endswith("BitVect"):
                    s["dtype"] = "bool"
                    continue
            else:
                continue
    else:
        s["type"] = "unknown"
        continue

    # print(n)
    if len(data) != len(testmols):
        data = [try_f(m) for m in testmols]
    data = [np.array(d) for d in data if d is not None]

    s["sample_data"] = data
    if len(data) < len(testmols) * 0.5:
        raise ValueError(len(data), len(testmols))
    # print(data)
    # print(s)
    # print(data)
    if s["length_type"] == "independend_length":
        a = np.stack(data)
    else:
        a = np.array(data, dtype=object)
    # print(a)
    if a.dtype != object:
        if a.dtype == float:
            try:
                np.array(a, dtype=np.float32)
                s["dtype"] = "np.float32"
            except OverflowError:
                s["dtype"] = "np.float64"
        elif a.dtype == int:
            try:
                np.array(a, dtype=np.int32)
                s["dtype"] = "np.int32"
            except OverflowError:
                s["dtype"] = "np.int64"
        else:
            s["dtype"] = "np." + str(a.dtype)
    else:
        # print(s)
        # print([i.dtype for i in a])
        dt = np.unique(np.array([i.dtype for i in a]))
        if len(dt) == 1:
            dt = dt[0]
            s["dtype"] = "np." + str(dt)
        else:
            print("AAA", dt)
            print(n, a)
    # print(n,s["dtype"])
succs

for s in succs:
    if "func_call" not in s:
        s["func_call"] = s["func_name"]
    if "additional_imports" not in s:
        s["additional_imports"] = []

# for s in succs:
#    if s["module"] == "self":
#        print(s["func_name"])

# valid
# numeric
# independend_length
# int_vec
# bit_vec, 1
# numpy_arrays

# invalid
# dependend_length
# stringf
# too_long_vec
# none_returned
# mol_returned
# exception_returned
# unknown
# numeric
# dtypes

# succs


len(succs)

from rdkit.Chem import rdqueries

from molNet.utils.mol import ATOMIC_SYMBOL_NUMBERS


def gen_f(num):
    def _f(mol):
        return len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom(num)))

    return _f


for sym, num in ATOMIC_SYMBOL_NUMBERS.items():
    if num > 0:
        succs.append(
            {
                "func_name": f"GetNumberAtoms{sym}",
                "func_call": f"len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom({num})))",
                "func": gen_f(num),
                "module": "custom_inline",
                "type": "numeric",
                "dtype": "np.uint32",
                "additional_imports": [["rdkit.Chem", "rdqueries"]],
            }
        )

        succs.append(
            {
                "func_name": f"GetRelativeContent{sym}",
                "func_call": f"len(mol.GetAtomsMatchingQuery(rdqueries.AtomNumEqualsQueryAtom({num})))/mol.GetNumAtoms()",
                "func": gen_f(num),
                "module": "custom_inline",
                "type": "numeric",
                "dtype": "np.float32",
                "additional_imports": [["rdkit.Chem", "rdqueries"]],
            }
        )

len(succs)

from collections import defaultdict

for s in succs:
    if "red_name" not in s:
        s["red_name"] = reduce_name(s["func_name"])
    if "classname" not in s:
        s["classname"] = f"{s['red_name']}_Featurizer"

old_succs = succs
succs = []
for i, s in enumerate(old_succs):
    dont_add = False
    if i > 0:
        for ss in old_succs[:i]:
            if ss["classname"] == s["classname"]:
                d1 = s["sample_data"]
                d2 = ss["sample_data"]
                if len(d1) != len(d2):
                    warn(f"double declaration with unuequal data size: {ss['classname']}")
                    continue
                if not all([np.array_equal(d1[k], d2[k]) for k in range(len(d1))]):
                    warn(f"double declaration with unuequal data content: {ss['classname']}")
                    continue

                dont_add = True
                break

    if not dont_add:
        succs.append(s)

succs = sorted(succs, key=lambda d: d["red_name"])

# avail_norms = []
# data_folder = "molecule_ecdf_data"
# if os.path.exists(data_folder):
#    df_cont = os.listdir(data_folder)
#    avail_norms = [f.replace(".data", "") for f in df_cont if ".data" in f]
#    avail_norms = [f for f in avail_norms if f + ".data" in df_cont]
#    avail_norms = [f for f in avail_norms if f + ".ecdf" in df_cont]
# len(avail_norms)

for s in succs:
    if s["func_name"] in MANUAL_PRECLASSCODES:
        if "preclasscode" not in s:
            s["preclasscode"] = MANUAL_PRECLASSCODES[s["func_name"]]
        else:
            s["preclasscode"] += "\n" + MANUAL_PRECLASSCODES[s["func_name"]]

imports = defaultdict(lambda: defaultdict(lambda: set()))


def numeric_coder(s):
    class_string_numeric = """class  {classname}(SingleValueMoleculeFeaturizer):
    # statics
    dtype={dtype}
    featurize=staticmethod({classcall})
    # normalization
    # functions
    
    """

    f1 = s["func"]
    mod_name1 = s["module"]
    if mod_name1 == "self":
        class_string_numeric = """class  {classname}(SingleValueMoleculeFeaturizer):
    # statics
    dtype={dtype}
    # normalization
    # functions
    def featurize(self,mol):
        return mol.{classcall}()
    """
    if mod_name1 == "custom_inline":
        class_string_numeric = """class  {classname}(SingleValueMoleculeFeaturizer):
    # statics
    dtype={dtype}
    # normalization
    # functions
    def featurize(self,mol):
        return {classcall}
    """

    for ss in succs:
        if ss == s:
            continue
        if "red_name" in ss:
            if ss["red_name"] == s["red_name"]:
                r1 = f1(Chem.Mol(testmols[0]))
                f2 = ss["func"]
                mod_name2 = ss["module"]
                r2 = f2(Chem.Mol(testmols[0]))
                if r1 != r2:
                    raise ValueError("doublefunc ({},{})".format(s, ss))
                else:
                    # print(s)
                    # print(ss)
                    # print()
                    continue
    for mod, imp in s["additional_imports"]:
        imports[s["type"]][mod].add(imp)
    imports[s["type"]][s["module"]].add(s["func_name"])
    code = ""
    if "preclasscode" in s:
        code += s["preclasscode"] + "\n"
    code += class_string_numeric.format(
        classname=s["classname"], classcall=s["func_call"], dtype=s["dtype"]
    )

    # if s["classname"] in avail_norms:
    #     # print(s["classname"])
    #     with open(os.path.join(data_folder, s["classname"] + ".data"), "r") as f:
    #         ecdf_data = json.load(f)["0"]
    #     idx = code.index("# normalization") + len("# normalization")
    #     precode = code[:idx] + "\n"
    #     postcode = code[idx:]
    #
    #     best = None
    #     global_data_keys = ["sample_bounds", "sample_bounds99"]
    #     global_data = {}
    #     for datakey, parakey, best_key in [
    #         ("linear_norm", "linear_norm_parameter", "linear"),
    #         ("min_max_norm", "min_max_norm_parameter", "min_max"),
    #         ("sig_norm", "sigmoidal_norm_parameter", "sig"),
    #         ("dual_sig_norm", "dual_sigmoidal_norm_parameter", "dual_sig"),
    #         ("genlog_norm", "genlog_norm_parameter", "genlog"),
    #     ]:
    #
    #         if datakey in ecdf_data:
    #             norm_data = ecdf_data[datakey]
    #             precode += f"    {parakey} = ({', '.join([str(i) for i in norm_data['parameter']])})"
    #             precode += f"  # error of {norm_data['error']:.2E} with sample range ({norm_data['sample_bounds'][0][0]:.2E},{norm_data['sample_bounds'][0][1]:.2E}) "
    #             precode += f"resulting in fit range ({norm_data['sample_bounds'][1][0]:.2E},{norm_data['sample_bounds'][1][1]:.2E})\n"
    #
    #             red_norm_data = {
    #                 k: v for k, v in norm_data.items() if k not in global_data_keys
    #             }
    #             del red_norm_data["parameter"]
    #             precode += f"    {parakey}_normdata ={red_norm_data}\n"
    #
    #             for k, v in norm_data.items():
    #                 if k in global_data_keys:
    #                     global_data[k] = v
    #
    #         if (
    #                 "sample_bounds99" not in norm_data
    #                 or norm_data["sample_bounds"][0][0] == norm_data["sample_bounds"][0][1]
    #         ):
    #             best = ("unity", 0, norm_data["sample_bounds"])
    #         else:
    #             if (
    #                     norm_data["sample_bounds"][1][0] <= 0.3
    #                     and norm_data["sample_bounds"][1][1] > 0.5
    #             ):
    #                 if best is None:
    #                     best = (
    #                         best_key,
    #                         norm_data["error"],
    #                         norm_data["sample_bounds"],
    #                     )
    #                 else:
    #                     if norm_data["error"] < best[1]:
    #                         best = (
    #                             best_key,
    #                             norm_data["error"],
    #                             norm_data["sample_bounds"],
    #                         )
    #     precode += f"    autogen_normdata ={global_data}\n"
    #     if best is not None:
    #         precode += f"    preferred_normalization = '{best[0]}'"
    #     code = precode + postcode
    s["code"] = code
    # print(code)


def rdkit_vec_coder(s):
    mod_name1 = s["module"]
    if mod_name1 == "self":
        raise NotImplementedError()
    if s["length_type"] == "independend_length":
        class_string = """class  {classname}(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = {length}
    dtype={dtype}
    # normalization
    # functions
    def featurize(self,mol):
        a=np.zeros(len(self),dtype=self.dtype)
        ConvertToNumpyArray({classcall}(mol),a)
        return a
        """
        code = ""
        if "preclasscode" in s:
            code += s["preclasscode"] + "\n"
        code += class_string.format(
            classname=s["classname"],
            classcall=s["func_call"],
            dtype=s["dtype"],
            length=s["length"],
        )
    elif s["length_type"] == "dependend_length":
        class_string = """class  {classname}(VarSizeMoleculeFeaturizer):
    # statics
    dtype={dtype}
    # normalization
    # functions
    
    def featurize(self,mol):
        r={classcall}(mol)
        try:
            l=r.GetLength()
        except:
            l=len(r)
            
        a=np.zeros(len(self),dtype=self.dtype)
        ConvertToNumpyArray(r,a)
        return a
        """
        code = ""
        if "preclasscode" in s:
            code += s["preclasscode"] + "\n"
        code += class_string.format(
            classname=s["classname"],
            classcall=s["func_call"],
            dtype=s["dtype"],
        )
    elif s["length_type"] == "too_long":
        return
    else:
        print(s)
        raise NotImplementedError(s["length_type"])

    imports[s["type"]][s["module"]].add(s["func_name"])
    s["code"] = code


def list_coder(s):
    mod_name1 = s["module"]
    if mod_name1 == "self":
        raise NotImplementedError()
    if s["length_type"] == "independend_length":
        class_string = """class  {classname}(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = {length}
    dtype={dtype}
    featurize=staticmethod({classcall})
    # normalization
    # functions
        """
        code = ""
        if "preclasscode" in s:
            code += s["preclasscode"] + "\n"
        code += class_string.format(
            classname=s["classname"],
            classcall=s["func_call"],
            dtype=s["dtype"],
            length=s["length"],
        )
    elif s["length_type"] == "dependend_length":
        class_string = """class {classname}(VarSizeMoleculeFeaturizer):
    # statics
    dtype={dtype}
    
    featurize=staticmethod({classcall})
    # normalization
    # functions
        """
        code = ""
        if "preclasscode" in s:
            code += s["preclasscode"] + "\n"
        code += class_string.format(
            classname=s["classname"],
            classcall=s["func_call"],
            dtype=s["dtype"],
        )
    elif s["length_type"] == "too_long":
        return
    else:
        print(s)
        raise NotImplementedError(s["length_type"])
    imports[s["type"]][s["module"]].add(s["func_name"])
    s["code"] = code


def numpy_arrays_coder(s):
    mod_name1 = s["module"]
    if mod_name1 == "self":
        raise NotImplementedError()
    if s["length_type"] == "independend_length":
        class_string = """class  {classname}(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = {length}
    dtype={dtype}
    # normalization
    # functions
    
    def featurize(self,mol):
        return {classcall}(mol).flatten()
        """
        code = ""
        if "preclasscode" in s:
            code += s["preclasscode"] + "\n"
        code += class_string.format(
            classname=s["classname"],
            classcall=s["func_call"],
            dtype=s["dtype"],
            length=s["length"],
        )
    elif s["length_type"] == "dependend_length":
        class_string = """class {classname}(VarSizeMoleculeFeaturizer):
    # statics
    dtype={dtype}
    # normalization
    # functions
    
    def featurize(self,mol):
        return {classcall}(mol).flatten()
        """
        code = ""
        if "preclasscode" in s:
            code += s["preclasscode"] + "\n"
        code += class_string.format(
            classname=s["classname"],
            classcall=s["func_call"],
            dtype=s["dtype"],
        )

    elif s["length_type"] == "too_long":
        return
    else:
        print(s)
        raise NotImplementedError(s["length_type"])
    imports[s["type"]][s["module"]].add(s["func_name"])
    s["code"] = code


def get_coder(t):
    if t == "none":
        return None
    elif t == "numeric":
        return numeric_coder
    elif t == "rdkit_vec":
        return rdkit_vec_coder
    elif t == "list":
        return list_coder
    elif t == "mol":
        return None
    elif t == "unknown":
        return None
    elif t == "numpy_arrays":
        return numpy_arrays_coder
    elif t == "string":
        return None
    elif t == "exception":
        return None
    else:
        raise ValueError("missing coder for " + str(t))


for s in succs:
    coder = get_coder(s["type"])
    # print(s)
    if coder:
        coder(s)

sheets = {}
for s in succs:
    if "code" in s:
        # print(s["code"])
        if s["type"] not in sheets:
            sheets[s["type"]] = []
        sheets[s["type"]].append(s)
sheets["numeric"]

for s, d in sheets.items():
    full_code = "from molNet.featurizer._molecule_featurizer import MoleculeFeaturizer,SingleValueMoleculeFeaturizer, FixedSizeMoleculeFeaturizer, VarSizeMoleculeFeaturizer\n"
    full_code += "import numpy as np\n"
    full_code += "from numpy import inf, nan\n"
    full_code += "from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray\n"
    for k, v in imports[s].items():
        if k == "self":
            continue
        if k == "custom_inline":
            continue
        full_code += "from {} import ({},)\n".format(k, ",".join(v))
    full_code += "\n" * 2

    for ss in d:
        full_code += ss["code"] + "\n\n"

    available_featurizer = []
    for ss in d:
        sn = "molecule_" + ss["red_name"]
        available_featurizer.append(sn)
        full_code += f'{sn}={ss["classname"]}()\n'

    full_code += "\n"
    full_code += "_available_featurizer = {\n"
    for af in available_featurizer:
        full_code += f"    '{af}' : {af},\n"
    full_code += "}\n"

    full_code += "def get_available_featurizer():\n    return _available_featurizer\n"

    full_code += "__all__ = ["
    for ss in d:
        full_code += "'{}',".format(ss["classname"])
    for af in available_featurizer:
        full_code += f"'{af}',"
    full_code += "]\n"

    full_code += """\n\n\n
def main():
    from rdkit import Chem
    testmol=Chem.MolFromSmiles("c1ccccc1")
    for n,f in get_available_featurizer().items():
        print(n,f(testmol))

if __name__=='__main__':
    main()
    """
    # print(full_code)

    code = black.format_str(full_code, mode=black.FileMode())
    with open(
            os.path.join(
                os.path.dirname(molecule_featurizer.__file__),
                f"_autogen_{os.path.basename(os.path.dirname(os.path.abspath(__file__)))}_{s}_molecule_featurizer.py",
            ),
            "w+b",
    ) as f:
        f.write(code.encode("utf8"))

    print(len(succs))
