import urllib
from urllib.request import urlopen

import rdkit

from molNet.utils.polymers import (
    name_polymer_check,
    detect_polymer_type_by_name,
    monomer_to_repeating_unit_smiles,
)


def cactus(identifier):
    identifier = urllib.parse.quote_plus(identifier)
    try:
        url = "http://cactus.nci.nih.gov/chemical/structure/{}/smiles".format(
            identifier
        )
        ans = urlopen(url).read().decode("utf8")
        return ans
    except:
        return None


def opsin(identifier):
    identifier = urllib.parse.quote_plus(identifier)
    try:
        url = "https://opsin.ch.cam.ac.uk/opsin/{}.smi".format(identifier)
        ans = urlopen(url).read().decode("utf8")
        return ans
    except:
        return None


available_methods = {
    "cactus": cactus,
    "opsin": opsin,
}


def name_to_smiles(identifier, precheck=True):
    raw_answers = {}
    if precheck:
        new_identifier, is_polymer = name_polymer_check(identifier)
        if is_polymer:
            poly_type = detect_polymer_type_by_name(identifier)
            for smiles, data in name_to_smiles(new_identifier, precheck=False).items():
                poly_smiles = monomer_to_repeating_unit_smiles(smiles, poly_type)
                raw_answers[poly_smiles] = data
            return raw_answers

    for n, method in available_methods.items():
        raw_answers[n] = method(identifier)

    mol_answers = {
        n: rdkit.Chem.MolToSmiles(rdkit.Chem.MolFromSmiles(smiles))
        for n, smiles in raw_answers.items()
        if smiles is not None
    }
    if len(mol_answers) == 0:
        return None

    count_answers = {}
    for n, smiles in mol_answers.items():
        if smiles not in count_answers:
            count_answers[smiles] = 1
        else:
            count_answers[smiles] += 1

    smiles, counts = zip(*count_answers.items())
    counts, smiles = zip(*sorted(zip(counts, smiles)))

    ans = {}
    for i, smiles in enumerate(smiles):
        ans[smiles] = {
            "count": counts[i],
            "by": [
                n for n, test_smiles in mol_answers.items() if test_smiles == smiles
            ],
        }
    return ans
