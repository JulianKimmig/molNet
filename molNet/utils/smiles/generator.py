import random
from typing import List, Dict, Callable, Tuple, Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import MolToSmiles
from tqdm import tqdm
from molNet.utils.smiles import mol_from_smiles

from molNet.utils.mol.generator import (
    generate_random_carbon_lattice as mol_generate_random_carbon_lattice,
    generate_random_unsaturated_carbon_lattice as mol_generate_random_unsaturated_carbon_lattice,
    generate_random_hetero_carbon_lattice as mol_generate_random_hetero_carbon_lattice,
    fragment_random_molecule_generator as mol_fragment_random_molecule_generator,
    shuffle_buffer_mol_gen as mol_shuffle_buffer_mol_gen,
    random_carbon_lattice_generator as mol_random_carbon_lattice_generator,
    random_unsaturated_carbon_lattice_generator as mol_random_unsaturated_carbon_lattice_generator,
    random_hetero_carbon_lattice_generator as mol_random_hetero_carbon_lattice_generator,
    generate_n_random_carbon_lattice as mol_generate_n_random_carbon_lattic,
    generate_n_random_unsaturated_carbon_lattice as mol_generate_n_random_unsaturated_carbon_lattice,
    generate_n_random_hetero_carbon_lattice as mol_generate_n_random_hetero_carbon_lattice,
)


def _converter(moler):
    def _resp(*args, **kwargs):
        return MolToSmiles(moler(*args, **kwargs))

    return _resp


def _yielder(generator):
    def _resp(*args, **kwargs):
        for m in generator(*args, **kwargs):
            if m is not None:
                yield MolToSmiles(m)

    return _resp


def _lconverter(moler):
    def _resp(*args, **kwargs):
        return [MolToSmiles(m) for m in moler(*args, **kwargs) if m is not None]

    return _resp


generate_random_carbon_lattice = _converter(mol_generate_random_carbon_lattice)
generate_random_unsaturated_carbon_lattice = _converter(
    mol_generate_random_unsaturated_carbon_lattice
)
generate_random_hetero_carbon_lattice = _converter(
    mol_generate_random_hetero_carbon_lattice
)

fragment_random_molecule_generator = _yielder(mol_fragment_random_molecule_generator)
shuffle_buffer_mol_gen = _yielder(mol_shuffle_buffer_mol_gen)

random_carbon_lattice_generator = _yielder(mol_random_carbon_lattice_generator)
random_unsaturated_carbon_lattice_generator = _yielder(
    mol_random_unsaturated_carbon_lattice_generator
)
random_hetero_carbon_lattice_generator = _yielder(
    mol_random_hetero_carbon_lattice_generator
)

generate_n_random_carbon_lattice = _lconverter(mol_generate_n_random_carbon_lattic)
generate_n_random_unsaturated_carbon_lattice = _lconverter(
    mol_generate_n_random_unsaturated_carbon_lattice
)
generate_n_random_hetero_carbon_lattice = _lconverter(
    mol_generate_n_random_hetero_carbon_lattice
)
