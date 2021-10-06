import logging
import sys
import unittest

from molNet import MOLNET_LOGGER
from molNet.utils.mol.generator import (
    generate_random_carbon_lattice,
    generate_random_unsaturated_carbon_lattice,
    generate_random_hetero_carbon_lattice,
    random_carbon_lattice_generator,
    generate_n_random_carbon_lattice,
    random_unsaturated_carbon_lattice_generator,
    generate_n_random_unsaturated_carbon_lattice,
    random_hetero_carbon_lattice_generator,
    generate_n_random_hetero_carbon_lattice,
)

MOLNET_LOGGER.setLevel("DEBUG")


class MolTest(unittest.TestCase):
    def test_generate_random_carbon_lattice(self):
        m = generate_random_carbon_lattice(20, cross_rate=0)
        # rdkit.Chem.Draw.ShowMol(m)
        assert len(m.GetAtoms()) == 20
        assert len(m.GetBonds()) == 19, len(m.GetBonds())
        m = generate_random_carbon_lattice(20, cross_rate=1)
        assert len(m.GetAtoms()) == 20
        # rdkit.Chem.Draw.ShowMol(m)
        assert len(m.GetBonds()) > 24, len(m.GetBonds())

        assert len(set([a.GetSymbol() for a in m.GetAtoms()])) == 1

        d = [k for k in random_carbon_lattice_generator(rounds=10, max_c=5)]
        d = [k for k in generate_n_random_carbon_lattice(n=5, rounds=10, max_c=10)]

        # for k in d:
        #     rdkit.Chem.Draw.ShowMol(k)
        assert len(d) == 5
        # prof = pprofile.Profile()
        # with prof():
        d = [k for k in generate_n_random_carbon_lattice(n=1_000)]
        # prof.print_stats()

    def test_generate_random_unsaturated_carbon_lattice(self):
        m = generate_random_unsaturated_carbon_lattice(8)
        assert len(m.GetAtoms()) == 8
        d = [k for k in random_unsaturated_carbon_lattice_generator(rounds=10, max_c=5)]
        d = [k for k in generate_n_random_unsaturated_carbon_lattice(n=5, max_c=10)]

    def test_generate_random_hetero_carbon_lattice(self):
        m = generate_random_hetero_carbon_lattice()
        d = [k for k in random_hetero_carbon_lattice_generator(rounds=10, max_c=5)]

        d = [k for k in generate_n_random_hetero_carbon_lattice(n=1_000, max_c=10)]

    #    with cProfile.Profile() as pr:
    #        d = [k for k in generate_n_random_hetero_carbon_lattice(n=10_000, max_c=10)]
    #    pr.print_stats(sort="cumtime")
    #    for k in d:
    #        rdkit.Chem.Draw.ShowMol(k)
    #        break


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
