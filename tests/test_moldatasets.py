import os
import unittest
from tempfile import gettempdir
from typing import Type

import numpy as np
from tqdm import tqdm

from molNet import MOLNET_LOGGER
from molNet.dataloader.molecular.ESOL import ESOL
from molNet.dataloader.molecular.Tox21 import Tox21Train
from molNet.dataloader.molecular.dataloader import MolDataLoader
from molNet.dataloader.molecular.prepmol import PreparedMolDataLoader, PreparedMolAdjacencyListDataLoader, \
    PreparedMolPropertiesDataLoader

MOLNET_LOGGER.setLevel("DEBUG")


class BaseTestClass:
    class DataSetTest(unittest.TestCase):
        DS_NAME: str = None
        DS_CLASS: Type[MolDataLoader] = None
        DS_KWARGS: dict = dict()

        TEST_DL = True

        ADJ_TEST_FIRST_SAMPLE=None

        def setUp(self) -> None:
            print("setup")
            assert self.DS_NAME is not None, "DS_NAME is not set"
            assert self.DS_CLASS is not None, "DS_CLASS is not set"

            self.dir = os.path.join(gettempdir(), "molNet", self.DS_NAME)
            print(self.dir)
            os.makedirs(self.dir, exist_ok=True)
            self.loader = self.DS_CLASS(self.dir, data_streamer_kwargs=self.DS_KWARGS)

        def test_dl(self):
            if self.TEST_DL:
                self.loader.download()

        def test_expected_size(self):
            self.loader.close()
            mol_count = 0
            iter_count = 0
            expdc = self.loader.expected_data_size
            expmc = self.loader.expected_mol_count
            for m in self.loader:
                iter_count += 1
                if m is not None:
                    mol_count += 1
            self.assertEqual(iter_count, expdc), "Expected data size does not match"
            self.assertEqual(mol_count, expmc), "Expected mol count does not match"

        def test_prepmol(self):
            self.loader.close()
            loader = PreparedMolDataLoader(self.loader, parent_dir=os.path.join(self.dir, "prepmol"))
            count = 0
            for m in tqdm(loader, total=loader.expected_mol_count):
                if m is not None:
                    count += 1
            self.assertEqual(count, loader.expected_mol_count), "Expected mol count does not match"

        def test_propmolproperties(self):
            self.loader.close()
            loader = PreparedMolPropertiesDataLoader(self.loader, parent_dir=os.path.join(self.dir, "prepmol"))
            count=0
            i=None
            for i in loader:
                count+=len(i)
            self.assertEqual(count, loader.expected_mol_count), "Expected mol count does not match"
            assert i is not None, "No data returned"
            self.assertEqual(i.index[-1], self.loader.expected_data_size-1), "Expected data size does not match"

        def test_prepmoladj(self):
            self.loader.close()
            loader = PreparedMolAdjacencyListDataLoader(self.loader, parent_dir=os.path.join(self.dir, "prepmol"))
            count = 0
            for m in tqdm(loader, total=loader.expected_mol_count):
                if m is not None:
                    count += 1
                    if count == 1 and self.ADJ_TEST_FIRST_SAMPLE is not None:
                        np.testing.assert_array_equal(m, self.ADJ_TEST_FIRST_SAMPLE), "Expected adjacency list does not match"
            self.assertEqual(count, loader.expected_mol_count), "Expected mol count does not match"


class ESOLTest(BaseTestClass.DataSetTest):
    DS_NAME = "ESOL"
    DS_CLASS = ESOL

    TEST_DL = False

    ADJ_TEST_FIRST_SAMPLE = np.array([[0, 1], [1, 2], [2, 3], [2, 4], [2, 5]])


class Tox21TrainTest(BaseTestClass.DataSetTest):
    DS_NAME = "Tox21Train"
    DS_CLASS = Tox21Train
    DS_KWARGS = dict(iter_None=True)

    TEST_DL = False

    ADJ_TEST_FIRST_SAMPLE = np.array(
        [[1, 2], [2, 3], [2, 12], [3, 4], [3, 9], [4, 5], [5, 6], [5, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
         [11, 17], [12, 13], [13, 14], [14, 15], [14, 16], [16, 17], [18, 19], [18, 28], [19, 20], [19, 25], [20, 21],
         [21, 22], [21, 23], [23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [27, 33], [28, 29], [29, 30], [30, 31],
         [30, 32], [32, 33]])
