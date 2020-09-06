import time
import unittest

import numpy as np

from molNet.dataloader.molecule_loader import MoleculeFromCsvLoader, MoleculeGraphFromCsvLoader, \
    PytorchGeomMolGraphFromCsvLoader, \
    SmilesFromCsvLoader
from molNet.featurizer.atom_featurizer import default_atom_featurizer, atom_hybridization_one_hot, atom_mass
from molNet.featurizer.featurizer import FeaturizerList
from molNet.featurizer.molecule_featurizer import default_molecule_featurizer, molecule_num_heavy_atoms
from molNet.mol.molecules import DATATYPES


class DataLoaderTest(unittest.TestCase):
    new_load_time = 0

    def test_local_csv_smiles_loader_multifile__A_force_new(self):
        s = time.time()
        loader = SmilesFromCsvLoader(file="../datasets/delaney-processed.csv", smiles_column="smiles", reload=True,
                                     additional_data='all',
                                     seed=42, batch_size=3)
        loader.prepare_data()
        loader.setup()
        DataLoaderTest.new_load_time = time.time() - s
        for batch in loader.train_dataloader():
            testdata = ['CC=CC=O', 'CCc1ccccc1C', 'Oc1cc(Cl)ccc1Oc2ccc(Cl)cc2Cl']
            assert batch['smiles'] == testdata
            assert len(set(batch['data']['Compound ID']).symmetric_difference(
                {'2-butenal', '2-Ethyltoluene', 'Triclosan'})) == 0
            break

    def test_local_csv_smiles_loader_multifile__B__reload(self):
        s = time.time()
        loader = SmilesFromCsvLoader(file="../datasets/delaney-processed.csv", smiles_column="smiles", reload=False,
                                     additional_data='all',
                                     seed=42, batch_size=3)
        loader.prepare_data()
        loader.setup()
        assert DataLoaderTest.new_load_time / 10 > time.time() - s, "loading took too long ({})".format(
            time.time() - s)
        for batch in loader.train_dataloader():
            testdata = ['CC=CC=O', 'CCc1ccccc1C', 'Oc1cc(Cl)ccc1Oc2ccc(Cl)cc2Cl']
            assert batch['smiles'] == testdata
            assert len(set(batch['data']['Compound ID']).symmetric_difference(
                {'2-butenal', '2-Ethyltoluene', 'Triclosan'})) == 0
            break

    def test_local_csv_smiles_loader_multifile__C_on_demand(self):
        s = time.time()
        loader = SmilesFromCsvLoader(file="../datasets/delaney-processed.csv", smiles_column="smiles", save=False,
                                     additional_data='all',
                                     seed=42, batch_size=3)
        loader.prepare_data()
        loader.setup()
        DataLoaderTest.new_load_time = time.time() - s
        for batch in loader.val_dataloader():
            testdata = ['Clc1ccc(cc1Cl)c2cc(Cl)c(Cl)c(Cl)c2Cl ', 'CCCCCCc1ccccc1', 'CC12CC(=O)C3C(CCC4=CC(=O)CCC34C)C2CCC1(O)C(=O)CO']
            assert batch['smiles'] == testdata,batch['smiles']
            assert len(set(batch['data']['Compound ID']).symmetric_difference(
                {"2,3,3',4,4',5-PCB", 'Hexylbenzene ', 'Cortisone'})) == 0,batch['data']['Compound ID']
            break

    def test_local_csv_mol_loader_multifile__A_force_new(self):
        s = time.time()
        loader = MoleculeFromCsvLoader(file="../datasets/delaney-processed.csv", smiles_column="smiles", reload=True,
                                       additional_data='all',
                                       seed=42, batch_size=10)
        loader.prepare_data()
        loader.setup()
        DataLoaderTest.new_load_time = time.time() - s
        for batch in loader.train_dataloader():
            assert len(
                set([d.as_dict()['properties'][DATATYPES.STRING]['Compound ID'] for d in batch]).symmetric_difference(
                    {'hydrazobenzene',
                     'Hexylbenzene ',
                     '2-butenal',
                     'bromadiolone',
                     'Butan-2-ol',
                     'Diethyl sulfide',
                     'Ethyl hexanoate',
                     'DDT',
                     'Octane',
                     '1,3,5-Trimethylbenzene '})) == 0
            break

    def test_local_csv_mol_loader_multifile__B__reload(self):
        s = time.time()
        loader = MoleculeFromCsvLoader(file="../datasets/delaney-processed.csv", smiles_column="smiles", reload=False,
                                       additional_data='all',
                                       seed=42, batch_size=10)
        loader.prepare_data()
        loader.setup()
        assert DataLoaderTest.new_load_time / 10 > time.time() - s, "loading took too long ({})".format(
            time.time() - s)
        for batch in loader.train_dataloader():
            assert len(
                set([d.as_dict()['properties'][DATATYPES.STRING]['Compound ID'] for d in batch]).symmetric_difference(
                    {'hydrazobenzene',
                     'Hexylbenzene ',
                     '2-butenal',
                     'bromadiolone',
                     'Butan-2-ol',
                     'Diethyl sulfide',
                     'Ethyl hexanoate',
                     'DDT',
                     'Octane',
                     '1,3,5-Trimethylbenzene '})) == 0
            break

    def test_local_csv_mol_loader_multifile__C_on_demand(self):
        s = time.time()
        loader = MoleculeFromCsvLoader(file="../datasets/delaney-processed.csv", smiles_column="smiles", save=False,
                                       additional_data='all',
                                       seed=42, batch_size=10)
        loader.prepare_data()
        loader.setup()
        DataLoaderTest.new_load_time = time.time() - s
        assert len(loader.test_dataloader()) == 17, len(loader.test_dataloader())
        for batch in loader.test_dataloader():
            print([d.as_dict()['properties'][DATATYPES.STRING]['Compound ID'] for d in batch])
            assert len(
                set([d.as_dict()['properties'][DATATYPES.STRING]['Compound ID'] for d in batch]).symmetric_difference(
                    {'1-Iodoheptane', 'Hexamethylbenzene', 'Lorazepam', '1,2,4,5-Tetrachlorobenzene',
                     '2-Methylphenanthrene', 'Lenacil', 'Dicofol', 'Coumatetralyl', 'cyclobarbital',
                     'Chlorthalidone'})) == 0
            break

    def test_local_csv_molgraph_loader_multifile__A_force_new(self):
        s = time.time()
        loader = MoleculeGraphFromCsvLoader(file="../datasets/delaney-processed.csv", smiles_column="smiles",
                                            reload=True, additional_data='all',
                                            seed=42, batch_size=3)
        loader.prepare_data()
        loader.setup()
        DataLoaderTest.new_load_time = time.time() - s
        for batch in loader.train_dataloader():
            batch = [b.as_dict() for b in batch]

            assert batch[0]['properties'][DATATYPES.STRING]['smiles'] == 'CCCCCCO', \
            batch[0]['properties'][DATATYPES.STRING]['smiles']
            assert batch[1]['properties'][DATATYPES.FLOAT]['measured log solubility in mols per litre'] == -4.16
            test = set([(4, 5),
                        (4, 6),
                        (4, 7),
                        (5, 3),
                        (5, 0),
                        (5, 2),
                        (6, 1),
                        (11, 4),
                        (11, 8),
                        (11, 9),
                        (11, 10)])
            assert len(set(batch[2]['edges']).symmetric_difference(test)) == 0

            break

    def test_local_csv_molgraph_loader_multifile__B__reload(self):

        s = time.time()
        loader = MoleculeGraphFromCsvLoader(file="../datasets/delaney-processed.csv", smiles_column="smiles",
                                            reload=False, additional_data='all',
                                            seed=42, batch_size=3)
        loader.prepare_data()
        loader.setup()

        assert DataLoaderTest.new_load_time / 10 > time.time() - s, "loading took too long ({})".format(
            time.time() - s)
        for batch in loader.train_dataloader():
            [b.to_graph_input() for b in batch]
            batch = [b.as_dict() for b in batch]
            assert batch[1]['properties'][DATATYPES.FLOAT]['measured log solubility in mols per litre'] == -4.16

            break

    def test_local_csv_molgraph_loader_multifile__C_on_demand(self):
        s = time.time()
        loader = MoleculeGraphFromCsvLoader(file="../datasets/delaney-processed.csv", smiles_column="smiles",
                                            save=False, additional_data='all',
                                            seed=42, batch_size=3)
        loader.prepare_data()
        loader.setup()
        DataLoaderTest.new_load_time = time.time() - s
        for batch in loader.train_dataloader():
            batch = [b.as_dict() for b in batch]

            assert batch[0]['properties'][DATATYPES.STRING]['smiles'] == 'CCCCc1c(C)nc(N(C)C)nc1O', \
            batch[0]['properties'][DATATYPES.STRING]['smiles']
            assert batch[1]['properties'][DATATYPES.FLOAT]['measured log solubility in mols per litre'] == -4.328,\
                batch[1]['properties'][DATATYPES.FLOAT]['measured log solubility in mols per litre']
            test = {(0, 13), (1, 6), (1, 8), (7, 10), (7, 1), (10, 0), (12, 7), (12, 4), (13, 12), (14, 15), (15, 16),
                    (15, 9), (16, 17), (16, 2), (16, 5), (17, 18), (17, 11), (17, 3), (18, 0), (18, 14), (19, 14)}
            assert len(set(batch[2]['edges']).symmetric_difference(test)) == 0,batch[2]['edges']

            break

    def test_local_csv_ptgeom_molgraph_loader_multifile__A_force_new(self):
        s = time.time()
        loader = PytorchGeomMolGraphFromCsvLoader(file="../datasets/delaney-processed.csv", smiles_column="smiles",
                                                  reload=True,
                                                  seed=42, batch_size=3, additional_data='all',
                                                  y_properties=[
                                                      "measured log solubility in mols per litre",
                                                  ],
                                                  atom_featurizer=FeaturizerList([
                                                      atom_hybridization_one_hot,
                                                      atom_mass,
                                                  ]),
                                                  molecule_featurizer=FeaturizerList([molecule_num_heavy_atoms]))
        loader.prepare_data()
        loader.setup()
        DataLoaderTest.new_load_time = time.time() - s
        for batch in loader.train_dataloader():
            print(batch['string_data'])
            test_data = ['', 'Nc1ncnc2c1ccn2C1OC(CO)C(O)C1O', 'tubercidin']
            assert all(
                batch['string_data'][0][i] == test_data[i] for i in range(len(test_data))), "strings for 0 dont match"
            test_data = ['', 'CCOP(=O)(OCC)OCC', 'Triethyl phosphate']
            assert all(
                batch['string_data'][1][i] == test_data[i] for i in range(len(test_data))), "strings for 1 dont match"

            test_data = np.array([19.0000, -0.8920, 1.0000, 266.2570, 4.0000, 3.0000, 2.0000,
                                  126.6500, 11.0000, -0.9530, 1.0000, 182.1560, 0.0000, 0.0000,
                                  6.0000, 44.7600, 10.0000, -3.0990, 1.0000, 146.1110, 0.0000,
                                  1.0000, 0.0000, 0.0000])
            assert np.allclose(test_data, batch['graph_features'].numpy()), batch['graph_features'].numpy()

            test_data = np.array([-1.9500, 0.4300, -2.5100])
            assert np.allclose(test_data, batch['y'].numpy()), batch['y'].numpy()
            break

    def test_local_csv_ptgeom_molgraph_loader_multifile__B__reload(self):
        s = time.time()
        loader = PytorchGeomMolGraphFromCsvLoader(file="../datasets/delaney-processed.csv", smiles_column="smiles",
                                                  reload=False,
                                                  y_properties=[
                                                      "measured log solubility in mols per litre",
                                                  ], additional_data='all',
                                                  seed=42, batch_size=3,
                                                  atom_featurizer=FeaturizerList([
                                                      atom_hybridization_one_hot,
                                                      atom_mass,
                                                  ]),
                                                  molecule_featurizer=FeaturizerList([molecule_num_heavy_atoms]))
        loader.prepare_data()
        loader.setup()
        assert DataLoaderTest.new_load_time / 10 > time.time() - s, "loading took too long ({})".format(
            time.time() - s)
        for batch in loader.train_dataloader():
            test_data = ['', 'Nc1ncnc2c1ccn2C1OC(CO)C(O)C1O', 'tubercidin']
            assert all(
                batch['string_data'][0][i] == test_data[i] for i in range(len(test_data))), "strings for 0 dont match"
            break

    def test_local_csv_ptgeom_molgraph_loader_multifile__C_on_demand(self):
        s = time.time()
        loader = PytorchGeomMolGraphFromCsvLoader(file="../datasets/delaney-processed.csv", smiles_column="smiles",
                                                  save=False,
                                                  seed=42, batch_size=3, additional_data='all',
                                                  y_properties=[
                                                      "measured log solubility in mols per litre",
                                                  ],
                                                  atom_featurizer=FeaturizerList([
                                                      atom_hybridization_one_hot,
                                                      atom_mass,
                                                  ]),
                                                  molecule_featurizer=FeaturizerList([molecule_num_heavy_atoms]))
        loader.prepare_data()
        loader.setup()
        DataLoaderTest.new_load_time = time.time() - s
        for batch in loader.train_dataloader():
            print(batch['string_data'])
            test_data = ['', 'CCCCc1c(C)nc(N(C)C)nc1O', 'dimethirimol']
            assert all(
                batch['string_data'][0][i] == test_data[i] for i in range(len(test_data))), "strings for 0 dont match"
            test_data = ['', 'CN(C)C(=O)Nc1ccc(-n2nc(C(C)(C)C)oc2=O)c(Cl)c1', 'Dimefuron']
            assert all(
                batch['string_data'][1][i] == test_data[i] for i in range(len(test_data))), "strings for 1 dont match"

            test_data = np.array([15.0000,  -3.5700,   1.0000, 209.2930,   1.0000,   1.0000,   4.0000,
                                  49.2500,  23.0000,  -3.8310,   1.0000, 338.7950,   1.0000,   2.0000,
                                  2.0000,  80.3700,  14.0000,  -1.9480,   1.0000, 214.2060,   1.0000,
                                  2.0000,   2.0000,  88.3700])
            assert np.allclose(test_data, batch['graph_features'].numpy()), batch['graph_features']

            test_data = np.array([-2.24,  -4.328, -3.22])
            assert np.allclose(test_data, batch['y'].numpy()), batch['y'].numpy()
            break
