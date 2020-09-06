import os
import pickle

import pandas as pd
import torch_geometric
from rdkit import Chem
from torch.utils.data import DataLoader

from .base_loader import SingleFileLoader, FileDataset, ObjectDataset, GeneratorDataLoader, GeneratorDataset, \
    DirectDataLoader
from ..mol.molecules import Molecule, MolGraph


class MoleculeFileDataset(FileDataset):
    def __getitem__(self, item):
        file = self.file_list[item]
        if self.parent_folder:
            file = os.path.join(self.parent_folder, file)
        d = Molecule.load(file)
        return d


class MoleculeGraphFileDataset(FileDataset):
    def __getitem__(self, item):
        file = self.file_list[item]
        if self.parent_folder:
            file = os.path.join(self.parent_folder, file)

        d = MolGraph.load(file)
        return d


class SmilesLoaderMixin():
    def __init__(self, *args, **kwargs):
        super(SmilesLoaderMixin, self).__init__(*args, **kwargs)

    @staticmethod
    def generate_smiles_object(smiles, add_data):
        return {'smiles': smiles, 'data': add_data}


class SmilesFromCsvLoader(SmilesLoaderMixin, SingleFileLoader):
    prefix = "smiles_"
    save_dataset = FileDataset
    on_demand_dataset = ObjectDataset

    def __init__(self, smiles_column, file, additional_data=None, y_properties=None, *args, **kwargs):
        super().__init__(source_file=file, *args, **kwargs)
        if additional_data is None:
            additional_data = []
        if y_properties is None:
            y_properties = []
        self.y_properties = y_properties
        self.additional_data = additional_data
        self.smiles_column = smiles_column

    def __repr__(self):
        return "{}_{}_{}".format(self.__class__.__name__, str(self.additional_data), str(self.y_properties))

    def load_csv(self):
        df = pd.read_csv(self.source_file)
        columns = list(df.columns)
        if self.smiles_column not in columns:
            raise KeyError("smiles column '{}' not in '{}'".format(self.smiles_column, self.source_file))

        columns.remove(self.smiles_column)
        add_data = self.additional_data
        if add_data == "all":
            add_data = columns
        add_data = add_data.copy()

        for yp in self.y_properties:
            if yp not in add_data:
                add_data.append(yp)

        for row, data in df.iterrows():
            yield row, data[self.smiles_column], {add_date: data[add_date] for add_date in add_data}

    def check_reload(self):
        if not self.save:
            return True
        reload = self.reload
        if not self.reload:
            found = False
            for f in os.listdir(self.folder):
                if f.startswith(self.prefix):
                    found = True
                    break
            if not found:
                reload = True
        return reload

    def generate_full_dataset(self):
        if self.save:
            file_list = []
            folder = os.path.join(self.data_dir, self.data_map_entry['folder'])
            for f in os.listdir(folder):
                if f.startswith(self.prefix):
                    file_list.append(f)
            return self.save_dataset(file_list=file_list, parent_folder=folder)
        else:
            return self.on_demand_dataset(objects=self.prepare_data(return_data=True))

    def generate_data_element(self, row, smiles, add_data, save):
        obj = self.generate_smiles_object(smiles, add_data)
        if save:
            with open(os.path.join(self.folder, self.prefix + str(row)), "wb") as f:
                pickle.dump(obj, f)
        return obj

    def prepare_data(self,return_data=False):
        super().prepare_data()
        if return_data:
            objcts = []
        if self.check_reload():
            for row, smiles, add_data in self.load_csv():
                obj = self.generate_data_element(row, smiles, add_data, save=self.save)
                if return_data:
                    objcts.append(obj)
        if return_data:
            return objcts
        return

class MoleculeLoaderMixin():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def generate_molecule(smiles, add_data):
        molecule = Molecule(Chem.MolFromSmiles(smiles))
        for name, add_date in add_data.items():
            molecule.set_property(name, add_date)
        return molecule


class MoleculeFromCsvLoader(MoleculeLoaderMixin, SmilesFromCsvLoader):
    prefix = "mol_"
    save_dataset = MoleculeFileDataset
    dataloader = DirectDataLoader
    def generate_data_element(self, row, smiles, add_data, save):
        molecule = self.generate_molecule(smiles, add_data)
        if save:
            Molecule.save(molecule, os.path.join(self.folder, self.prefix + str(row)))
        return molecule


class MoleculeGraphLoaderMixin():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def generate_mol_graph(smiles, add_data):
        molecule = MoleculeLoaderMixin.generate_molecule(smiles, add_data)
        molgraph = MolGraph.from_molecule(molecule)
        return molgraph


class MoleculeGraphFromCsvLoader(MoleculeGraphLoaderMixin, MoleculeFromCsvLoader):
    dataloader = DirectDataLoader
    prefix = "molgraph_"
    save_dataset = MoleculeGraphFileDataset

    def generate_data_element(self, row, smiles, add_data, save):
        molgraph = self.generate_mol_graph(smiles, add_data)
        if save:
            MolGraph.save(molgraph, os.path.join(self.folder, self.prefix + str(row)))
        return molgraph


class PytorchGeomMolLoaderMixin():
    def __init__(self, atom_featurizer=None, molecule_featurizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "y_properties"):
            self.y_properties = []
        self.molecule_featurizer = molecule_featurizer
        self.atom_featurizer = atom_featurizer

    def generate_pytorch_geom_graph_input(self, smiles, add_data, ):
        molgraph = MoleculeGraphLoaderMixin.generate_mol_graph(smiles, add_data)
        molgraph.featurize(atom_featurizer=self.atom_featurizer, molecule_featurizer=self.molecule_featurizer)
        gip = molgraph.to_graph_input(y_properties=self.y_properties)
        return gip


class PytorchGeomMolGraphFromCsvLoader(PytorchGeomMolLoaderMixin, MoleculeGraphFromCsvLoader):
    dataloader = lambda s, *args, **kwargs: torch_geometric.data.DataLoader(*args, **kwargs,
                                                                            follow_batch=['graph_features'])
    prefix = "ptg_molgraph_"
    save_dataset = FileDataset

    def __repr__(self):
        return "{}_{}_{}".format(super().__repr__(), str(self.atom_featurizer),
                                 str(self.molecule_featurizer))

    def generate_data_element(self, row, smiles, add_data, save):
        gip = self.generate_pytorch_geom_graph_input(smiles=smiles, add_data=add_data)
        if save:
            with open(os.path.join(self.folder, self.prefix + str(row)), "wb") as f:
                pickle.dump(gip, f)
        return gip


class SmilesGenerator(SmilesLoaderMixin, GeneratorDataset):
    def data_transformer(self, data):
        smiles, add_data = data
        return self.generate_smiles_object(smiles, add_data)


class SmilesFromGeneratorLoader(GeneratorDataLoader):
    def __init__(self, generator, *args, **kwargs):
        super().__init__(generator, generatordataset_class=SmilesGenerator, *args, **kwargs)

class MoleculeGenerator(MoleculeLoaderMixin, GeneratorDataset):
    def data_transformer(self, data):
        smiles, add_data = data
        return self.generate_molecule(smiles, add_data)


class MoleculeFromGeneratorLoader(GeneratorDataLoader):
    dataloader = DirectDataLoader

    def __init__(self, generator, *args, **kwargs):
        super().__init__(generator, generatordataset_class=MoleculeGenerator, *args, **kwargs)

class MoleculeGraphGenerator(MoleculeGraphLoaderMixin, GeneratorDataset):

    def data_transformer(self, data):
        smiles, add_data = data
        return self.generate_mol_graph(smiles, add_data)


class MoleculeGraphFromGeneratorLoader(GeneratorDataLoader):
    dataloader = DirectDataLoader

    def __init__(self, generator, *args, **kwargs):
        super().__init__(generator, *args, generatordataset_class=MoleculeGraphGenerator, **kwargs)
