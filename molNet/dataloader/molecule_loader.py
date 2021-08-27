import os
import pickle
import warnings

import pandas as pd
import torch_geometric
from rdkit import Chem
from torch.utils.data import DataLoader
from molNet.utils.multiprocess import multi_process_apply
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

from .base_loader import (
    SingleFileLoader,
    FileDataset,
    ObjectDataset,
    GeneratorDataLoader,
    GeneratorDataset,
    DirectDataLoader,
    DataFrameGenerator,
    DataFrameDataLoader,
    PandasDfLoader,
)
from ..mol.molecules import Molecule, molecule_from_smiles, MolGraph, MolGenerationError


class PytorchGeomMolGraphDataLoader(torch_geometric.data.DataLoader):
    #    def __init__(self, dataset, **kwargs):
    #       super().__init__(dataset, **kwargs,follow_batch=['graph_features'])
    pass


class MoleculeDfLoader(PandasDfLoader):
    dataloader = DirectDataLoader

    def __init__(
        self,
        df,
        molecule_column="molNet_mol",
        mol_create_source="smiles",
        mol_create_function=molecule_from_smiles,
        *args,
        **kwargs
    ):
        super().__init__(df, *args, **kwargs)
        self.molecule_column = molecule_column
        self.mol_create_source = mol_create_source
        self.mol_create_function = mol_create_function

    def generate_molecules(self):
        if self.molecule_column not in self.df.columns:
            def _create_mol(source):
                try:
                    return self.mol_create_function(source)
                except MolGenerationError as e:
                    print(e)
                    return None

            self.df[self.molecule_column] = multi_process_apply(
                self.df[self.mol_create_source],
                _create_mol,self.worker
            )

            for c in self.columns:
                if not c in self.df.columns:
                    raise Exception("expected collumn '{}' not in dataframe".format(c))

            for r, data in self.df[self.columns].iterrows():
                molecule = self.df.loc[r, self.molecule_column]
                if molecule:
                    for k, d in data.items():
#                        molecule.set_property(k, d.values)
                        molecule.set_property(k, d)

    def generate_full_dataset(self):
        for y_column in self.y_columns:
            if y_column in self.columns:
                self.columns.remove(y_column)
        self.generate_molecules()
        data = []
        for r, d in self.df[[self.molecule_column] + self.y_columns].iterrows():
            mol = d[self.molecule_column]
            if mol:
                data.append([mol, d[self.y_columns].tolist()])
        return data


class MolGraphlDfLoader(MoleculeDfLoader):
    def __init__(
        self,
        df,
        molgraph_column="molNet_graph",
        atom_featurizer=None,
        molecule_featurizer=None,
        y_atom_featurizer=None,
        y_molecule_featurizer=None,
        *args,
        **kwargs
    ):
        super().__init__(df, *args, **kwargs)
        self.molgraph_column = molgraph_column
        self.atom_featurizer = atom_featurizer
        self.molecule_featurizer = molecule_featurizer
        self.y_atom_featurizer = y_atom_featurizer
        self.y_molecule_featurizer = y_molecule_featurizer

    def generate_molgraph(self):
        super().generate_molecules()
        if self.molgraph_column not in self.df.columns:

            def _moltograph(row):
                mol = row[self.molecule_column]
                mg = MolGraph.from_molecule(mol)
                mg.featurize(
                    atom_featurizer=self.atom_featurizer,
                    molecule_featurizer=self.molecule_featurizer,
                )
                mg.featurize(
                    atom_featurizer=self.y_atom_featurizer,
                    molecule_featurizer=self.y_molecule_featurizer,
                    as_y=True,
                )
                for _y in self.y_columns:
                    mg.featurize(
                        molecule_featurizer=lambda mol: row[_y], name=_y, as_y=True
                    )
                return mg

            self.df[self.molgraph_column] = multi_process_apply(self.df[
                [self.molecule_column] + self.y_columns
            ],_moltograph,worker=self.worker, axis=1)

    def generate_full_dataset(self):
        for y_column in self.y_columns:
            if y_column in self.columns:
                self.columns.remove(y_column)
        self.generate_molgraph()
        # data = []
        # for r, d in self.df[[self.molgraph_column] + self.y_columns].iterrows():
        #    data.append([d[self.molgraph_column], d[self.y_columns].tolist()])

        return self.df[self.molgraph_column].tolist()


class PytorchGeomMolDfLoader(MolGraphlDfLoader):
    dataloader = PytorchGeomMolGraphDataLoader

    def __init__(
        self,
        df,
        pytorchgeomolgraph_column="molNet_ptgeograph",
        to_graph_input_kwargs={},
        *args,
        **kwargs
    ):
        super().__init__(df, *args, **kwargs)
        self.pytorchgeomolgraph_column = pytorchgeomolgraph_column
        self.to_graph_input_kwargs = to_graph_input_kwargs

    def generate_ptgmmolgraph(self):
        super().generate_molgraph()

        def _to_mol_graph(molgraph):
            return molgraph.to_graph_input(**self.to_graph_input_kwargs)

        if self.pytorchgeomolgraph_column not in self.df.columns:
            self.df[self.pytorchgeomolgraph_column] = multi_process_apply(
                self.df[
                self.molgraph_column
            ],_to_mol_graph,worker=self.worker)

    def generate_full_dataset(self):
        for y_column in self.y_columns:
            if y_column in self.columns:
                self.columns.remove(y_column)
        self.generate_ptgmmolgraph()
        return self.df[self.pytorchgeomolgraph_column].tolist()
    
    def load_data(self):
        import pickle
        with open(self.path,"rb") as f:
            data= pickle.load(f)
        return data
    
    def save_data(self,data):
        import pickle
        with open(self.path,"w+b") as f:
            pickle.dump(data,f)
        return True


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


# Loader Mixins
class SmilesLoaderMixin:
    def __init__(self, *args, **kwargs):
        super(SmilesLoaderMixin, self).__init__(*args, **kwargs)

    @staticmethod
    def generate_smiles_object(smiles, add_data):
        return {"smiles": smiles, "data": add_data}


class MoleculeLoaderMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def generate_molecule(smiles, add_data):
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)
        molecule = Molecule(mol)
        for name, add_date in add_data.items():
            molecule.set_property(name, add_date)
        return molecule


class MoleculeGraphLoaderMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def generate_mol_graph(smiles, add_data):
        molecule = MoleculeLoaderMixin.generate_molecule(smiles, add_data)
        molgraph = MolGraph.from_molecule(molecule)
        return molgraph


class PytorchGeomMolLoaderMixin:
    def __init__(self, *args, to_graph_params=None, **kwargs):
        if to_graph_params is None:
            to_graph_params = {}
        if "y_properties" not in to_graph_params:
            if "y_properties" in kwargs:
                to_graph_params["y_properties"] = kwargs["y_properties"]
            else:
                to_graph_params["y_properties"] = []
        if "with_properties" not in to_graph_params:
            if "with_properties" in kwargs:
                to_graph_params["with_properties"] = kwargs["with_properties"]
            else:
                to_graph_params["with_properties"] = True
        self.to_graph_params = to_graph_params
        super().__init__(*args, **kwargs)

    def generate_pytorch_geom_graph_input(self, smiles, add_data):
        molgraph = MoleculeGraphLoaderMixin.generate_mol_graph(smiles, add_data)
        molgraph.featurize(
            atom_featurizer=self.to_graph_params.get("atom_featurizer"),
            molecule_featurizer=self.to_graph_params.get("molecule_featurizer"),
        )
        gip = molgraph.to_graph_input(**self.to_graph_params)
        return gip


# loader


# FromCsvLoader
class SmilesFromCsvLoader(SmilesLoaderMixin, SingleFileLoader):
    prefix = "smiles_"
    save_dataset = FileDataset
    on_demand_dataset = ObjectDataset

    def __init__(
        self,
        smiles_column,
        file,
        additional_data=None,
        y_properties=None,
        *args,
        **kwargs
    ):
        super().__init__(source_file=file, *args, **kwargs)
        if additional_data is None:
            additional_data = []

        if not hasattr(self, "y_properties"):
            if y_properties is None:
                y_properties = []
            self.y_properties = y_properties
        self.additional_data = additional_data
        self.smiles_column = smiles_column

    def __repr__(self):
        return "{}_{}_{}".format(
            self.__class__.__name__, str(self.additional_data), str(self.y_properties)
        )

    def load_csv(self):
        df = pd.read_csv(self.source_file)
        columns = list(df.columns)
        if self.smiles_column not in columns:
            raise KeyError(
                "smiles column '{}' not in '{}'".format(
                    self.smiles_column, self.source_file
                )
            )

        columns.remove(self.smiles_column)
        add_data = self.additional_data
        if add_data == "all":
            add_data = columns
        add_data = add_data.copy()

        for yp in self.y_properties:
            if yp not in add_data:
                add_data.append(yp)

        for row, data in df.iterrows():
            yield row, data[self.smiles_column], {
                add_date: data[add_date] for add_date in add_data
            }

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
            folder = os.path.join(self.data_dir, self.data_map_entry["folder"])
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

    def prepare_data(self, return_data=False):
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


class MoleculeFromCsvLoader(MoleculeLoaderMixin, SmilesFromCsvLoader):
    prefix = "mol_"
    save_dataset = MoleculeFileDataset
    dataloader = DirectDataLoader

    def generate_data_element(self, row, smiles, add_data, save):
        molecule = self.generate_molecule(smiles, add_data)
        if save:
            Molecule.save(molecule, os.path.join(self.folder, self.prefix + str(row)))
        return molecule


class MoleculeGraphFromCsvLoader(MoleculeGraphLoaderMixin, MoleculeFromCsvLoader):
    dataloader = DirectDataLoader
    prefix = "molgraph_"
    save_dataset = MoleculeGraphFileDataset

    def generate_data_element(self, row, smiles, add_data, save):
        molgraph = self.generate_mol_graph(smiles, add_data)
        if save:
            MolGraph.save(molgraph, os.path.join(self.folder, self.prefix + str(row)))
        return molgraph


class PytorchGeomMolGraphFromCsvLoader(
    PytorchGeomMolLoaderMixin, MoleculeGraphFromCsvLoader
):
    dataloader = PytorchGeomMolGraphDataLoader
    prefix = "ptg_molgraph_"
    save_dataset = FileDataset

    def __repr__(self):
        return "{}_{}_{}".format(
            super().__repr__(), str(self.atom_featurizer), str(self.molecule_featurizer)
        )

    def generate_data_element(self, row, smiles, add_data, save):
        gip = self.generate_pytorch_geom_graph_input(smiles=smiles, add_data=add_data)
        if save:
            with open(os.path.join(self.folder, self.prefix + str(row)), "wb") as f:
                pickle.dump(gip, f)
        return gip


# From Generator
class SmilesGenerator(SmilesLoaderMixin, GeneratorDataset):
    def data_transformer(self, data):
        smiles, add_data = data
        return self.generate_smiles_object(smiles, add_data)


class SmilesFromGeneratorLoader(GeneratorDataLoader):
    def __init__(self, generator, *args, **kwargs):
        super().__init__(
            generator, generatordataset_class=SmilesGenerator, *args, **kwargs
        )


class MoleculeGenerator(MoleculeLoaderMixin, GeneratorDataset):
    def data_transformer(self, data):
        smiles, add_data = data
        return self.generate_molecule(smiles, add_data)


class MoleculeFromGeneratorLoader(GeneratorDataLoader):
    dataloader = DirectDataLoader

    def __init__(self, generator, *args, **kwargs):
        super().__init__(
            generator, generatordataset_class=MoleculeGenerator, *args, **kwargs
        )


class MoleculeGraphGenerator(MoleculeGraphLoaderMixin, GeneratorDataset):
    def data_transformer(self, data):
        smiles, add_data = data
        return self.generate_mol_graph(smiles, add_data)


class MoleculeGraphFromGeneratorLoader(GeneratorDataLoader):
    dataloader = DirectDataLoader

    def __init__(self, generator, *args, **kwargs):
        super().__init__(
            generator, *args, generatordataset_class=MoleculeGraphGenerator, **kwargs
        )


class PytorchGeomMolGraphGenerator(PytorchGeomMolLoaderMixin, GeneratorDataset):
    def data_transformer(self, data):
        smiles, add_data = data
        return self.generate_pytorch_geom_graph_input(smiles, add_data)


class PytorchGeomMolGraphFromGeneratorLoader(GeneratorDataLoader):
    dataloader = PytorchGeomMolGraphDataLoader


class SmilesfromDfGenerator(DataFrameGenerator):
    def __init__(self, df, smiles_col, **kwargs):
        super().__init__(df, processing=self.processing, **kwargs)
        self.add_data = list(df.columns)
        self.smiles_col = smiles_col
        self.add_data.remove(smiles_col)

    def processing(self, data):
        return data[self.smiles_col], {
            add_date: data[add_date] for add_date in self.add_data
        }


# From dataframe


class SmilesfromDfLoader(SmilesLoaderMixin, DataFrameDataLoader):
    on_demand_dataset = ObjectDataset

    def __init__(
        self,
        df,
        smiles_column,
        additional_data=None,
        y_properties=None,
        *args,
        **kwargs
    ):
        super().__init__(df=df, *args, **kwargs)
        if additional_data is None:
            additional_data = []

        if not hasattr(self, "y_properties"):
            if y_properties is None:
                y_properties = []
            self.y_properties = y_properties
        self.additional_data = additional_data
        self.smiles_column = smiles_column

    def __repr__(self):
        return "{}_{}_{}".format(
            self.__class__.__name__, str(self.additional_data), str(self.y_properties)
        )

    def load_csv(self):
        columns = list(self.df.columns)
        if self.smiles_column not in columns:
            raise KeyError("smiles column '{}' not in df".format(self.smiles_column))

        columns.remove(self.smiles_column)
        add_data = self.additional_data
        if add_data == "all":
            add_data = columns
        add_data = add_data.copy()

        for yp in self.y_properties:
            if yp not in add_data:
                add_data.append(yp)

        for row, data in self.df.iterrows():
            yield row, data[self.smiles_column], {
                add_date: data[add_date] for add_date in add_data
            }

    def generate_full_dataset(self):
        return self.on_demand_dataset(objects=self.prepare_data(return_data=True))

    def generate_data_element(self, smiles, add_data):
        obj = self.generate_smiles_object(smiles, add_data)
        return obj

    def prepare_data(self, return_data=False):
        super().prepare_data()
        objcts = []
        for row, smiles, add_data in self.load_csv():
            obj = self.generate_data_element(smiles, add_data)
            if return_data:
                objcts.append(obj)
        return objcts


class MoleculeFromDfLoader(MoleculeLoaderMixin, SmilesfromDfLoader):
    dataloader = DirectDataLoader

    def generate_data_element(self, smiles, add_data):
        molecule = self.generate_molecule(smiles, add_data)
        return molecule


class MoleculeGraphFromDfLoader(MoleculeGraphLoaderMixin, MoleculeFromDfLoader):
    dataloader = DirectDataLoader

    def generate_data_element(
        self,
        smiles,
        add_data,
    ):
        molgraph = self.generate_mol_graph(smiles, add_data)
        return molgraph


class PytorchGeomMolGraphFromDfLoader(
    PytorchGeomMolLoaderMixin, MoleculeGraphFromDfLoader
):
    dataloader = PytorchGeomMolGraphDataLoader

    def __repr__(self):
        return "{}_{}_{}".format(
            super().__repr__(),
            str(self.to_graph_params.get("atom_featurizer")),
            str(self.to_graph_params.get("molecule_featurizer")),
        )

    def generate_data_element(self, smiles, add_data):
        gip = self.generate_pytorch_geom_graph_input(smiles=smiles, add_data=add_data)
        return gip
