import unittest

import pandas as pd

from molNet.dataloader.base_loader import (
    PandasDfLoader,
)
from molNet.dataloader.molecule_loader import (
    PytorchGeomMolDfLoader,
    MolGraphlDfLoader,
    MoleculeDfLoader,
)
from molNet.featurizer.atom_featurizer import atom_partial_charge, atom_formal_charge


class DataLoaderTest(unittest.TestCase):
    data = {
        "smiles": ["C", "CC", "CCC", "CCCC"],
        "mass": [12, 24, 36, 48],
        "charge": [0, 1, -2, -1],
        "n": [1, 2, 3, 4],
        "prop": [1.0, 2, 17, 0.32],
        "prop2": [0.0, 2.2, 170, 10],
        "prop3": [True, False, True, False],
    }
    df = pd.DataFrame(data)

    def test_df_loader(self):
        dfl = PandasDfLoader(self.df, y_columns=["prop", "prop2", "prop3"])
        dfl.setup()
        for batch in dfl.train_dataloader():
            print(batch)

    def test_molecule_df_loader(self):
        dfl = MoleculeDfLoader(self.df, y_columns=["prop", "prop2", "prop3"])
        dfl.setup()
        for batch in dfl.train_dataloader():
            print(batch)

    def test_molgraph_df_loader(self):
        dfl = MolGraphlDfLoader(self.df, y_columns=["prop", "prop2", "prop3"])
        dfl.setup()
        for batch in dfl.train_dataloader():
            print(batch)
            for g in batch:
                print(g.get_property_names())

    def test_pytorgeomolgraph_df_loader(self):
        for i in range(10):
            dfl = PytorchGeomMolDfLoader(
                self.df,
                y_columns=["prop", "prop2", "prop3"],
                atom_featurizer=atom_partial_charge,
                y_atom_featurizer=atom_formal_charge,
                # inplace=True,
                to_graph_input_kwargs=dict(
                    keep_string_data=True, include_graph_features_titles=True
                ),
            )
            dfl.setup()
        for batch in dfl.train_dataloader():
            print(batch)
            # print(batch.x)

            # print(batch.y)

            # print(batch.x_graph_features_titles)
            # print(batch.x_graph_features)

            # print(batch.y_graph_features_titles)
            # print(batch.y_graph_features)

            # print(batch.string_data_titles)
            # print(batch.string_data)

        print(self.df)
