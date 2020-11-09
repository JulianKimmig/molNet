# TODO Data dir managemnt
import io

import requests
import pandas as pd

from molNet.dataloader.base_loader import df_to_generator
from molNet.dataloader.molecule_loader import SmilesfromDfGenerator, PytorchGeomMolGraphGenerator


class DataSet():
    pass


class DfDataSet(DataSet):
    def __init__(self, df=None):
        self._df = df

    @property
    def df(self):
        if self._df is None:
            self.get()
        return self._df

    def get(self):
        raise NotImplementedError


class DelaneySolubility(DfDataSet):
    smiles_col = "SMILES"
    url = "https://pubs.acs.org/doi/suppl/10.1021/ci034243x/suppl_file/ci034243xsi20040112_053635.txt"
    y_properties = ['measured log(solubility:mol/L)']
    name_columns='Compound ID'

    def get(self):
        s = requests.get(self.url).content
        self._df = pd.read_csv(io.StringIO(s.decode('utf-8')))
        self._df.rename(columns={self.name_columns: "name"},inplace=True)

    def to_generator(self, **kwargs):
        return df_to_generator(self.df, generator_class=SmilesfromDfGenerator, smiles_col=self.smiles_col, **kwargs)

    def to_pytorchgeo_molgraph_generator(self, to_graph_params=None, generator_params=None):
        if generator_params is None:
            generator_params = {}
        if to_graph_params is None:
            to_graph_params = {'with_properties': True,
                               'y_properties':self.y_properties,
                                }

        if 'y_properties' not in to_graph_params:
            to_graph_params['y_properties'] = self.y_properties

        return [PytorchGeomMolGraphGenerator(generator=g, to_graph_params = to_graph_params)
                for g in self.to_generator(**generator_params)]
