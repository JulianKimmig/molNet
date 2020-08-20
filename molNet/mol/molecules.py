import networkx as nx
import numpy as np
import rdkit
import torch
import torch_geometric
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops, rdchem
from rdkit.Chem.Draw import rdMolDraw2D

import molNet.utils.base_classes as mnbc
from molNet.featurizer.atom_featurizer import default_atom_featurizer
from molNet.featurizer.molecule_featurizer import default_molecule_featurizer
from molNet.utils.mol.draw import mol_to_svg


class Molecule(mnbc.ValidatingObject):

    def __init__(self, mol, name=None):
        super().__init__()
        self._mol = mol
        self._name = name

    def __str__(self):
        if self._name:
            return self._name

        return Chem.MolToSmiles(self._mol)

    @property
    def mol(self):
        return self._mol

    def get_mol(self, with_numbers=False):
        mol = rdchem.EditableMol(self._mol)
        mol = mol.GetMol()

        if with_numbers:
            atoms = mol.GetNumAtoms()
            for idx in range(atoms):
                mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))

        return mol

    def to_svg(self, size=(200, 200), svg_data=None, mol_data=None):
        if mol_data is None:
            mol_data = {}
        return mol_to_svg(mol=self.get_mol(**mol_data),size=size,svg_data=svg_data)

    def to_molgraph(self, *args, **kwargs):
        return MolGraph.from_molecule(self, *args, **kwargs)


class MolGraph(nx.DiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.mol_features=[]
        self._mol = None

    def featurize(self, atom_featurizer=None, name="molNet_features", molecule_featurizer=None):
        if atom_featurizer is None:
            atom_featurizer = default_atom_featurizer

        if molecule_featurizer is not None:
            self.mol_features = molecule_featurizer(self.mol)
        else:
            self.mol_features=[]

        for n in self.nodes:
            node = self.nodes[n]
            node[name] = atom_featurizer(self.mol.GetAtomWithIdx(n))

    @property
    def mol(self):
        return self._mol

    @staticmethod
    def from_molecule(molecule, with_H=True, canonical_rank=True):
        g = MolGraph()

        mol = molecule.mol
        if with_H:
            mol = rdkit.Chem.AddHs(mol)
        elif with_H is None:
            pass
        else:
            mol = rdkit.Chem.RemoveHs(mol)

        if canonical_rank:
            atom_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, atom_order)

        for atom in mol.GetAtoms():
            g.add_node(atom.GetIdx())

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            g.add_edge(start, end)

        g._mol = mol
        return g

    def to_graph_input(self, node_feature_name="molNet_features", **add_kwargs):

        if node_feature_name not in self.nodes[next(iter(self.nodes))]:
            raise ValueError("'{}' not available, please call featurize first".format(node_feature_name))
        node_features = np.zeros((self.number_of_nodes(), len(self.nodes[next(iter(self.nodes))][node_feature_name])))
        for n in self.nodes:
            node = self.nodes[n]
            node_features[n] = node[node_feature_name]

        row, col = [], []
        for start, end in self.edges:
            row += [start, end]
            col += [end, start]

        edge_index = np.array([row, col])

        data = torch_geometric.data.data.Data(
            node_features=torch.from_numpy(node_features, ).float(),
            edge_index=torch.from_numpy(edge_index, ).long(),
            graph_features = self.mol_features,
            # y=y,
            **add_kwargs
        )

        return data
