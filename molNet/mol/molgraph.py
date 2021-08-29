import networkx as nx
import numpy as np
from rdkit.Chem.rdchem import Mol

from molNet.featurizer._atom_featurizer import AtomFeaturizer
from molNet.mol.molecule import Molecule


class MolGraph(nx.DiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)
        self._mol = None
        self._properties = {}

    def get_property(self, name):
        return self._properties[name]

    def set_property(self, name, value):
        self._properties[name] = value

    def featurize_mol(self, mol_featurizer: AtomFeaturizer, name: str = None):
        if name is None:
            name = str(mol_featurizer)
        self.set_property(name, mol_featurizer(self.mol))

    def featurize_atoms(self, atom_featurizer: AtomFeaturizer, name: str = None):
        if name is None:
            name = str(atom_featurizer)
        for n in self.nodes:
            node = self.nodes[n]
            node[name] = atom_featurizer(self.mol.GetAtomWithIdx(n))

    def get_atom_features_dict(self):
        features = {}
        for k in self.nodes[0].keys():
            features[k] = self.get_atom_feature(k)
        return features

    def get_atom_feature(self, feature_name):
        feature = np.zeros((len(self)), dtype=self.nodes[0][feature_name].dtype)
        for node_id, node_data in self.nodes(data=True):
            feature[node_id] = node_data[feature_name]
        return feature

    def get_mol(self):
        return self._mol

    @property
    def mol(self):
        return self.get_mol()

    @property
    def ege_array(self):
        return np.array(self.edges)

    def as_arrays(self):
        return {
            "size": len(self),
            "eges": self.ege_array,
            "node_features": self.get_atom_features_dict(),
            "graph_features": self._properties,
        }


def mol_graph_from_molecule(
    molecule: Molecule, with_H: bool = True, canonical_rank: bool = True
) -> MolGraph:
    g = MolGraph()

    mol = molecule.get_mol(with_H=with_H, canonical_rank=canonical_rank)

    for atom in mol.GetAtoms():
        g.add_node(atom.GetIdx())

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        g.add_edge(start, end)

    for prop, dtype in molecule.get_property_names_with_type():
        g.set_property(prop, molecule.get_property(prop))

    # required for featurization
    g._mol = mol

    # make imutable_ish
    nx.freeze(g)
    return g


def mol_graph_from_mol(mol: Mol, *args, **kwargs) -> MolGraph:
    return mol_graph_from_molecule(Molecule(mol, *args, **kwargs))
