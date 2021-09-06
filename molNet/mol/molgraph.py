import copy
from typing import Dict

import networkx as nx
import numpy as np
from rdkit.Chem.rdchem import Mol

from molNet.featurizer._atom_featurizer import AtomFeaturizer
from molNet.featurizer._molecule_featurizer import MoleculeFeaturizer
from molNet.mol.molecule import Molecule, molecule_from_smiles


class BaseMolGraph(nx.DiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)
        self._properties = {}

    def get_property(self, name):
        return self._properties[name]

    def get_atom_features_dict(self):
        features = {}
        for k in self.nodes[0].keys():
            features[k] = self.get_atom_feature(k)
        return features

    def get_atom_feature(self, feature_name):
        feature = np.zeros(
            (len(self), *self.nodes[0][feature_name].shape),
            dtype=self.nodes[0][feature_name].dtype,
        )
        for node_id, node_data in self.nodes(data=True):
            feature[node_id] = node_data[feature_name]
        return feature

    @property
    def ege_array(self):
        return np.array(self.edges)

    def as_arrays(self) -> Dict:
        return {
            "size": len(self),
            "eges": self.ege_array,
            "node_features": self.get_atom_features_dict(),
            "graph_features": self._properties,
        }


class FrozenMolGraph(BaseMolGraph):
    def __init__(self, source: BaseMolGraph, **attr):
        super().__init__(**attr)
        self._properties = copy.deepcopy(source._properties)
        self.add_nodes_from(source.nodes(data=True))
        self.add_edges_from(source.edges)
        nx.freeze(self)


class MolGraph(BaseMolGraph):
    def __init__(self, **attr):
        super().__init__(**attr)
        self._mol = None

    def set_property(self, name, value):
        self._properties[name] = value

    def featurize_mol(
        self, mol_featurizer: MoleculeFeaturizer, name: str = None, as_y: bool = False
    ):
        if name is None:
            name = str(mol_featurizer)
        if as_y:
            name = "_y_" + name
        self.set_property(name, mol_featurizer(self.mol))

    def featurize_atoms(
        self, atom_featurizer: AtomFeaturizer, name: str = None, as_y: bool = False
    ):
        if name is None:
            name = str(atom_featurizer)
        if as_y:
            name = "_y_" + name
        for n in self.nodes:
            node = self.nodes[n]
            node[name] = atom_featurizer(self.mol.GetAtomWithIdx(n))

    def get_mol(self):
        return self._mol

    @property
    def mol(self):
        return self.get_mol()

    def freeze(self) -> FrozenMolGraph:
        return FrozenMolGraph(self)


def mol_graph_from_molecule(
    molecule: Molecule, with_H: bool = True, canonical_rank: bool = True
) -> MolGraph:
    g = MolGraph()

    mol = molecule.get_mol(
        with_H=with_H, canonical_rank=canonical_rank, with_properties=False
    )

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


def mol_graph_from_smiles(smiles: str, *args, **kwargs) -> MolGraph:
    return mol_graph_from_molecule(molecule_from_smiles(smiles, *args, **kwargs))


class MolgraphEqualsException(Exception):
    pass


def assert_molgraphs_data_equal(mg1: BaseMolGraph, mg2: BaseMolGraph):
    mg1d = mg1.as_arrays()
    mg2d = mg2.as_arrays()

    if not mg1d["size"] == mg2d["size"]:
        raise MolgraphEqualsException("size mismatch")

    if not np.allclose(mg1d["eges"], mg2d["eges"]):
        raise MolgraphEqualsException("edge mismatch")

    for n in ["node_features", "graph_features"]:
        d1, d2 = mg1d[n], mg2d[n]
        for cd1, cd2 in [(d1, d2), (d2, d1)]:
            for k in cd1.keys():
                if k not in cd2:
                    raise MolgraphEqualsException(
                        "feature missmatch('{}')".format(n + "," + k)
                    )
                if isinstance(cd1[k], np.ndarray):
                    if not np.allclose(cd1[k], cd2[k]):
                        raise MolgraphEqualsException(
                            "feature missmatch('{}')".format(n + "," + k)
                        )
                else:
                    if not cd1[k] == cd2[k]:
                        raise MolgraphEqualsException(
                            "feature missmatch('{}')".format(n + "," + k)
                        )


def molgraphs_data_equal(mg1: BaseMolGraph, mg2: BaseMolGraph):
    try:
        assert_molgraphs_data_equal(mg1, mg2)
    except MolgraphEqualsException:
        return False
    return True
