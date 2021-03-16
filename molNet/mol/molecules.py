import pickle
from io import StringIO, BytesIO

import networkx as nx
import numpy as np
import rdkit
import torch
import torch_geometric
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops, rdchem, AllChem
import matplotlib.pyplot as plt

import molNet.utils.base_classes as mnbc
from molNet.featurizer.atom_featurizer import default_atom_featurizer
from molNet.utils.identifier2smiles import name_to_smiles
from molNet.utils.mol.draw import mol_to_svg

DATATYPES_MAP = {
    "FLOAT": (np.floating, float),
    "INT": (np.integer, int),
    "STRING": (str, np.str),
    "NONE": (type(None)),
    "BOOL": (bool, np.bool_),
}
DATATYPES = type('', (), {})()
DATATYPES.STRING = "STRING"
DATATYPES.NONE = "NONE"
DATATYPES.INT = "INT"
DATATYPES.FLOAT = "FLOAT"
DATATYPES.BOOL = "BOOL"


class MolDataPropertyHolder:
    @staticmethod
    def get_dtype(obj):
        if isinstance(obj, np.ndarray):
            ta = np.array([1.1]).astype(obj.dtype)
            for key, val in DATATYPES_MAP.items():
                if isinstance(ta[0], val):
                    return key
        if isinstance(obj, DATATYPES_MAP[DATATYPES.STRING]):
            return DATATYPES.STRING
        try:
            dtypes = [MolDataPropertyHolder.get_dtype(sobj) for sobj in obj]
            if len(set(dtypes)) == 1:
                return dtypes[0]
            return DATATYPES.NONE
        except:
            pass

        for key, val in DATATYPES_MAP.items():
            if isinstance(obj, val):
                return key
        return DATATYPES.NONE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._properties = {k: {} for k in DATATYPES_MAP.keys()}
        self._names = []
        self._dtypes = []

    def set_property(self, name, value, dtype=None):
        if dtype is None:
            dtype = MolDataPropertyHolder.get_dtype(value)

        if dtype not in DATATYPES_MAP.keys():
            raise TypeError("unknown type '{}'".format(dtype))

        if name in self._names:
            or_index = self._names.index(name)
            or_key = self._dtypes[or_index]
            if or_key != dtype:
                del self._properties[or_key][name]
                self._dtypes[or_index] = dtype
        else:
            self._names.append(name)
            self._dtypes.append(dtype)
        self._properties[dtype][name] = value

    def get_property(self, name, with_dtype=False):
        index = self._names.index(name)
        dtype = self._dtypes[index]
        if with_dtype:
            return self._properties[dtype][name], dtype
        return self._properties[dtype][name]

    def has_property(self, name):
        return name in self._properties

    def get_property_names(self, with_dtype=False):
        if with_dtype:
            return list(zip(self._names.copy(), self._dtypes.copy()))
        else:
            return self._names.copy()


class SMILEError(Exception):
    pass


class Molecule(MolDataPropertyHolder, mnbc.ValidatingObject):

    def __init__(self, mol, name=""):
        super().__init__()
        self._mol = mol
        self.name = name
        self.smiles = Chem.MolToSmiles(mol)

    def __str__(self):
        _name = self.get_property("name")
        if _name is not None and len(_name) > 0:
            return _name

        return Chem.MolToSmiles(self._mol)

    @property
    def smiles(self):
        return self.get_property("smiles")

    @smiles.setter
    def smiles(self, smiles):
        to_s = Chem.MolToSmiles(self.mol)
        if smiles == to_s:
            self.set_property("smiles", smiles, dtype=DATATYPES.STRING)
            return
        
        conv_s = Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=False))
        if not to_s == conv_s:
            raise SMILEError("smiles dont match {} and {} as {}".format(
                to_s,
                smiles,
                conv_s
            ))
        self.set_property("smiles", smiles, dtype=DATATYPES.STRING)

    @property
    def mol(self):
        return self._mol

    @property
    def name(self):
        return self.get_property("name")

    @name.setter
    def name(self, name):
        self.set_property("name", name, dtype=DATATYPES.STRING)

    def get_mol(self, with_numbers=False, with_H=None) -> rdkit.Chem.Mol:
        """

        :rtype: rdkit.Chem.Mol
        """
        mol = rdchem.EditableMol(self._mol)
        mol = mol.GetMol()

        if with_H:
            mol = rdkit.Chem.AddHs(mol)
        elif with_H is None:
            pass
        else:
            mol = rdkit.Chem.RemoveHs(mol)

        if with_numbers:
            atoms = mol.GetNumAtoms()
            for idx in range(atoms):
                mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
        return mol

    def to_svg(self, size=(200, 200), svg_data=None, mol_data=None):
        if mol_data is None:
            mol_data = {}
        return mol_to_svg(mol=self.get_mol(**mol_data), size=size, svg_data=svg_data)

    def to_molgraph(self, *args, **kwargs):
        return MolGraph.from_molecule(self, *args, **kwargs)

    def _repr_svg_(self):
        return self.to_svg()

    def get_smiles(self):
        return Chem.MolToSmiles(self.mol)

    def as_dict(self):
        return {
            'properties': self._properties,
            **{'_mol_prob_' + str(key): val for key, val in self._mol.GetPropsAsDict()}
        }

    def get_random_smiles(self, max_smiles=1000):
        s = []
        mol = self.get_mol(with_H=False)

        possibilities = []
        for m in [mol, Chem.AddHs(mol), Chem.RemoveHs(mol)]:
            for allBondsExplicit in [False, True]:
                for allHsExplicit in [False, True]:
                    for kekuleSmiles in [False, True]:
                        for isomericSmiles in [True, False]:
                            for canonical in [True, False]:
                                for atom in m.GetAtoms():
                                    possibilities.append(dict(mol=m, canonical=canonical,
                                                              allHsExplicit=allHsExplicit,
                                                              rootedAtAtom=atom.GetIdx(),
                                                              allBondsExplicit=allBondsExplicit,
                                                              kekuleSmiles=kekuleSmiles,
                                                              isomericSmiles=isomericSmiles, ))

        indices = np.arange(len(possibilities))
        np.random.shuffle(indices)
        i = 0
        for idx in indices:
            try:
                ns = Chem.MolToSmiles(**possibilities[idx])
                if ns not in s:
                    i += 1
                    s.append(ns)
                    if i >= max_smiles:
                        return s[:max_smiles]
            except:
                pass
        return s[:max_smiles]

    @staticmethod
    def save(molecule, path, method="pickle"):
        with open(path, 'wb') as f:
            if method == "pickle":
                pickle.dump(molecule, f)
            else:
                method(molecule, f)

    @staticmethod
    def load(path, method="pickle"):
        obj = None
        with open(path, 'rb') as f:
            if method == "pickle":
                obj = pickle.load(f)
                if not isinstance(obj, Molecule):
                    raise ValueError("loaded object not of type Molecule")
            else:
                obj = method(f)
        return obj

    @classmethod
    def from_smiles(cls, mol_smile, *args, **kwargs):
        m = cls(Chem.MolFromSmiles(mol_smile, *args, **kwargs))
        return m

    @classmethod
    def from_name(cls, name, *args, **kwargs):
        ns = name_to_smiles(name)
        return cls.from_smiles(mol_smile=list(ns.keys())[0], name=name, *args, **kwargs)


def molecule_from_name(name, *args, **kwargs):
    return Molecule.from_name(name, *args, **kwargs)


class MolGraph(MolDataPropertyHolder, nx.DiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.mol_features = []
        self._mol = None
        self._molecule = None

    def featurize(self, atom_featurizer=None, name="molNet_features", molecule_featurizer=None):
        if atom_featurizer is None:
            atom_featurizer = default_atom_featurizer

        if molecule_featurizer is not None:
            self.mol_features = molecule_featurizer(self.mol)
        else:
            self.mol_features = []

        for n in self.nodes:
            node = self.nodes[n]
            node[name] = atom_featurizer(self.mol.GetAtomWithIdx(n))

        return np.array([data[name] for n, data in self.nodes(data=True)]), np.array(self.mol_features)

    def get_mol(self):
        return self.molecule.mol

    def get_molecule(self):
        if self._molecule is None:
            raise ValueError("no molecule")
        return self._molecule

    @property
    def mol(self):
        return self.get_mol()

    @property
    def molecule(self):
        return self.get_molecule()

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

        # g._mol = mol
        g._molecule = Molecule(mol, name=molecule.name)
        for prop, dtype in molecule.get_property_names(with_dtype=True):
            try:
                g._molecule.set_property(prop, molecule.get_property(prop), dtype=dtype)
            except SMILEError:
                pass
            g.set_property(prop, molecule.get_property(prop), dtype=dtype)
        return g

    def to_graph_input(self, node_feature_names=True, with_properties=True, y_properties=[], with_mol_graph=False,
                       with_mol=False, **add_kwargs):
        first_node = self.nodes[next(iter(self.nodes))]

        if node_feature_names is True:
            node_feature_names = list(first_node.keys())

        feature_length = 0
        for node_feature_name in node_feature_names:
            if node_feature_name not in first_node:
                raise ValueError("'{}' not available, please call featurize first".format(node_feature_name))
            feature_length += len(first_node[node_feature_name])

        node_features = np.zeros((self.number_of_nodes(), feature_length))
        for n in self.nodes:
            node = self.nodes[n]
            node_feats = np.zeros(feature_length)
            f = 0
            for node_feature_name in node_feature_names:
                fl = len(node[node_feature_name])
                node_feats[f:f + fl] = node[node_feature_name]
                f += fl
            node_features[n] = node_feats

        row, col = [], []
        for start, end in self.edges:
            row += [start, end]
            col += [end, start]

        edge_index = np.array([row, col])

        if with_properties is True:
            with_properties = self.get_property_names()
        if with_properties is False:
            with_properties = []

        if len(with_properties) > 0:
            for prop in y_properties:
                if prop in with_properties:
                    with_properties.remove(prop)

        graph_features = self.mol_features.copy()
        graph_features_titles = {0: "mol_features"}
        y = []
        y_titles = {}
        string_data = []
        string_data_titles = []

        for prop in y_properties:
            p, dtype = self.get_property(prop, with_dtype=True)
            if dtype in [DATATYPES.INT, DATATYPES.FLOAT, DATATYPES.BOOL]:
                new_feats = np.array([p], dtype=float)  # .flatten()
                y_titles[len(y)] = prop
                y.append(new_feats)
            else:
                raise TypeError("for y values the prob should be numerical but is '{}'".format(dtype))

        for prop in with_properties:
            p, dtype = self.get_property(prop, with_dtype=True)
            if dtype in [DATATYPES.INT, DATATYPES.FLOAT]:
                new_feats = np.array([p], dtype=float)  # .flatten()
                graph_features_titles[len(graph_features)] = prop
                graph_features.extend(new_feats)

            elif dtype == DATATYPES.STRING:
                string_data.append(p)
                string_data_titles.append(prop)
            else:
                add_kwargs[prop] = p

        if with_mol:
            add_kwargs["mol"] = self.mol

        if with_mol_graph:
            add_kwargs["mol_graph"] = self

        if len(string_data) > 0:
            add_kwargs['string_data_titles'] = string_data_titles,
            add_kwargs['string_data'] = string_data,

        red_y = np.squeeze(y)
        while len(red_y.shape) < 2:
            red_y = np.expand_dims(red_y, axis=-1)

        red_x = np.squeeze(node_features)
        while len(red_x.shape) < 2:
            red_x = np.expand_dims(red_x, axis=-1)

        #print(red_x.shape)
        data = torch_geometric.data.data.Data(
            x=torch.from_numpy(red_x).float(),
            edge_index=torch.from_numpy(edge_index).long(),
            graph_features_titles=graph_features_titles,
            graph_features=torch.from_numpy(np.array([graph_features]), ).float(),
            num_nodes=self.number_of_nodes(),
            y=torch.from_numpy(red_y).float(),
            **add_kwargs
        )
        return data

    def as_dict(self):
        return {
            'properties': self._properties,
            'edges': list(self.edges),
            'nodes': list(self.nodes),
            'graph_features': self.mol_features,
        }

    @staticmethod
    def save(molgraph, path, method="pickle"):
        with open(path, 'wb') as f:
            if method == "pickle":
                pickle.dump(molgraph, f)
            else:
                method(molgraph, f)

    @staticmethod
    def load(path, method="pickle"):
        obj = None
        with open(path, 'rb') as f:
            if method == "pickle":
                obj = pickle.load(f)
                if not isinstance(obj, MolGraph):
                    raise ValueError("loaded object not of type Molecule")
            else:
                obj = method(f)
        return obj

    def calc_position(self, norm=True):
        pos = None
        molecule = self.molecule
        
        if molecule:
            pos=molecule.calc_position(norm=norm)
    
        if pos is None:
            pos = nx.nx_pylab.spring_layout(self, iterations=5000,
                                            # scale=10,
                                            k=1 / (len(self) ** 2),
                                            pos=nx.nx_pylab.kamada_kawai_layout(
                                                self,
                                                pos=nx.nx_pylab.spring_layout(
                                                    self,
                                                    iterations=200,
                                                    k=1,
                                                    pos=nx.nx_pylab.circular_layout(self)
                                                ),
                                            )
                                            )
            if norm:
                pos_list = np.zeros((len(pos), 2))
                for i in range(pos_list.shape[0]):
                    pos_list[i] = pos[i]
                pos_list[:, 0] -= pos_list[:, 0].min()
                pos_list[:, 1] -= pos_list[:, 1].min()
                pos_list /= pos_list.max()

                pos = {i: pos_list[i] for i in range(pos_list.shape[0])}
        return pos

    def get_fig(self, labels=None, with_labels=True, **draw_options):
        pos = self.calc_position(norm=True)
        pos_list = np.array(list(pos.values()))
        fig = plt.figure(figsize=(
            0.1 + 4 * pos_list[:, 0].max(),
            0.1 + 4 * pos_list[:, 1].max(),
        ))
        # ax = fig.add_subplot(111)

        if with_labels:
            if isinstance(labels, (list, tuple)):
                labels = {i: labels[i] for i in self.nodes}

            if labels is None:
                labels = {i: self.mol.GetAtomWithIdx(i).GetSymbol() for i in self.nodes}

        draw_options = {**{'pos': pos, 'with_labels': len(self) < 100 and with_labels,
                           'node_size': 400 * pos_list[:, 1].max(), 'labels': labels, 'font_size': 10,
                           'arrowstyle': "-"},
                        **draw_options}

        nx.nx_pylab.draw(
            self,
            **draw_options
        )
        return fig

    def get_png(self, labels=None, with_labels=True):
        fig = self.get_fig(labels=labels, with_labels=with_labels)
        output = BytesIO()
        fig.savefig(output, format='png')
        plt.close()
        # plt.ion() # turn on interactive mode
        return output.getvalue()

    def _repr_png_(self):
        return self.get_png()
