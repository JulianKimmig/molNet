import matplotlib
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolfiles, rdmolops, rdchem

import numpy as np

class ConnectionPointException(Exception):
    pass

class ConnectableGroup():
    colors = [
        (1, 0.6, 0.6),
        (0.5,0.8,0.5),
        (0.7,0.7,1),
        (0.6,1,1),
        (1,0.8,0.6),
        (0,0.9,0.8),

    ]
    fee_colors = colors

    def __init__(self,mol,expected_connections,connection_indices=None,connection_names=None,connection_map=None,name=None,color=None):
        self._mol=mol

        self._connection_names=connection_names
        self._connection_indices = connection_indices
        self._validated=False
        self._expected_connections = expected_connections
        self._connection_map=connection_map
        self._name=name
        if color is None:
            color = self.fee_colors[0]
        if color in self.fee_colors:
            self.fee_colors.remove(color)
        if len(self.fee_colors) == 0:
            self.fee_colors = self.colors
        self.color = color

    def __str__(self):
        if self._name:
            return self._name

        return Chem.MolToSmiles(self._mol)

    def validate(self):
        self._check_connections(self._expected_connections)
        if self._connection_names is None:
            self._connection_names = [None for i in self._connection_indices]
        else:
            if len(self._connection_names)!= len(self._connection_indices):
                raise  ConnectionPointException(
                    "expect connection_names to be of the same size as the connection indices({})".format(len(self._connection_indices)))


    def connection_name(self,index):
        return self._connection_names[index]

    def connection_name_from_idx(self,atom_idx):
        return self.connection_name(self._connection_indices.index(atom_idx))

    def _check_connections(self,expected_connections):

        n_radicals = Descriptors.NumRadicalElectrons(self._mol)
        if n_radicals < expected_connections:
            raise ConnectionPointException(
                "no enough connection points (radicals) in molecule ({} of {})".format(n_radicals,
                                                                                       expected_connections))

        if self._connection_indices is None:
            if n_radicals == expected_connections:
                self._connection_indices = []
                for atom in self._mol.GetAtoms():
                    self._connection_indices.extend([atom.GetIdx()]*atom.GetNumRadicalElectrons())
            else:
                raise ConnectionPointException(
                    "more radicals avaiable ({}) than needed ({}). Please specify conencting point via the 'connection_indices' argument. To get indices look at ConnectableGroup.display(with_numbers=True).".format(
                        n_radicals, expected_connections)
                )


        if expected_connections > len(self._connection_indices):
            raise ConnectionPointException(
                "more connection points need ({}) than specified ({})".format(expected_connections, len(
                    self._connection_indices))
            )

        if expected_connections < len(self._connection_indices):
            raise ConnectionPointException(
                "more connection points specified ({}) than need ({})".format(
                    len(self._connection_indices),
                    expected_connections)
            )

        for i in set(self._connection_indices):
            if self._mol.GetAtomWithIdx(i).GetNumRadicalElectrons() < self._connection_indices.count(i):
                raise ConnectionPointException("Expect the atom with index '{}' to have at least {} radicals, but found only {}".format(
                    i,self._connection_indices.count(i),self._mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
                ))

    @property
    def mol(self):
        return self._mol

    @property
    def connection_names(self):
        return self._connection_names

    def get_connection_indices(self,name=None):
        if name is None:
            return self._connection_indices
        return[self._connection_indices[i] for i,n in enumerate(self._connection_names) if n==name]


    connection_indices = property(get_connection_indices)


    def get_connectables_to(self,name):
        if self._connection_map is None:
            return None
        if name is None:
            return list(self._connection_map.keys())
        return list(set([key for key,clist in self._connection_map.items() if name in clist]))

    def display(self,with_numbers=False,with_connection_indicator=True):
        mol = rdchem.EditableMol(self._mol)
        mol = mol.GetMol()

        if with_numbers:
            atoms = mol.GetNumAtoms()
            for idx in range( atoms ):
                mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )



        if with_connection_indicator:
            mol = rdchem.EditableMol(mol)
            if not self._validated:
                self.validate()
            for i,idx in enumerate(self._connection_indices):
                add_atom = Chem.Atom(0)
                add_atom.SetProp('atomLabel',"R"+str(i) if self._connection_names[i] is None else self._connection_names[i])
                add_idx = mol.AddAtom(add_atom)
                mol.AddBond(add_idx,idx,rdchem.BondType.SINGLE)
            mol = mol.GetMol()
            for i,idx in enumerate(self._connection_indices):
                c_atom = mol.GetAtomWithIdx(idx)
                c_atom.SetNumRadicalElectrons(c_atom.GetNumRadicalElectrons()-1)

        display(mol)