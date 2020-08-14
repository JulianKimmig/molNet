import warnings

import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolfiles, rdmolops, rdchem

import numpy as np


def modification(func):
    def _wrapper(*args, **kwargs):
        args[0]._modify()
        func(*args, **kwargs)

    return _wrapper


def needs_valid(func):
    def _wrapper(*args, **kwargs):
        if not args[0].is_validated():
            args[0].validate()
        return func(*args, **kwargs)

    return _wrapper


class ConnectionPointException(Exception):
    pass


class ValidatingObject:
    def __init__(self):
        self._validated = False

    def _modify(self):
        self.set_invalid()

    def is_validated(self):
        return self._validated

    def set_invalid(self):
        self._validated = False

    def validate(self):
        self._validated = True
        return self._validated


class ConnectableGroup(ValidatingObject):
    colors = [
        (1, 0.6, 0.6),
        (0.5, 0.8, 0.5),
        (0.7, 0.7, 1),
        (0.6, 1, 1),
        (1, 0.8, 0.6),
        (0, 0.9, 0.8),

    ]
    fee_colors = []

    def __init__(self, mol, expected_connections, connection_indices=None, connection_names=None, connection_map=None,
                 name=None, color=None):
        super().__init__()
        self._mol = mol

        self._connection_names = connection_names
        self._connection_indices = connection_indices
        self._validated = False
        self._expected_connections = expected_connections
        self._connection_map = connection_map
        self._name = name
        if len(ConnectableGroup.fee_colors) == 0:
            ConnectableGroup.fee_colors = ConnectableGroup.colors.copy()
        if color is None:
            color = ConnectableGroup.fee_colors[0]
        if color in ConnectableGroup.fee_colors:
            ConnectableGroup.fee_colors.remove(color)
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
            if len(self._connection_names) != len(self._connection_indices):
                raise ConnectionPointException(
                    "expect connection_names to be of the same size as the connection indices({})".format(
                        len(self._connection_indices)))
        return super(ConnectableGroup, self).validate()

    def connection_name(self, index):
        return self._connection_names[index]

    def connection_name_from_idx(self, atom_idx):
        return self.connection_name(self._connection_indices.index(atom_idx))

    def _check_connections(self, expected_connections):

        n_radicals = Descriptors.NumRadicalElectrons(self._mol)
        if n_radicals < expected_connections:
            raise ConnectionPointException(
                "no enough connection points (radicals) in molecule ({} of {})".format(n_radicals,
                                                                                       expected_connections))

        if self._connection_indices is None:
            if n_radicals == expected_connections:
                self._connection_indices = []
                for atom in self._mol.GetAtoms():
                    self._connection_indices.extend([atom.GetIdx()] * atom.GetNumRadicalElectrons())
            else:
                raise ConnectionPointException(
                    "more radicals avaiable ({}) than needed ({}). Please specify conencting point via the 'connection_indices' argument. To get indices look at the mol object from ConnectableGroup.get_mol(with_numbers=True).".format(
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
                raise ConnectionPointException(
                    "Expect the atom with index '{}' to have at least {} radicals, but found only {}".format(
                        i, self._connection_indices.count(i), self._mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
                    ))

    @property
    def mol(self):
        return self._mol

    @property
    def connection_names(self):
        return self._connection_names

    def get_connection_indices(self, name=None):
        if name is None:
            return self._connection_indices
        return [self._connection_indices[i] for i, n in enumerate(self._connection_names) if n == name]

    connection_indices = property(get_connection_indices)

    def get_connectables_to(self, name):
        if self._connection_map is None:
            return None
        if name is None:
            return list(self._connection_map.keys())
        return list(set([key for key, clist in self._connection_map.items() if name in clist]))

    def get_mol(self, with_numbers=False, with_connection_indicator=True):
        mol = rdchem.EditableMol(self._mol)
        mol = mol.GetMol()

        if with_numbers:
            atoms = mol.GetNumAtoms()
            for idx in range(atoms):
                mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))

        if with_connection_indicator:
            mol = rdchem.EditableMol(mol)
            if not self._validated:
                self.validate()
            for i, idx in enumerate(self._connection_indices):
                add_atom = Chem.Atom(0)
                add_atom.SetProp('atomLabel',
                                 "R" + str(i) if self._connection_names[i] is None else self._connection_names[i])
                add_idx = mol.AddAtom(add_atom)
                mol.AddBond(add_idx, idx, rdchem.BondType.SINGLE)
            mol = mol.GetMol()
            for i, idx in enumerate(self._connection_indices):
                c_atom = mol.GetAtomWithIdx(idx)
                c_atom.SetNumRadicalElectrons(c_atom.GetNumRadicalElectrons() - 1)

        return mol


class TerminalGroup(ConnectableGroup):
    def __init__(self, *args, **kwargs):
        super(TerminalGroup, self).__init__(*args, **kwargs, expected_connections=1)


class RepatingUnit(ConnectableGroup):
    def __init__(self, *args, expected_connections=None, **kwargs):

        if expected_connections is None:
            if "connection_indices" in kwargs:
                connection_indices = kwargs["connection_indices"]
                if connection_indices is not None and len(connection_indices) > 1:
                    expected_connections = len(connection_indices)
        if expected_connections is None:
            expected_connections = 2
        super(RepatingUnit, self).__init__(*args, **kwargs, expected_connections=expected_connections)


class Polymer(ValidatingObject):
    def __init__(self):
        self._repeating_units = []
        self._starting_group: TerminalGroup = None
        self._mn = 0
        self._connection_map = {}
        self._validated = False

    def validate(self):
        expected_connections = set()
        if self._starting_group is not None:
            if not self._starting_group.is_validated():
                self._starting_group.validate()
            expected_connections.update(set(self._starting_group.connection_names))

        for ru in self._repeating_units:
            if ru["ratio"] <= 0:
                warnings.warn("ratio for '{}' is {}, which is not really expected")

            if not ru["repeating_unit"].is_validated():
                ru["repeating_unit"].validate()

            expected_connections.update(set(ru["repeating_unit"].connection_names))

        from_connections = set(self._connection_map.keys())
        to_connections = set()
        for to_con in self._connection_map.values():
            to_connections.update(set(to_con))

        for expected_connection in expected_connections:
            if expected_connection not in self._connection_map:
                self._connection_map[expected_connection] = []
        # if len(expected_connections-from_connections) > 0:
        #    warnings.warn("cannot create connection from '{}'".format(",".join(expected_connections-from_connections)))

        # if len(expected_connections-to_connections) > 0:
        #    warnings.warn("cannot create connection to '{}'".format(",".join(expected_connections-to_connections)))

        if len(expected_connections - (to_connections.union(from_connections))) > 0:
            warnings.warn(
                "cannot connect '{}'".format(",".join(expected_connections - (to_connections.union(from_connections)))))

        return super(Polymer, self).validate()

    def get_starting_group(self):
        return self._starting_group

    @modification
    def set_starting_group(self, starting_group):
        self._starting_group = starting_group

    starting_group = property(get_starting_group, set_starting_group)

    def get_mn(self):
        return self._mn

    @modification
    def set_mn(self, mn):
        self._mn = mn

    mn = property(get_mn, set_mn)

    def get_connection_map(self):
        return self._connection_map

    @modification
    def set_connection_map(self, connection_map):
        for source, targets in connection_map.copy().items():
            for target in targets:
                if target not in connection_map:
                    connection_map[target] = []
                if source not in connection_map[target]:
                    connection_map[target].append(source)
        self._connection_map = connection_map

    connection_map = property(get_connection_map, set_connection_map)

    @modification
    def add_repeating_unit(self, repeating_unit, ratio=1):
        repeating_unit.validate()
        if ratio < 0:
            raise ValueError("ratio cannot be smaller than 0")
        self._repeating_units.append({
            'ratio': ratio,
            'repeating_unit': repeating_unit,
        })

    @needs_valid
    def get_random_mol(self, g=None, highlight_units=False):
        data = {}
        if highlight_units:
            data['highlight_atoms'] = {}
            data['highlight_bonds'] = {}

        units = []
        if g is None:
            g = self.get_random_graph()
        for node in g.nodes:
            unit = g.nodes[node]['unit']
            units.append(unit)
        em = Chem.EditableMol(Chem.Mol())

        edge_at_idx = {}
        unit_atomidx = []
        free_connection_indices = []
        bond_atoms = []
        for unit in units:
            mol = unit.mol
            at_idxs = {}

            for i, atom in enumerate(mol.GetAtoms()):
                at_idx = em.AddAtom(atom)
                at_idxs[i] = at_idx;
                if highlight_units:
                    data['highlight_atoms'][at_idx] = unit.color

            for i, bond in enumerate(mol.GetBonds()):
                bond_id = em.AddBond(at_idxs[bond.GetBeginAtomIdx()], at_idxs[bond.GetEndAtomIdx()],
                                     bond.GetBondType()) - 1
                if highlight_units:
                    data['highlight_bonds'][bond_id] = unit.color

            free_connection_indices.append(unit.get_connection_indices().copy())
            unit_atomidx.append(at_idxs)

        for edge_ids in g.edges:
            edge = g.edges[edge_ids]

            start_unit = edge_ids[0]
            end_unit = edge_ids[1]

            start_atomx = unit_atomidx[start_unit]
            end_atomx = unit_atomidx[end_unit]

            start_free_connection_indices = free_connection_indices[start_unit]
            end_free_connection_indices = free_connection_indices[end_unit]

            from_indices = units[start_unit].get_connection_indices(edge['from_type'])
            to_indices = units[end_unit].get_connection_indices(edge['to_type'])

            possible_froms = list(set(start_free_connection_indices).intersection(set(from_indices)))
            possible_tos = list(set(end_free_connection_indices).intersection(set(to_indices)))

            _from = possible_froms[0]
            _to = possible_tos[-1]

            start_free_connection_indices.remove(_from)
            end_free_connection_indices.remove(_to)

            full_from = unit_atomidx[start_unit][_from]
            full_to = unit_atomidx[end_unit][_to]

            b = em.AddBond(full_from, full_to) - 1
            bond_atoms.append(full_from)
            bond_atoms.append(full_to)

        polymol = em.GetMol()
        for atid in bond_atoms:
            atom = polymol.GetAtomWithIdx(atid)
            atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() - 1)

        return polymol, data

    @needs_valid
    def get_random_graph(self):
        mass = 0

        mol_indices = np.arange(len(self._repeating_units))
        mol_weights = np.array([ru['ratio'] for ru in self._repeating_units])
        mol_weights = mol_weights / mol_weights.sum()

        g = nx.DiGraph()
        #

        free_connections = []
        n_free_connections = []

        connections_dict_name_idx = {}

        def add_unit(unit):
            g.add_node(len(g), unit=unit,
                       label=str(unit),
                       free_conenctions=unit.connection_names,
                       )

            fc = []
            for n in unit.connection_names:
                if n not in connections_dict_name_idx:
                    connections_dict_name_idx[n] = len(connections_dict_name_idx)
                fc.append(connections_dict_name_idx[n])
            free_connections.append(np.array(fc))
            n_free_connections.append(len(unit.connection_names))

        if self.starting_group is not None:
            add_unit(self.starting_group)
            mass += Descriptors.MolWt(self.starting_group.mol)

        while mass < self.mn:
            mol_id = np.random.choice(mol_indices, p=mol_weights)
            ru_o = self._repeating_units[mol_id]
            ru = ru_o['repeating_unit']
            add_unit(ru)
            if 'mass' not in ru_o:
                ru_o['mass'] = Descriptors.MolWt(ru.mol)
            mass += ru_o['mass']

        possible_connection_type_matrix = np.zeros((len(connections_dict_name_idx), len(connections_dict_name_idx)),dtype=bool)
        for ct, i in connections_dict_name_idx.items():
            for t in self.connection_map[ct]:
                possible_connection_type_matrix[i, connections_dict_name_idx[t]] = True
        n_free_connections=np.array(n_free_connections)
        connections_dict_idx_name = {x: y for y, x in connections_dict_name_idx.items()}
        node_indces = np.array(np.arange(len(g)))
        bond_indces = np.array(np.arange(possible_connection_type_matrix.shape[0]))
        nodes_bonds_matrix = np.zeros((node_indces.shape[0], possible_connection_type_matrix.shape[0]),dtype=int)
        for i, fc_set in enumerate(free_connections):
            for j in fc_set:
                nodes_bonds_matrix[i, j] += 1


        active_nodes=np.zeros(node_indces.shape[0],dtype=bool)

        active_nodes[0]=True


        while True:
            possible_active_nodes = node_indces[active_nodes]
            free_other_nodes = node_indces[n_free_connections>0]
            np.random.shuffle(possible_active_nodes)
            np.random.shuffle(free_other_nodes)

            found=False
            for pan in possible_active_nodes:
                source_bond_indices = bond_indces[nodes_bonds_matrix[pan]>0]
                for fon in free_other_nodes[free_other_nodes!=pan]:
                    #if pan == fon:
                    #    continue
                    for sbi in source_bond_indices:
                        sbc=nodes_bonds_matrix[pan,sbi]
                        if sbc<=0:
                            continue
                        for tbi in bond_indces[possible_connection_type_matrix[sbi]]:
                            tbc=nodes_bonds_matrix[fon,tbi]
                            if tbc<=0:
                                continue
                            found=True
                            #print(pan,fon,sbi,tbi)
                            nodes_bonds_matrix[pan,sbi]-=1
                            nodes_bonds_matrix[fon,tbi]-=1
                            n_free_connections[pan]-=1
                            n_free_connections[fon]-=1
                            active_nodes[fon]=True
                            active_nodes[pan] = n_free_connections[pan] > 0
                            active_nodes[fon] = n_free_connections[fon] > 0
                            e = g.add_edge(pan, fon,
                                           from_type=connections_dict_idx_name[sbi],
                                           to_type=connections_dict_idx_name[tbi],
                                           )
                            break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break

            if not found:
                break

        n_sg = len(list(nx.connected_components(g.to_undirected())))
        if n_sg > 1:
            warnings.warn("could not connect all units. {} subunits found".format(n_sg))
        return g

    def as_macrocycle(self,size=10)
        in_us = 0
        mol_indices = np.arange(len(self._repeating_units))
        mol_weights = np.array([ru['ratio'] for ru in self._repeating_units])
        mol_weights = mol_weights / mol_weights.sum()

        g = nx.DiGraph()

        next_connectable=[]

        next_connection_sources={}

        def add_unit(unit):
            global next_conenctable
            g.add_node(len(g), unit=unit,
                       label=str(unit),
                       free_conenctions=unit.connection_names,
                       )
            next_connectable=[]
            for n in unit.connection_names:
                next_connectable.extend(self.connection_map[n])

        closed=False

        mol_id = np.random.choice(mol_indices, p=mol_weights)
        ru_o = self._repeating_units[mol_id]
        add_unit(ru_o)

        while len(g) < size and not closed:
            mol_id = np.random.choice(mol_indices, p=mol_weights)
            ru_o = self._repeating_units[mol_id]
            if ru_o.connection_names in next_connectable:
                add_unit(ru_o)
                e = g.add_edge(len(g)-2, len(g)-1,
                               from_type=connections_dict_idx_name[sbi],
                               to_type=connections_dict_idx_name[tbi],
                               )
