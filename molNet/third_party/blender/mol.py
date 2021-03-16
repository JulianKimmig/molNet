import inspect
import sys
import types

import bpy
import rdkit
from rdkit.Chem import AllChem

from blender_script_creator.geometry import (
    BlenderObject,
    create_plain_object,
    Sphere,
    Connection,
)
from blender_script_creator.materials import new_material, get_or_create_material
from blender_script_creator.script import blender_function, BlenderVariable
from molNet.mol.molecules import Molecule


@blender_function(dependencies=[new_material, get_or_create_material])
def default_atom_material(name, color):
    mat, new = get_or_create_material(name=name)

    bsdf, n = mat.get_or_create_node("principled bsdf")
    new = new or n
    emission, n = mat.get_or_create_node(type="ShaderNodeEmission", name="Emission")
    new = new or n
    color_node, n = mat.get_or_create_node(type="ShaderNodeRGB", name="colornode")
    new = new or n
    mixer, n = mat.get_or_create_node(type="ShaderNodeMixShader", name="Mixer")
    new = new or n

    bsdf.Base_Color = color
    color_node.Color = color

    emission.Strength = 1
    mixer.Fac = 0.5
    if new:
        emission.Color = color_node.Color
        # bsdf.Color = color_node.Color

        mixer.Shader_0 = emission.Emission
        mixer.Shader_1 = bsdf.BSDF

        mat.material_output.Surface = mixer.Shader_2

    return mat


@blender_function(dependencies=[default_atom_material])
def get_default_atom_map(atom, index=-1):
    am = {
        "H": {
            "dia": 0.5,
            "material": default_atom_material(
                name="default_Atom_H", color=(1, 1, 1, 1)
            ),
        },
        "C": {
            "dia": 1,
            "material": default_atom_material(
                name="default_Atom_C", color=(0.1, 0.1, 0.1, 1)
            ),
        },
        "O": {
            "dia": 0.95,
            "material": default_atom_material(
                name="default_Atom_O", color=(1, 0, 0, 1)
            ),
        },
        "N": {
            "dia": 0.95,
            "material": default_atom_material(
                name="default_Atom_N", color=(0, 0, 1, 1)
            ),
        },
        "Unknown": {
            "dia": 2,
            "material": default_atom_material(
                name="default_Atom_Unknown", color=(0.5, 0.5, 0.5, 1)
            ),
        },
    }
    return am.get(atom, am.get("Unknown"))


@blender_function(dependencies=[default_atom_material])
def get_bond_material():
    return default_atom_material("bond", color=(0.7, 0.7, 0.7, 1))


class BlenderAtom(Sphere):
    pass


class BlenderMol(BlenderObject):
    dependencies = BlenderObject.dependencies + [
        get_default_atom_map,
        Sphere,
        Connection,
        BlenderAtom,
        get_bond_material,
    ]

    def __init__(
        self,
        obj,
        name,
        atoms,
        coordinates,
        bonds,
        dist=0,
        bond_size=0.3,
        atom_map=get_default_atom_map,
        bond_material=get_bond_material,
    ):
        super().__init__(obj, name=name)
        self._atoms = []
        self._bonds = []
        self._check_atoms(atoms, atom_map, coordinates, dist)
        self._check_bonds(bonds, bond_size)

    @property
    def atoms(self):
        return [a for a in self._atoms]

    @property
    def bonds(self):
        return [a for a in self._bonds]

    def _check_atoms(self, atoms, atom_map, coordinates, dist):
        for i in range(len(atoms)):
            atom = atom_map(atoms[i], i)
            x, y, z = coordinates[i]
            x, y, z = x * (dist + 1), y * (dist + 1), z * (dist + 1)
            sphere = BlenderAtom.get_or_create_object(
                name="{}_atom_{}".format(self.name, i), dia=atom["dia"]
            )
            sphere.location = (x, y, z)
            sphere.parent = self
            sphere.material = atom["material"]
            self._atoms.append(sphere)

    def _check_bonds(self, bonds, bond_size):
        for i, b in enumerate(bonds):
            con = Connection.get_or_create_object(
                "{}_bond_{}".format(self.name, i),
                p1=self._atoms[b[0]].location,
                p2=self._atoms[b[1]].location,
                d=bond_size,
            )
            con.material = get_bond_material()
            con.parent = self
            self._bonds.append(con)


def mol_to_model(mol: Molecule, varname, dist=0, seed=None):
    _mol = mol.get_mol()
    AllChem.EmbedMolecule(_mol, randomSeed=-1 if seed is None else seed)
    conf = _mol.GetConformer()

    return BlenderVariable(
        varname,
        dict(
            atoms=[a.GetSymbol() for a in _mol.GetAtoms()],
            coordinates=[
                list(conf.GetAtomPosition(i)) for i, a in enumerate(_mol.GetAtoms())
            ],
            bonds=[(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in _mol.GetBonds()],
            dist=dist,
        ),
    )
