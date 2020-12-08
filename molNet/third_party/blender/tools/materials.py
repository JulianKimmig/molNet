import bpy

from molNet.third_party.blender.tools import blender_function


@blender_function()
class BlenderMaterialNode():
    def __init__(self, node):
        self._node = node
        self.input_index_map = {o.name: i for i, o in enumerate(self._node.inputs)}
        self.output_index_map = {o.name: i for i, o in enumerate(self._node.outputs)}

    def input(self,name):
        if isinstance(name,int):
            return self._node.inputs[name]
        if name not in self.input_index_map:
            return None
        return self._node.inputs[self.input_index_map[name]]

    def output(self,name):
        if isinstance(name,int):
            return self._node.outputs[name]
        if name not in self.output_index_map:
            return None
        return self._node.outputs[self.output_index_map[name]]

    def set(self, name, value):
        if self.input(name):
            self.input(name).default_value = value
        else:
            self.output(name).default_value = value
        return self


@blender_function(dependencies=[BlenderMaterialNode])
class BlenderMaterial():
    def __init__(self, material):
        self._material = material
        self._node_tree = material.node_tree
        self._nodes = self._node_tree.nodes
        self.material_output = BlenderMaterialNode(self._nodes.get("Material Output"))

    @property
    def material(self):
        return self._material

    def new_node(self, type="ShaderNodeBsdfPrincipled",name=None):
        nn = self._nodes.new(type)
        if name:
            nn.label = name
        return BlenderMaterialNode(nn)

    def connect(self,node1,node2):
        self._node_tree.links.new(node1,node2)

@blender_function(dependencies=[BlenderMaterial])
def new_material(name="material"):
    newmat = bpy.data.materials.new(name)
    newmat.use_nodes = True
    node_tree = newmat.node_tree
    nodes = node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    nodes.remove(bsdf)
    return BlenderMaterial(newmat)

@blender_function(dependencies=[])
def set_material(obj,mat,copy=False):
    if isinstance(mat,BlenderMaterial):
        mat = mat.material
    if copy:
        mat=mat.copy()

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    return obj,mat