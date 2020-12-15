import bpy
from rdkit.Chem import AllChem

from molNet.featurizer.atom_featurizer import atom_formal_charge, atom_partial_charge, atom_mass, \
    atom_total_degree_one_hot, atom_degree_one_hot, atom_symbol_one_hot, atom_hybridization_one_hot, \
    atom_num_radical_electrons, atom_is_aromatic, atom_is_in_ring_size_3_to_20_one_hot, atom_symbol_hcnopsclbr_one_hot
from molNet.featurizer.featurizer import FeaturizerList
from molNet.mol.molecules import Molecule, MolGraph
from molNet.third_party.blender.mol import mol_to_model, create_molecule, default_atom_material
from molNet.third_party.blender.script import BlenderScript
from molNet.third_party.blender.tools import blender_function, blender_basic_script, blender_lambda_var, \
    blender_lambda_call, lambda_string
from molNet.third_party.blender.tools.animation import BlenderAnimationTracker, finish_animation, move_object, \
    move_objects, wait, new_camera, move_rotate_object
from molNet.third_party.blender.tools.geometry import create_text, set_parent
from molNet.third_party.blender.tools.groups import tree_copy
from molNet.third_party.blender.tools.materials import new_material, set_material
from molNet.third_party.blender.tools.scene import clear_scene, add_light, find_object_by_name
import numpy as np

SEED = 1163
DIST = 2
MOl_MOVE_HEIGHT = -0.1
OUT_DIST=300

def find_seed():
     z_error=10000000
     s=SEED
     for i in range(10000):
        mol = Molecule.from_smiles("c1ccccc1C(=O)OC")
        _mol = mol.get_mol()
        AllChem.EmbedMolecule(_mol,randomSeed=i)
        conf = _mol.GetConformer()
        ze = np.abs([list(conf.GetAtomPosition(i)) for i, a in enumerate(_mol.GetAtoms())])[:,2].std()

        if ze < z_error:
            z_error=ze
            s=i
        print(s)
     return s
#SEED=find_seed()



script = BlenderScript()
mol = Molecule.from_smiles("c1ccccc1C(=O)OC")
mol = MolGraph.from_molecule(mol)


atom_featurizer = FeaturizerList([
    atom_partial_charge,
    atom_mass,
    atom_hybridization_one_hot,
    atom_is_aromatic,
    atom_symbol_hcnopsclbr_one_hot,
    atom_formal_charge,
    atom_total_degree_one_hot,
    atom_degree_one_hot,
    atom_num_radical_electrons,
    atom_is_in_ring_size_3_to_20_one_hot,
],
    name="default_atom_featurizer")

mol.featurize(atom_featurizer=atom_featurizer)

def ini():
    script.append(clear_scene())
    script.append(add_light())

    base_mol = mol_to_model(mol, mol_name="base_mol", dist=DIST, varname="base_mol", seed=SEED)

    script.append(base_mol)

@blender_function(dependencies=[find_object_by_name],flat=True)
def reini():
    base_mol = find_object_by_name("base_mol",sub=True)[0]

script.append(reini())

@blender_function(dependencies=[tree_copy])
def copy_mol(mol, name):
    # bpy.context.view_layer.objects.active = mol["molecule"]
    return tree_copy(mol["molecule"], parent=None)


anim = BlenderAnimationTracker().set_varname("anim")

script.append(anim)

cam = new_camera(lens=64, location=(0, 0, 60)).set_varname("cam")
script.append(cam)


@blender_basic_script
def blink_atoms(mol, t=0.2):
    anim.go_to_second(1)

    for atom in mol["atoms"]:
        curMat = atom.active_material
        emitStr = curMat.node_tree.nodes['Mix Shader'].inputs[0]
        default_val = emitStr.default_value
        emitStr.keyframe_insert('default_value', frame=anim.current_frame)
        anim.run_seconds(t / 2)
        emitStr.default_value = 0.3
        emitStr.keyframe_insert('default_value', frame=anim.current_frame)
        anim.run_seconds(t / 2)
        emitStr.default_value = default_val
        emitStr.keyframe_insert('default_value', frame=anim.current_frame)
        # anim.run_seconds(0.5)


@blender_function(dependencies=[move_objects])
def move_molecule(mol, with_cam=False, x=0.0, y=0.0, z=0.0, time=0.001, delta=True, animator=None):
    _mol = mol["molecule"]
    if with_cam:
        cam = bpy.context.scene.camera

    if with_cam:
        move_objects([_mol, cam], x=x, y=y, z=z, delta=delta, animator=animator, time=time)
    else:
        move_objects([_mol], x=x, y=y, z=z, delta=delta, animator=animator, time=time)


sqre = int(np.sqrt(len(atom_featurizer)))
bsqre = int(np.ceil(np.sqrt(len(atom_featurizer))))

_mol = mol.get_mol()
conf = _mol.GetConformer()
coordds = np.array([list(conf.GetAtomPosition(i)) for i, a in enumerate(_mol.GetAtoms())]) * (DIST + 1) * 1.5
min_x, min_y, min_z = coordds.min(axis=0)
max_x, max_y, max_z = coordds.max(axis=0)
x_size = max_x - min_x
y_size = max_y - min_y
print(sqre,bsqre)


sqare_points_x = np.array([i for i in range(bsqre)])# * x_size
sqare_points_y = np.array([j for j in range(bsqre)])# * y_size
sqare_points_x = sqare_points_x - sqare_points_x.mean()
sqare_points_y = sqare_points_y - sqare_points_y.mean()

xv, yv = np.meshgrid(sqare_points_x,sqare_points_y)
xv, yv  = xv.flatten(), yv.flatten()
unfilled=[[xv[i],yv[i]] for i in range(len(xv))]
ordered=[]

#max_dist=max(x_size,y_size)*1.1

def next_min(p):
    ufa=np.array(unfilled)
    if len(ufa)<=1:
        return ufa[0]
    ufd = np.linalg.norm(ufa-p,axis=1)
    md=1
    min_indices = np.where(ufd <= md)[0]
    while len(min_indices) == 0:
        md=md*1.1
        min_indices = np.where(ufd <= md)[0]
    all_mins = ufa[min_indices]

    min_from_zero = np.linalg.norm(all_mins-np.array([0,0]),axis=1)
    nextp = all_mins[np.argmin(min_from_zero)]
    return nextp

p = np.array([0,0])
while len(unfilled)>0:
    p = next_min(p)
    unfilled.remove(p.tolist())
    ordered.append(p)

ordered = np.array(ordered)
ordered[:,0]=ordered[:,0]*x_size
ordered[:,1]=ordered[:,1]*y_size

print(ordered)

script.append(move_molecule(
    base_mol,
    x=ordered[0][0],
    y=ordered[0][1],
    z=MOl_MOVE_HEIGHT,
    delta=False,
    with_cam=True,
    animator=anim,
))

layer0_copyies = []


@blender_function(dependencies=[new_material])
def feature_material():
    mat = new_material()
    bsdf = mat.new_node()
    emission = mat.new_node(type="ShaderNodeEmission", name="Emission")
    color_node = mat.new_node(type="ShaderNodeRGB")
    mixer = mat.new_node(type="ShaderNodeMixShader", name="Mixer")

    color_node.set("Color", (1, 1, 1, 1))
    bsdf.set('Base Color', (0, 0, 0, 0))
    emission.set("Strength", 3)
    mixer.set("Fac", .5)

    mat.connect(color_node.output("Color"), emission.input("Color"))
    # mat.connect(color_node.output("Color"),bsdf.input("Base Color"))

    mat.connect(emission.output("Emission"), mixer.input(2))
    mat.connect(bsdf.output("BSDF"), mixer.input(1))

    mat.connect(mixer.output("Shader"), mat.material_output.input("Surface"))
    return mat


fm = feature_material().set_varname('fm')
script.append(fm)

featuremap = blender_lambda_var("featuremap", "np.array", [data['molNet_features'] for n, data in mol.nodes(data=True)],
                                dtype=float)
script.append(featuremap)


@blender_function(dependencies=[set_material])
def set_atom_mat(mol, mat, f):
    for child in mol.children:
        if child.name.startswith("atom_"):
            i = int(child.name.split("_")[1].split(".")[0])
            _, nm = set_material(child, mat.material, copy=True)
            emitStr = nm.node_tree.nodes['Mix Shader'].inputs[0]
            emitStr.default_value = featuremap[i, f]
    return
    for i, a in enumerate(mol['atoms']):
        _, nm = set_material(a, mat.material, copy=True)
        emitStr = nm.node_tree.nodes['Mix Shader'].inputs[0]
        emitStr.default_value = featuremap[i, f]

@blender_function(dependencies=[tree_copy])
def copy_mol(mol):
   return tree_copy(mol["molecule"])

tm = default_atom_material(color=(0, 0, 1, 1)).set_varname("tm")
script.append(tm)




def ini():
    for f in range(len(atom_featurizer)):
        #cm = mol_to_model(mol, mol_name="test_mol", dist=2, varname="layer0_mol{}".format(f), seed=SEED)
        cm = copy_mol(base_mol).set_varname("layer0_mol{}".format(f))

        script.append(cm)

        fd = create_text(atom_featurizer.describe_features()[f], "feature_desc{}".format(f), size=3)
        fd.set_varname("feature_desc{}".format(f))
        script.append(fd)

        script.append(set_material(fd, tm))

        #script.append(set_parent(fd, lambda_string(str(cm) + "['molecule']")))
        script.append(set_parent(fd, cm))

        move_text = move_object(fd, x=-x_size / 2 * 2 / 3, y=-y_size / 2, delta=True)
        script.append(move_text)

        script.append(set_atom_mat(cm, fm, f))

        layer0_copyies.append(cm)
        #move_to_hide = move_molecule(cm, x=sqare_points_x[f % sqre],
        #                             y=sqare_points_y[int(f / sqre)],
        #                             z=OUT_DIST, delta=False, with_cam=False,
        #                             animator=None
        #                             )
        move_to_hide = move_object(cm,
                                   #x=sqare_points_x[f % sqre],
                                   #y=sqare_points_y[int(f / sqre)],
                                   x=ordered[f][0],
                                   y=ordered[f][1],
                                   z=OUT_DIST,
                                   delta=False,
                                   animator=None
                                   )

        script.append(move_to_hide)

def ini():
    for f in range(len(atom_featurizer)):
        cm = copy_mol(base_mol).set_varname("layer0_mol{}".format(f))

def reini():
    for f in range(len(atom_featurizer)):
        cm = copy_mol(base_mol).set_varname("layer0_mol{}".format(f))
        script.append(cm)
        fd = create_text(atom_featurizer.describe_features()[f], "feature_desc{}".format(f), size=3)
        fd.set_varname("feature_desc{}".format(f))
        script.append(fd)

        script.append(set_material(fd, tm))

        script.append(set_parent(fd, cm))

        move_text = move_object(fd, x=-x_size / 2 * 2 / 3, y=-y_size / 2, delta=True)
        script.append(move_text)

        script.append(set_atom_mat(cm, fm, f))

        layer0_copyies.append(cm)


def start():
    for f in range(len(atom_featurizer)):
        #cm = mol_to_model(mol, mol_name="test_mol", dist=2, varname="layer0_mol{}".format(f), seed=SEED)
        cm = copy_mol(base_mol).set_varname("layer0_mol{}".format(f))

        script.append(cm)

        fd = create_text(atom_featurizer.describe_features()[f], "feature_desc{}".format(f), size=3)
        fd.set_varname("feature_desc{}".format(f))
        script.append(fd)

        script.append(set_material(fd, tm))

        #script.append(set_parent(fd, lambda_string(str(cm) + "['molecule']")))
        script.append(set_parent(fd, cm))

        move_text = move_object(fd, x=-x_size / 2 * 2 / 3, y=-y_size / 2, delta=True)
        script.append(move_text)

        script.append(set_atom_mat(cm, fm, f))

        layer0_copyies.append(cm)
        #move_to_hide = move_molecule(cm, x=sqare_points_x[f % sqre],
        #                             y=sqare_points_y[int(f / sqre)],
        #                             z=OUT_DIST, delta=False, with_cam=False,
        #                             animator=None
        #                             )
        move_to_hide = move_object(cm,
                                   #x=sqare_points_x[f % sqre],
                                   #y=sqare_points_y[int(f / sqre)],
                                   x=ordered[f][0],
                                   y=ordered[f][1],
                                   z=OUT_DIST,
                                   delta=False,
                                   animator=None
                                   )

        script.append(move_to_hide)
start()
def scene1():
    SHOW_MOLS = 32

    script.append(blink_atoms(base_mol))

    script.append(move_object(cam, delta=True, z=60, animator=anim, time=0.5))

    for f in range(SHOW_MOLS):
        script.append(move_molecule(
            base_mol,
#            x=sqare_points_x[f % sqre],
#            y=sqare_points_y[int(f / sqre)],
            x=ordered[f][0],
            y=ordered[f][1],
            z=MOl_MOVE_HEIGHT,
            delta=False,
            time=0.5,
            with_cam=True,
            animator=anim,
        ))
        script.append(wait(anim, 0.2))
        #move_to_show = move_molecule(layer0_copyies[f],
        #                             z=-OUT_DIST,
        #                             delta=True,
         #                            animator=anim,
         #                            time=0.00001
         #                            )
        move_to_show = move_object(layer0_copyies[f],
                                     z=-OUT_DIST,
                                     delta=True,
                                     animator=anim,
                                     time=0.00001
                                     )
        script.append(move_to_show)
        script.append(wait(anim, 0.3))

    script.append(move_molecule(
        base_mol,
        z=OUT_DIST,
        delta=True,
        time=0.00001,
        with_cam=False,
        animator=anim,
    ))

    script.append(move_object(cam, delta=False, z=250, animator=anim, time=0.5))
    script.append(wait(anim, 0.2))
    script.append(
        move_rotate_object(cam, x_move=110, y_move=-150, z_move=125, x_rot=56, y_rot=0, z_rot=42, delta_move=False,
                           delta_rot=False, time=1, animator=anim))


scene1()

script.append(finish_animation(anim))

with open("{}_out".format(__file__), "w+") as f:
    f.write(script.to_script(with_times=True))
