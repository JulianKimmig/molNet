from blender_script_creator import bpy
from blender_script_creator.animation import BlenderAnimationTracker, Camera
from blender_script_creator.geometry import (
    get_or_create_object,
    create_plain_object,
    BlenderObject,
    BlenderText,
    Sphere,
    Connection,
)
from blender_script_creator.materials import (
    new_material,
    get_or_create_material,
    MixShader,
)
from blender_script_creator.modifier import Wave, Subsurface, Build
from blender_script_creator.nodes import Node, connect_node_sockets
from blender_script_creator.scene import delete_all_but, store_scene, LightSource
from blender_script_creator.script import (
    BlenderScript,
    blender_function,
    BlenderVariable,
)

import numpy as np

from molNet.featurizer.atom_featurizer import (
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
)
from molNet.featurizer.featurizer import FeaturizerList
from molNet.mol.molecules import Molecule, MolGraph
from molNet.third_party.blender.mol import (
    mol_to_model,
    BlenderMol,
    get_default_atom_map,
)

SEED = 1163
DIST = 2

script = BlenderScript()

mol = Molecule.from_smiles("c1ccccc1C(=O)OC")
mol = MolGraph.from_molecule(mol)

atom_featurizer = FeaturizerList(
    [
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
    name="default_atom_featurizer",
)

feats = mol.featurize(atom_featurizer=atom_featurizer)[0]
feats = feats - feats.min(axis=0)
feats = feats / feats.max(axis=0)
feats = np.nan_to_num(feats)

featuremap = BlenderVariable("featuremap", feats.tolist())
feature_names = BlenderVariable("feature_names", atom_featurizer.describe_features())

base_mol_data = mol_to_model(mol, dist=DIST, varname="base_mol_data", seed=SEED)


class BeamLayer(Connection):
    dependencies = Connection.dependencies + [Connection, Wave]

    def __init__(self, obj, name, beam):
        super().__init__(obj, name, p1=(0, 0, 0), p2=(0, 0, 10), d=1)
        self.beam = beam
        self._ini()
        self._reload_from_beam()

        # self.create(dia=dia,color=color,trans_dens=trans_dens,wave_height=wave_height)

    def _ini(self):
        self.parent = self.beam

        # self._beam_layer = Connection.get_or_create_object(self.name+"_beam_layer",
        #                                        p1=(0, 0, 0),
        #                                        p2=(0, 0, 10),
        #                                        d=1
        #                                        )
        # self._beam_layer.parent=self
        # self.add_modifier('subsurf', Subsurface(levels=3, render_levels=3))

        self._wave_mod = self.add_modifier("wave", Wave())

        self._wave_mod.use_normal = True

        self._wave_mod.narrowness = 0.5
        self._wave_mod.width = 3
        self._wave_mod.height = 0.5

        self._beam_layer_mat, new_mat = get_or_create_material(
            name=self.name + "material",
        )

        self._em, n = self._beam_layer_mat.get_or_create_node(
            name="emission", type="ShaderNodeEmission"
        )
        new_mat = new_mat or n
        self._rgbramp, n = self._beam_layer_mat.get_or_create_node(
            name="rgbramp", type="ShaderNodeValToRGB"
        )
        new_mat = new_mat or n
        self._mapping, n = self._beam_layer_mat.get_or_create_node(
            name="mapping", type="ShaderNodeMapping"
        )
        new_mat = new_mat or n
        mixer, n = self._beam_layer_mat.get_or_create_node(
            name="mixer1", type="ShaderNodeMixShader"
        )
        new_mat = new_mat or n
        trans, n = self._beam_layer_mat.get_or_create_node(
            name="transparent", type="ShaderNodeBsdfTransparent"
        )
        new_mat = new_mat or n
        noisetexture, n = self._beam_layer_mat.get_or_create_node(
            name="noisetexture", type="ShaderNodeTexNoise"
        )
        new_mat = new_mat or n
        textcoord, n = self._beam_layer_mat.get_or_create_node(
            name="textcoord", type="ShaderNodeTexCoord"
        )
        new_mat = new_mat or n

        if new_mat:
            self._beam_layer_mat.mat.blend_method = "BLEND"
            self._beam_layer_mat.mat.shadow_method = "NONE"
            mout = self._beam_layer_mat.get_node("Material Output")

            mixer.Shader_2 = mout.Surface
            mixer.Fac.value = 0.5

            self._em.Strength = 10
            mixer.Shader_1 = self._em.Emission

            mixer.Shader_0 = trans.BSDF

            self._rgbramp.node.color_ramp.elements[0].position = 0.5

            mixer.Fac = self._rgbramp.Color

            noisetexture.Scale = 1.5
            noisetexture.Detail = 16
            noisetexture.Roughness = 0.4
            noisetexture.Distortion = 0.5
            self._rgbramp.Fac = noisetexture.Color

            noisetexture.Vector = self._mapping.Vector_1

            self._mapping.Vector_0 = textcoord.Object

            self._mapping.Location = (0, 0, 0)
        self.material = self._beam_layer_mat

    def set_color(self, color):
        self._em.Color = color

    def set_dia(self, dia):
        self.dia = dia

    def wave_height(self, wvh):
        self._wave_mod.height = wvh

    def set_trans_dens(self, trans_dens):
        self._rgbramp.node.color_ramp.elements[0].position = trans_dens

    def _set_speed(self):
        l = self.length
        self._wave_mod.speed = self.beam.speed
        self._wave_mod.time_offset = (
            -(l - self._wave_mod.start_position_x) / self.beam.speed
        )

    def _reload_from_beam(self):
        self._set_length(self.beam.length)
        self._set_speed()

    @property
    def length(self):
        return np.linalg.norm(self.end)

    def _set_length(self, l):
        self.end = (l, 0, 0)
        l = self.length
        self._wave_mod.start_position_x = -l
        self._wave_mod.time_offset = (
            -(l - self._wave_mod.start_position_x) / self.beam.speed
        )

    def create_motion(self, animator):
        cf = animator.current_frame
        animator.go_to_frame(0)
        animator.change_node_value(
            self._mapping.Location, value=(0, 0, 0), interpolation="LINEAR"
        )
        animator.change_node_value(
            self._mapping.Rotation, value=(0, 0, 0), interpolation="LINEAR"
        )
        animator.go_to_frame(animator.max_frame)
        animator.change_node_value(
            self._mapping.Location,
            value=(
                -animator.max_frame * self._wave_mod.speed,
                0,
                0,
            ),
            interpolation="LINEAR",
        )
        animator.change_node_value(
            self._mapping.Rotation,
            value=(
                animator.max_frame * 0.1,
                0,
                0,
            ),
            interpolation="LINEAR",
        )
        animator.go_to_frame(cf)


class Beam(BlenderObject):
    dependencies = BlenderObject.dependencies + [
        BeamLayer,
    ]

    def __init__(
        self, obj, name, start=(0, 0, 0), end=(0, 0, 50), speed=0.25, forcelength=None
    ):
        self.forcelength = forcelength
        self._layer = []
        super().__init__(obj, name)
        self.speed = speed
        self.name = name
        self._end = np.array(end)
        self._start = np.array(start)
        self.end = np.array(end)
        self.start = np.array(start)

    def add_layer(
        self, dia=1.0, color=(0, 0.7, 1, 1), trans_dens=0.52, wave_height=0.5
    ):
        l = BeamLayer.get_or_create_object(
            beam=self,
            name="{}_layer_{}".format(self.name, len(self._layer)),
            # trans_dens=trans_dens,
        )
        l.wave_height(wave_height)
        l.set_trans_dens(trans_dens)
        l.set_dia(dia)
        l.set_color(color)
        self._layer.append(l)

    def create_motion(self, animator):
        for layer in self._layer:
            layer.create_motion(animator)

    @property
    def length(self):
        return (
            np.linalg.norm(self.end - self.start)
            if self.forcelength is None
            else self.forcelength
        )

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @start.setter
    def start(self, loc):
        loc = np.array(loc)
        # for lay in self._layer:
        #    lay.length = np.linalg.norm(self.end - loc) if self.forcelength is None else self.forcelength
        self.set_location(*loc)
        self._start = loc
        self.set_rotation(*self.calc_rot())

    @end.setter
    def end(self, loc):
        loc = np.array(loc)
        for lay in self._layer:
            lay._set_length(
                np.linalg.norm(self.start - loc)
                if self.forcelength is None
                else self.forcelength
            )
        self._end = loc
        self.set_rotation(*self.calc_rot())

    def calc_rot(self):
        d_vec = self.end - self.start

        def rotation_matrix_from_vectors(vec1, vec2):
            a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
                vec2 / np.linalg.norm(vec2)
            ).reshape(3)
            v = np.cross(a, b)
            c = np.dot(a, b)
            s = np.linalg.norm(v)

            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
            return np.nan_to_num(rotation_matrix)

        a = d_vec
        b = np.array([np.linalg.norm(a), 0, 0])
        a, b = (a / np.linalg.norm(a)).reshape(3), (b / np.linalg.norm(b)).reshape(3)

        if np.all(a == b):
            return self.set_rotation(0, 0, 0)
        if np.all(a == -b):
            return self.set_rotation(0, 0, 180)

        R = rotation_matrix_from_vectors(a, b)
        Rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        Ry = -np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
        Rz = -np.degrees(np.arctan2(R[1, 0], R[0, 0]))

        return np.array([Rx, Ry, Rz])


class ThreePointBeam(BlenderObject):
    dependencies = BlenderObject.dependencies + [Sphere, Connection, Beam, Build]

    def __init__(
        self,
        obj,
        name,
        center,
        p1,
        p2,
        p3,
        ball_height=3,
        base_color=(0, 0, 1, 1),
        max_length=None,
    ):
        super().__init__(obj, name)
        self._animation_ranges = []
        if isinstance(center, BlenderObject):
            center = center.world_location
        if isinstance(p1, BlenderObject):
            p1 = np.array(p1.world_location) - center
        if isinstance(p2, BlenderObject):
            p2 = np.array(p2.world_location) - center
        if isinstance(p3, BlenderObject):
            p3 = np.array(p3.world_location) - center

        self._target = np.array([0, 0, 100])
        self.center_ball = Sphere.get_or_create_object(
            "{}_centerball".format(name), dia=1
        )
        self.center_ball.obj.scale = (1, 1, 1)
        self.center_ball.parent = self
        self.center_ball.location = [0, 0, ball_height]

        self.center_beam = Connection.get_or_create_object(
            "{}_centerbeam".format(name),
            p1=np.array([0, 0, 0]),
            p2=self.center_ball.location,
            d=0.1,
            resolution=50,
        )

        self.center_beam.parent = self
        self.p1_beam = Connection.get_or_create_object(
            "{}_p1_beam".format(name),
            p1=p1,
            p2=self.center_ball.location,
            d=0.1,
            resolution=50,
        )

        self.p1_beam.parent = self
        self.p2_beam = Connection.get_or_create_object(
            "{}_p2_beam".format(name),
            p1=p2,
            p2=self.center_ball.location,
            d=0.1,
            resolution=50,
        )

        self.p2_beam.parent = self
        self.p3_beam = Connection.get_or_create_object(
            "{}_p3_beam".format(name),
            p1=p3,
            p2=self.center_ball.location,
            d=0.1,
            resolution=50,
        )
        self.p3_beam.parent = self

        material, new_mat = get_or_create_material(name="beam_{}".format(base_color))
        emission, n = material.get_or_create_node(
            type="ShaderNodeEmission", name="Emission"
        )
        new_mat = new_mat or n
        color_node, n = material.get_or_create_node(
            type="ShaderNodeRGB", name="colornode"
        )
        new_mat = new_mat or n
        if new_mat:
            color_node.Color = base_color
            emission.Color = color_node.Color
            material.material_output.Surface = emission.Emission
        emission.Strength = 20

        self.center_ball.material = material
        self.center_beam.material = material
        self.p1_beam.material = material
        self.p2_beam.material = material
        self.p3_beam.material = material

        self.main_beam = Beam.get_or_create_object(
            "{}_main_beam".format(name),
            start=self.center_ball.location,
            end=(0, 0, 100),
            speed=0.25,
            forcelength=max_length,
        )
        self.main_beam.parent = self
        base_color = np.array(base_color, dtype=float)
        self.main_beam.add_layer(color=base_color)

        self.main_beam.add_layer(
            dia=0.4, color=(1, 1, 1, 1), trans_dens=0.48, wave_height=0.2
        )

        self.main_beam.add_layer(
            dia=1.75,
            color=base_color / 4
            + np.array([base_color[1], base_color[2], base_color[0], 8 * base_color[3]])
            / 8,
            trans_dens=0.56,
        )

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        self._target = target
        self._retarget()

    def _retarget(self):
        target = self._target
        if isinstance(target, BlenderObject):
            target = target.world_location
        target = target - self.world_location  # obj.matrix_world.to_translation()
        self.main_beam.end = target
        return np.linalg.norm(self.main_beam.end - (self.main_beam.start))

    def _stop_move(self, animator, outdist):

        animator.scale_object(self.main_beam, 0, 0, 0.0, time=None, reverse=True)

        animator.move_object(
            self.center_ball, 0, 0, outdist, delta=True, time=None, reverse=True
        )
        animator.move_object(
            self.center_beam, 0, 0, outdist, delta=True, time=None, reverse=True
        )
        animator.move_object(
            self.p1_beam, 0, 0, outdist, delta=True, time=None, reverse=True
        )
        animator.move_object(
            self.p2_beam, 0, 0, outdist, delta=True, time=None, reverse=True
        )
        animator.move_object(
            self.p3_beam, 0, 0, outdist, delta=True, time=None, reverse=True
        )
        animator.move_object(
            self.main_beam, 0, 0, outdist, delta=True, time=None, reverse=True
        )

        for l in self.main_beam._layer:
            animator.bevel_start_end(l, 0, 0, time=None, reverse=True)

        animator.bevel_start_end(self.center_beam, 0, 0, time=None, reverse=True)
        animator.bevel_start_end(self.p2_beam, 0, 0, time=None, reverse=True)
        animator.bevel_start_end(self.p3_beam, 0, 0, time=None, reverse=True)
        animator.bevel_start_end(self.p1_beam, 0, 0, time=None, reverse=True)

        animator.scale_object(self.center_ball, 0.1, 0.1, 0.0, time=None, reverse=True)
        animator.scale_object(self.center_beam, 0.1, 0.1, 0.0, time=None, reverse=True)
        animator.scale_object(self.p1_beam, 0.1, 0.1, 0.0, time=None, reverse=True)
        animator.scale_object(self.p2_beam, 0.1, 0.1, 0.0, time=None, reverse=True)
        animator.scale_object(self.p3_beam, 0.1, 0.1, 0.0, time=None, reverse=True)

    def in_animation(self, animator):
        cf = animator.current_frame
        for ar in self._animation_ranges:
            if cf >= ar[0] and cf < ar[1]:
                return True
        return False

    def animate(
        self, animator: BlenderAnimationTracker, t0=0.2, t1=1, t2=0.5, outdist=1000
    ):
        cf = animator.current_frame

        targetlength = self._retarget()
        length_fac = targetlength / self.main_beam.length

        target_rot = self.main_beam.calc_rot()
        animator.rotate_object(self.main_beam, *target_rot, time=None, reverse=True)

        if len(self._animation_ranges) == 0:
            animator.register_finish_hook(self.main_beam.create_motion)
            animator.go_to_frame(0)
            self._stop_move(animator, outdist)

            animator.go_to_frame(cf)

        for ar in self._animation_ranges:
            if cf >= ar[0] and cf < ar[1]:
                raise ValueError("CURRENTLY in an ANIMATION ({},{})".format(cf, ar))

        self._animation_ranges.append([cf, 10 ** 8])
        animator.run_frames(1)
        animator.move_object(
            self.center_beam, 0, 0, -outdist, delta=True, time=None, reverse=True
        )
        animator.move_object(
            self.p1_beam, 0, 0, -outdist, delta=True, time=None, reverse=True
        )
        animator.move_object(
            self.p2_beam, 0, 0, -outdist, delta=True, time=None, reverse=True
        )
        animator.move_object(
            self.p3_beam, 0, 0, -outdist, delta=True, time=None, reverse=True
        )
        animator.move_object(
            self.main_beam, 0, 0, -outdist, delta=True, time=None, reverse=True
        )

        animator.scale_object(self.center_beam, 1, 1, 1, time=None, reverse=True)
        animator.bevel_start_end(self.center_beam, 0, 1, time=t0)

        animator.move_object(
            self.center_ball, 0, 0, -outdist, delta=True, time=None, reverse=True
        )

        animator.scale_object(self.p1_beam, 1, 1, 1, time=None, reverse=True)
        animator.scale_object(self.p2_beam, 1, 1, 1, time=None, reverse=True)
        animator.scale_object(self.p3_beam, 1, 1, 1, time=None, reverse=True)

        animator.scale_object(self.center_ball, 1, 1, 1, time=t1, reverse=True)
        animator.bevel_start_end(self.p1_beam, 0, 1, time=t1 * 0.9, reverse=True)
        animator.bevel_start_end(self.p2_beam, 0, 1, time=t1 * 0.9, reverse=True)
        animator.bevel_start_end(self.p3_beam, 0, 1, time=t1 * 0.9)

        animator.scale_object(self.main_beam, 1, 0, 0, time=None, reverse=True)

        for l in self.main_beam._layer:
            animator.bevel_start_end(l, 0, length_fac, time=t2, reverse=True)

        # animator.scale_object(self.main_beam, 1, 0.1, 0.1, time=0.1)

        animator.scale_object(self.main_beam, 1, 0.5, 0.5, time=t2)

        # animator.scale_object(self.center_ball,1,1,1,time=t02,reverse=False)
        # animator.scale_object(self.center_ball,1,1,1,time=t02,reverse=False)

    def animate_stop(self, animator, t=2, outdist=1000):
        targetlength = self._retarget()
        length_fac = targetlength / self.main_beam.length
        cf = animator.current_frame
        # main_beam_unbuilds = [l.add_modifier('unbuild_{}'.format(cf), Build(use_reverse=True,use_random_order=True))
        #                           for l in self.main_beam._layer + [self.p1_beam,self.p2_beam,self.p3_beam,self.center_beam,self.center_ball]]

        # for main_beam_unbuild in main_beam_unbuilds:
        #    main_beam_unbuild.frame_start=animator.current_frame
        #     main_beam_unbuild.frame_duration=animator.seconds_to_frames(t)
        for l in [self.p1_beam, self.p2_beam, self.p3_beam, self.center_beam]:
            animator.bevel_start_end(l, 1, 1, time=t, reverse=True)
        for l in self.main_beam._layer:
            animator.bevel_start_end(l, length_fac, length_fac, time=t, reverse=True)
        animator.scale_object(self.center_ball, 0, 0, 0, time=t, reverse=True)

        animator.scale_object(self.main_beam, 1, 0, 0, time=t)
        target_rot = self.main_beam.calc_rot()
        animator.rotate_object(self.main_beam, *target_rot, time=None, reverse=True)

        self._animation_ranges[-1][1] = animator.current_frame
        animator.run_frames(1)
        self._stop_move(animator, outdist)
        animator.run_frames(1)


@blender_function(
    dependencies=[
        Node,
        Camera,
        connect_node_sockets,
        Wave,
        BlenderAnimationTracker,
        delete_all_but,
        get_or_create_material,
        Subsurface,
        BlenderObject,
        base_mol_data,
        BlenderMol,
        featuremap,
        feature_names,
        BlenderText,
        LightSource,
        ThreePointBeam,
    ]
)
def mol_main():
    # scene
    RENDER_EDIT = False
    DIST = 2
    MOl_MOVE_HEIGHT = -0.1
    OUT_DIST = 300
    LAYER_HEIGHT = 50

    LAYER_SIZES = [len(feature_names), 10, 4]
    # LAYER_SIZES=[4,1]
    LASER_MULTIPLIER_FIRST_LAYER = 3
    LASER_MULTIPLIER_SECOND_LAYER = 1
    MIN_SHOOTING_TIME = 0.05
    DEFAULT_SHOOTING_TIME = 0.3

    FEATS_PER_OUT = 2
    FEATBALLDIA = 3

    if RENDER_EDIT:
        bpy.context.scene.eevee.use_gtao = True
        bpy.context.scene.eevee.gtao_distance = 1
        bpy.context.scene.eevee.gtao_factor = 3

        bpy.context.scene.eevee.use_bloom = True
        bpy.context.scene.eevee.bloom_threshold = 0.1
        bpy.context.scene.eevee.use_ssr = True

        # bpy.context.scene.view_settings.look = 'Very High Contrast'

    bpy.context.scene.use_nodes = True
    ntree = bpy.context.scene.node_tree
    for currentNode in ntree.nodes:
        ntree.nodes.remove(currentNode)

    tree = bpy.context.scene.node_tree
    rlayer_node = Node(tree.nodes.new("CompositorNodeRLayers"), tree)
    composite_node = Node(tree.nodes.new("CompositorNodeComposite"), tree)
    node_viewer_node = Node(tree.nodes.new("CompositorNodeViewer"), tree)
    glare_node = Node(tree.nodes.new("CompositorNodeGlare"), tree)
    connect_node_sockets(tree, rlayer_node.Image, glare_node.Image_0)

    connect_node_sockets(tree, glare_node.Image_1, composite_node.Image)
    connect_node_sockets(tree, glare_node.Image_1, node_viewer_node.Image)

    glare_node.node.glare_type = "FOG_GLOW"
    glare_node.node.threshold = 0.5

    # calc positions

    sqre = int(np.sqrt(max(LAYER_SIZES)))
    bsqre = int(np.ceil(np.sqrt(max(LAYER_SIZES))))

    coordds = np.array(base_mol_data["coordinates"]) * (DIST + 1) * 1.5
    min_x, min_y, min_z = coordds.min(axis=0)
    max_x, max_y, max_z = coordds.max(axis=0)
    x_size = max_x - min_x
    y_size = max_y - min_y

    sqare_points_x = np.array([i for i in range(bsqre)])  # * x_size
    sqare_points_y = np.array([j for j in range(bsqre)])  # * y_size
    sqare_points_x = sqare_points_x - sqare_points_x.mean()
    sqare_points_y = sqare_points_y - sqare_points_y.mean()

    xv, yv = np.meshgrid(sqare_points_x, sqare_points_y)
    xv, yv = xv.flatten(), yv.flatten()
    unfilled = [[xv[i], yv[i]] for i in range(len(xv))]
    ordered = []

    # max_dist=max(x_size,y_size)*1.1

    def next_min(p):
        ufa = np.array(unfilled)
        if len(ufa) <= 1:
            return ufa[0]
        ufd = np.linalg.norm(ufa - p, axis=1)
        md = 1
        min_indices = np.where(ufd <= md)[0]
        while len(min_indices) == 0:
            md = md * 1.1
            min_indices = np.where(ufd <= md)[0]
        all_mins = ufa[min_indices]

        min_from_zero = np.linalg.norm(all_mins - np.array([0, 0]), axis=1)
        nextp = all_mins[np.argmin(min_from_zero)]
        return nextp

    p = np.array([0, 0])
    while len(unfilled) > 0:
        p = next_min(p)
        unfilled.remove(p.tolist())
        ordered.append(p)

    ordered = np.array(ordered)
    ordered[:, 0] = ordered[:, 0] * x_size
    ordered[:, 1] = ordered[:, 1] * y_size

    # objects

    base_mol = BlenderMol.get_or_create_object("base_mol", **base_mol_data)
    base_mol.location = (0, 0, 0)
    for atom in base_mol.atoms:
        atom.material = atom.material.copy()

    layer_molecules = [[None] * s for s in LAYER_SIZES]

    def get_feat_mat(strength=1):
        strength = int(strength * 100) / 100
        feature_material, new = get_or_create_material(
            name="feat_mat_{}".format(strength)
        )

        emission, n = feature_material.get_or_create_node(
            type="ShaderNodeEmission", name="Emission"
        )
        new = new or n
        bsdf, n = feature_material.get_or_create_node("principled bsdf")
        new = new or n
        color_node, n = feature_material.get_or_create_node(
            type="ShaderNodeRGB", name="colornode"
        )
        new = new or n
        mixer, n = feature_material.get_or_create_node(
            type="ShaderNodeMixShader", name="Mixer"
        )
        new = new or n

        if new:
            bsdf.Base_Color = (0.1, 0, 0.2, 1)
            color_node.Color = (0.5, 0.1, 0, 1)

            mixer.Fac = 0.5

            emission.Color = color_node.Color
            # bsdf.Color = color_node.Color

            mixer.Shader_0 = emission.Emission
            mixer.Shader_1 = bsdf.BSDF

            feature_material.material_output.Surface = mixer.Shader_2
        emission.Strength = strength
        return feature_material

    def feature_atom_map(atom, index, molindex=None):
        data = get_default_atom_map(atom, index)
        if molindex is None:
            f = np.random.random() ** 2
        else:
            f = featuremap[index][molindex]
        data["material"] = get_feat_mat(f * 10)
        return data

    for l in range(len(layer_molecules)):
        for i in range(len(layer_molecules[l])):
            if l == 0:
                m = BlenderMol.get_or_create_object(
                    "mol_layer_{}_{}".format(l, i),
                    atom_map=lambda atom, index: feature_atom_map(atom, index, i),
                    **base_mol_data
                )
            else:
                m = BlenderMol.get_or_create_object(
                    "mol_layer_{}_{}".format(l, i),
                    atom_map=feature_atom_map,
                    **base_mol_data
                )
            layer_molecules[l][i] = m
            m.location = (ordered[i][0], ordered[i][1], 0)

    for i in range(len(layer_molecules[0])):
        t = BlenderText.get_or_create_object(
            "l0_feature_desc_{}".format(i),
            text=feature_names[i],
            size=3,
            align_y="BOTTOM",
        )
        t.material = get_default_atom_map("N", 0)["material"]
        t.parent = layer_molecules[0][i]
        t.location = (0, -y_size / 2, 0)

    sun = LightSource.get_or_create_object("sun", type="SUN", energy=10)
    sun.location = (0, 0, OUT_DIST)
    cam = Camera.get_or_create_object("default_cam", lens=64, location=(0, 0, 60))
    # layer1_centermol_cam = Camera.get_or_create_object("layer1_centermol_cam", lens=64, location=(0, 0, 60))

    cams = [
        cam,
        # layer1_centermol_cam
    ]

    first_beam = ThreePointBeam.get_or_create_object(
        "first_beam",
        center=layer_molecules[0][10].atoms[16],
        p1=layer_molecules[0][10].atoms[17],
        p2=layer_molecules[0][10].atoms[15],
        p3=layer_molecules[0][10].atoms[7],
        max_length=LAYER_HEIGHT * 5,
    )
    first_beam.location = layer_molecules[0][10].atoms[16].world_location
    first_beam.target = layer_molecules[1][0].atoms[16]

    first_layer_lasers = []
    for i in range(len(layer_molecules[0][0].atoms)):
        connected = []
        for b in base_mol_data["bonds"]:
            if b[0] == i:
                connected.append(b[1])
            if b[1] == i:
                connected.append(b[0])

        connected = np.array([connected, connected, connected]).flatten()
        c = np.random.random(4)
        c[:3] = c[:3] / (3 * c[:3].sum())
        c[3] = 1
        for j in range(LASER_MULTIPLIER_FIRST_LAYER):
            fll = ThreePointBeam.get_or_create_object(
                "first_layer_laser_{}_{}".format(i, j),
                center=layer_molecules[0][0].atoms[i],
                p1=layer_molecules[0][0].atoms[connected[0]],
                p2=layer_molecules[0][0].atoms[connected[1]],
                p3=layer_molecules[0][0].atoms[connected[2]],
                max_length=LAYER_HEIGHT * 3,
                base_color=c,
            )
            fll.atom_n = i
            fll.target = layer_molecules[1][0].atoms[i]
            first_layer_lasers.append(fll)

    second_layer_lasers = []
    for i in range(len(layer_molecules[1][0].atoms)):
        connected = []
        for b in base_mol_data["bonds"]:
            if b[0] == i:
                connected.append(b[1])
            if b[1] == i:
                connected.append(b[0])

        connected = np.array([connected, connected, connected]).flatten()
        c = np.random.random(4)
        c[:3] = c[:3] / (3 * c[:3].sum())
        c[3] = 1
        for j in range(LASER_MULTIPLIER_SECOND_LAYER):
            fll = ThreePointBeam.get_or_create_object(
                "second_layer_laser_{}_{}".format(i, j),
                center=layer_molecules[1][0].atoms[i],
                p1=layer_molecules[1][0].atoms[connected[0]],
                p2=layer_molecules[1][0].atoms[connected[1]],
                p3=layer_molecules[1][0].atoms[connected[2]],
                max_length=LAYER_HEIGHT * 3,
                base_color=c,
            )
            fll.atom_n = i
            fll.target = layer_molecules[2][0].atoms[i]
            second_layer_lasers.append(fll)

    featballs = []
    n_balls = LAYER_SIZES[-1] * FEATS_PER_OUT
    ballsx = np.arange(n_balls) * FEATBALLDIA * 2.5
    ballsx = ballsx - ballsx.mean()
    b = 0
    for i in range(LAYER_SIZES[-1]):
        from_m = []
        featballs.append(from_m)
        for j in range(FEATS_PER_OUT):
            feat_out = Sphere.get_or_create_object(
                name="feat_ball_{}_{}".format(i, j), dia=FEATBALLDIA
            )
            from_m.append(feat_out)
            feat_out.location = (
                ballsx[b],
                0,
                len(LAYER_SIZES) * LAYER_HEIGHT - LAYER_HEIGHT / 2,
            )
            b += 1

            mat, new = get_or_create_material("feat_ball_{}_{}_mat".format(i, j))
            emission, n = mat.get_or_create_node(
                type="ShaderNodeEmission", name="Emission"
            )
            new = new or n
            bsdf, n = mat.get_or_create_node("principled bsdf")
            new = new or n
            color_node, n = mat.get_or_create_node(
                type="ShaderNodeRGB", name="colornode"
            )
            new = new or n
            mixer, n = mat.get_or_create_node(type="ShaderNodeMixShader", name="Mixer")
            new = new or n

            c = np.random.random(4)
            c[3] = 1
            ca = np.random.choice(np.arange(3))
            cb = np.random.choice([i for i in np.arange(3) if i != ca])
            c[ca] = 1
            c[cb] = 0

            if new:
                mixer.Fac = 0.5
                emission.Color = color_node.Color
                # bsdf.Color = color_node.Color

                mixer.Shader_0 = emission.Emission
                mixer.Shader_1 = bsdf.BSDF

                mat.material_output.Surface = mixer.Shader_2
                bsdf.Base_Color = c
                color_node.Color = c
            emission.Strength = 1
            feat_out.material = mat

    feat_lasers = []
    for i in range(len(layer_molecules[1][0].atoms)):

        connected = []
        for b in base_mol_data["bonds"]:
            if b[0] == i:
                connected.append(b[1])
            if b[1] == i:
                connected.append(b[0])

        connected = np.array([connected, connected, connected]).flatten()
        c = np.random.random(4)
        c[:3] = c[:3] / (3 * c[:3].sum())
        c[3] = 1

        main_beam = Beam.get_or_create_object(
            "feature_beam_{}".format(i),
            start=layer_molecules[-1][0].atoms[i].world_location,
            end=featballs[0][0].world_location,
            speed=0.25,
            forcelength=LAYER_HEIGHT * 2,
        )
        main_beam.atom_n = i
        c = np.array(c, dtype=float)
        main_beam.add_layer(color=c, dia=0.5)

        main_beam.add_layer(
            dia=0.05, color=(1, 1, 1, 1), trans_dens=0.48, wave_height=0.2
        )

        main_beam.add_layer(
            dia=0.6,
            color=c / 4 + np.array([c[1], c[2], c[0], 8 * c[3]]) / 8,
            trans_dens=0.56,
        )

        feat_lasers.append(main_beam)
        for l in main_beam._layer:
            l.obj.data.bevel_factor_end = 0

    delete_all_but(
        [
            base_mol,
            layer_molecules,
            cams,
            sun,
            first_beam,
            first_layer_lasers,
            second_layer_lasers,
            featballs,
            feat_lasers,
        ]
    )

    # packing
    layers_mols = BlenderObject.get_or_create_object("layer_mols")
    layers_mols_list = []
    for l in range(len(layer_molecules)):
        layer_mols = BlenderObject.get_or_create_object("layer_mols_{}".format(l))
        layer_mols.parent = layers_mols
        layers_mols_list.append(layer_mols)
        for i in range(len(layer_molecules[l])):
            layer_molecules[l][i].parent = layer_mols
        if l > 0:
            layer_mols.location = (0, 0, l * LAYER_HEIGHT)

    first_layer_lasers_pack = BlenderObject.get_or_create_object(
        "first_layer_lasers_pack"
    )
    for l in range(len(first_layer_lasers)):
        first_layer_lasers[l].parent = first_layer_lasers_pack

    second_layer_lasers_pack = BlenderObject.get_or_create_object(
        "second_layer_lasers_pack"
    )
    for l in range(len(second_layer_lasers)):
        second_layer_lasers[l].parent = second_layer_lasers_pack

    anim = BlenderAnimationTracker()
    anim.clear_all()

    cam.track(layer_molecules[1][0], influence=0, animator=anim)

    # move out
    for l in range(len(layer_molecules)):
        if l > 0:
            anim.move_object(layers_mols_list[l], z=OUT_DIST, reverse=True, delta=True)
    for i in range(len(layer_molecules[0])):
        anim.move_object(layer_molecules[0][i], z=OUT_DIST, reverse=True, delta=True)

    for i in range(LAYER_SIZES[-1]):
        for ball in featballs[i]:
            anim.move_object(ball, z=OUT_DIST, delta=True, time=None)

    anim.save_frame("start")
    for atom in base_mol.atoms:
        anim.change_node_value(
            atom.material.get_node("Emission").Strength, 30, time=0.25
        )
        anim.change_node_value(
            atom.material.get_node("Emission").Strength, 1, time=0.25
        )

    anim.save_frame("post_blink")

    anim.save_frame("first_layer_camera_follow_done")

    anim.move_object(cam, *(0, 0, 60), delta=True, time=0.4)
    max_mol_create = min(len(layer_molecules[0]), 16)

    for i in range(len(layer_molecules[0])):

        if i == max_mol_create:
            df = anim.current_frame
            anim.move_object(cam, *(0, 0, 240), time=1, reverse=False)
            anim.save_frame("first_layer_camera_follow_done")
            anim.go_to_frame(df)

        if i >= max_mol_create:
            anim.move_object(
                base_mol, *(ordered[i][0], ordered[i][1], MOl_MOVE_HEIGHT), time=0.1
            )
        else:
            anim.move_objects(
                [base_mol, cam],
                *(ordered[i][0], ordered[i][1], MOl_MOVE_HEIGHT),
                time=0.2
            )

        anim.move_object(
            layer_molecules[0][i], *(ordered[i][0], ordered[i][1], 0), time=None
        )
        anim.save_frame("first_layer_complete_down")

    anim.go_to_frame(anim.get_frame("first_layer_camera_follow_done"))

    anim.run_seconds(1)

    anim.rotate_object(cam, 45, 0, 30, reverse=True, time=2)
    anim.move_object(cam, 150, -250, 290, time=2)
    anim.run_seconds(
        max(
            0.5,
            anim.frames_to_seconds(
                anim.get_frame("first_layer_complete_down") - anim.current_frame
            ),
        )
    )
    anim.save_frame("first_layer_zoom_out")

    D = 75
    for l in range(len(layer_molecules)):
        if l > 0:
            anim.move_object(
                layers_mols_list[l],
                z=-OUT_DIST + D,
                delta=True,
                time=None,
                reverse=True,
            )
            # layer_mols.location = (0, 0, l * LAYER_HEIGHT + OUT_DIST)
    anim.run_frames(1)

    anim.rotate_object(cam, 90, 0, 0, time=1.9, reverse=True)
    anim.move_object(
        cam, 0, -LAYER_HEIGHT * 6, LAYER_HEIGHT / 2, time=1.9, reverse=True
    )

    anim.move_object(layers_mols_list[1], z=-D, delta=True, time=2)

    for l in range(len(layer_molecules)):
        if l > 1:
            anim.move_object(
                layers_mols_list[l], z=-D, delta=True, time=2, reverse=True
            )

    anim.move_object(
        cam, 0, -100, LAYER_HEIGHT * (len(layer_molecules) + 1), time=2, reverse=True
    )
    anim.rotate_object(cam, 45, 0, 0, time=2)
    anim.rotate_object(cam, 35, 0, 0, time=1)

    anim.save_frame("other_layer_inlet")

    anim.rotate_object(cam, 0, 0, 0, reverse=True, time=2)
    anim.move_object(cam, *(ordered[10][0], ordered[10][1], 50), time=2)

    anim.rotate_object(cam, 56, 0, 30, reverse=True, time=1)
    anim.move_object(cam, *(ordered[10][0] + 20, ordered[10][1] - 29, 22), time=1)

    anim.save_frame("beam_ini_zoom_in")

    t0, t1, t2 = 0.2, 1, 0.5
    t4 = 2

    first_beam.animate(anim, t0=t0, t1=t1, t2=t2)
    anim.save_frame("beam_fired")

    anim.rotate_object(cam, 60, 0, 90, reverse=True, time=1)
    anim.move_object(
        cam,
        *(
            ordered[10][0] + layer_molecules[0][10].atoms[16].location[0] + 30,
            ordered[10][1] + layer_molecules[0][10].atoms[16].location[1],
            22,
        ),
        time=1
    )

    anim.rotate_object(cam, 105, 0, 90, reverse=True, time=1)
    anim.move_object(
        cam,
        *(
            ordered[10][0] + layer_molecules[0][10].atoms[16].location[0],
            ordered[10][1] + layer_molecules[0][10].atoms[16].location[1],
            10,
        ),
        time=1
    )

    anim.move_object(
        cam,
        *(
            17,
            ordered[10][1] + layer_molecules[0][10].atoms[16].location[1],
            LAYER_HEIGHT - 10,
        ),
        time=1
    )

    anim.rotate_object(cam, 105, 0, 0, reverse=True, time=1)
    anim.move_object(cam, *(0, -200, 0), time=1)

    anim.save_frame("beam_followed")
    first_beam.animate_stop(anim, t=t4)

    anim.save_frame("beam_dissolved")

    rt = 10
    anim.rotate_object(cam, 90, 0, -90, reverse=True, time=rt)
    anim.move_object(cam, *(-270, -10, LAYER_HEIGHT / 2), time=rt)
    anim.save_frame("post_beam_dissolved_rotation")
    anim.go_to_frame("beam_dissolved")
    for a in range(len(layer_molecules[0])):
        nl = layer_molecules[0][a].atoms[16].world_location
        nl = np.array(nl)
        anim.move_object(first_beam, *nl, time=None)
        st = t1 + t2 + t0
        if st > MIN_SHOOTING_TIME:
            t0, t1, t2 = t0 / 2, t1 / 2, t2 / 2
        elif st < MIN_SHOOTING_TIME:
            df = MIN_SHOOTING_TIME / st
            t0, t1, t2 = t0 * df, t1 * df, t2 * df

        if t4 > 0.1:
            t4 = t4 / 2
        first_beam.animate(anim, t0=t0, t1=t1, t2=t2)
        first_beam.animate_stop(anim, t=t4)
    anim.save_frame("first_laser_finished")

    st = t1 + t2 + t0
    df = DEFAULT_SHOOTING_TIME / st
    t0, t1, t2 = t0 * df, t1 * df, t2 * df

    for i in range(anim.seconds_to_frames(10)):
        cf = anim.current_frame
        chances = np.ones(len(first_layer_lasers))
        for i, l in enumerate(first_layer_lasers):
            if l.in_animation(animator=anim):
                chances[i] = 0
        if chances.sum() == 0:
            continue
        chances = chances / chances.sum()
        laser = np.random.choice(first_layer_lasers, p=chances)

        if not laser.in_animation(animator=anim):
            a = np.random.choice(np.arange(len(layer_molecules[0])))
            nl = layer_molecules[0][a].atoms[laser.atom_n].world_location
            nl = np.array(nl)
            anim.move_object(laser, *nl, time=None)

            laser.animate(anim, t0=t0, t1=t1, t2=t2)
            laser.animate_stop(anim, t=t4)

        anim.go_to_frame(cf)
        anim.run_frames(1)

    anim.save_frame("first_layer_laser_single_mol_done")

    anim.go_to_frame(
        max(
            anim.get_frame("first_laser_finished"),
            anim.get_frame("post_beam_dissolved_rotation"),
        )
    )
    anim.save_frame("pre_under_mol_laser_flight")
    anim.run_seconds(1)

    cam.track(
        layer_molecules[1][0],
        influence=1,
        animator=anim,
        frames=anim.seconds_to_frames(1),
    )
    anim.rotate_object(cam, 180, 0, 0, reverse=True, time=4)
    loc = layer_molecules[1][0].location
    loc[2] = 1
    anim.move_object(cam, *loc, time=4)
    anim.run_frames(-anim.seconds_to_frames(1))
    cam.track(
        layer_molecules[1][0],
        influence=0,
        animator=anim,
        frames=anim.seconds_to_frames(1),
    )
    anim.run_frames(anim.seconds_to_frames(1))
    anim.run_seconds(1)

    anim.save_frame("under_mol_laser_flight")

    cam.track(
        layer_molecules[1][0],
        influence=1,
        animator=anim,
        frames=anim.seconds_to_frames(1),
    )
    anim.rotate_object(cam, 90, 0, 90, reverse=True, time=4)
    anim.move_object(cam, *(270, -10, LAYER_HEIGHT / 2), time=4)
    # anim.run_seconds(-1)

    anim.run_seconds(-1)
    cam.track(
        layer_molecules[1][0],
        influence=0,
        animator=anim,
        frames=anim.seconds_to_frames(1),
    )
    anim.run_seconds(1)

    anim.save_frame("post_under_mol_laser_flight")

    anim.go_to_frame(
        max(
            anim.get_frame("first_layer_laser_single_mol_done"),
            anim.get_frame("post_under_mol_laser_flight"),
        )
    )
    anim.save_frame("single_mol_done_flight_n_shoot_done")

    for i in range(anim.seconds_to_frames(15)):
        cf = anim.current_frame
        chances = np.ones(len(first_layer_lasers))
        for i, l in enumerate(first_layer_lasers):
            if l.in_animation(animator=anim):
                chances[i] = 0
        if chances.sum() == 0:
            continue
        chances = chances / chances.sum()
        laser_n = np.random.choice(np.arange(len(first_layer_lasers)), p=chances)
        laser = first_layer_lasers[laser_n]
        if not laser.in_animation(animator=anim):
            target_n = np.random.choice(np.arange(len(layer_molecules[1])))

            laser.target = layer_molecules[1][target_n].atoms[laser.atom_n]
            a = np.random.choice(np.arange(len(layer_molecules[0])))
            nl = layer_molecules[0][a].atoms[laser.atom_n].world_location

            nl = np.array(nl)

            anim.move_object(laser, *nl, time=None)

            laser.animate(anim, t0=t0, t1=t1, t2=t2)
            laser.animate_stop(anim, t=t4)

        anim.go_to_frame(cf)
        anim.run_frames(1)
    anim.save_frame("firslayer_full_shoot_done")

    anim.run_seconds(-7)
    anim.rotate_object(cam, 90, -90, 90, reverse=True, time=2)
    anim.move_object(cam, *(270, -10, LAYER_HEIGHT), time=2)

    anim.save_frame("startsecond_layer_shoot")

    for i in range(anim.seconds_to_frames(5)):
        cf = anim.current_frame
        chances = np.ones(len(second_layer_lasers))
        for i, l in enumerate(second_layer_lasers):
            if l.in_animation(animator=anim):
                chances[i] = 0
        if chances.sum() == 0:
            continue
        chances = chances / chances.sum()

        laser_n = np.random.choice(np.arange(len(second_layer_lasers)), p=chances)
        laser = second_layer_lasers[laser_n]
        if not laser.in_animation(animator=anim):
            target_n = np.random.choice(np.arange(len(layer_molecules[2])))
            a_n = laser.atom_n  # laser_n%len(layer_molecules[2][target_n].atoms)
            print("L2", a_n, laser_n, LASER_MULTIPLIER_SECOND_LAYER)
            laser.target = layer_molecules[2][target_n].atoms[a_n]
            a = np.random.choice(np.arange(len(layer_molecules[1])))
            nl = layer_molecules[1][a].atoms[a_n].world_location
            nl = np.array(nl)
            anim.move_object(laser, *nl, time=None)

            laser.animate(anim, t0=t0, t1=t1, t2=t2)
            anim.run_frames(1)
            laser.animate_stop(anim, t=t4)

        anim.go_to_frame(cf)
        anim.run_frames(1)

    anim.run_seconds(-3)

    anim.move_object(
        cam, cam.location[0] / 2, cam.location[1] / 2, LAYER_HEIGHT * 2, time=1
    )

    anim.rotate_object(cam, 0, 0, 0, time=1, reverse=True)
    anim.move_object(cam, 0, 0, LAYER_HEIGHT * 4, time=1)

    for i in range(LAYER_SIZES[-1]):
        for ball in featballs[i]:
            anim.move_object(ball, z=-OUT_DIST, delta=True, time=None)
    anim.rotate_object(cam, 75, 0, 0, time=1, reverse=True)
    anim.move_object(cam, 0, -150, LAYER_HEIGHT * 3, time=1)

    anim.save_frame("pregraph_collect")

    cf = anim.current_frame
    anim.go_to_frame(0)
    ball = featballs[0][0]
    for l in feat_lasers:
        anim.register_finish_hook(l.create_motion)
        l.end = ball.world_location
        target_rot = l.calc_rot()
        anim.rotate_object(l, *target_rot, time=None, reverse=True)

        for layer in l._layer:
            anim.bevel_start_end(layer, 0, 0, time=None, reverse=True)
    anim.go_to_frame(cf)

    for i in range(LAYER_SIZES[-1]):
        for ball in featballs[i]:
            for l in feat_lasers:
                preloc = l.location
                preend = l.end
                prestart = l.start
                l.start = layer_molecules[-1][i].atoms[l.atom_n].world_location
                l.end = ball.world_location
                target_rot = l.calc_rot()
                l.end = preend
                l.start = prestart
                anim.rotate_object(l, *target_rot, time=None, reverse=True)
                l.end = ball.world_location
                l.start = layer_molecules[-1][i].atoms[l.atom_n].world_location
                l.location = preloc
                for layer in l._layer:
                    anim.bevel_start_end(layer, 0, 0, time=None, reverse=True)
                anim.move_object(
                    l,
                    *layer_molecules[-1][i].atoms[l.atom_n].world_location,
                    time=None,
                    reverse=True
                )

            anim.run_frames(1)
            em = ball.material.get_node("Emission")
            for j, l in enumerate(feat_lasers):
                length = np.linalg.norm(
                    ball.world_location
                    - layer_molecules[-1][i].atoms[l.atom_n].world_location
                )
                bv = length / l.length
                for layer in l._layer:
                    anim.bevel_start_end(layer, 0, bv, time=0.2, reverse=True)

                anim.run_seconds(0.2)
                anim.change_node_value(
                    em.Strength,
                    1 + 15 * (j + 1) / len(feat_lasers),
                    time=0,
                    reverse=True,
                )

            anim.run_seconds(0.5)

            for l in feat_lasers:
                length = np.linalg.norm(
                    ball.world_location
                    - layer_molecules[-1][i].atoms[l.atom_n].world_location
                )
                bv = length / l.length
                for layer in l._layer:
                    anim.bevel_start_end(layer, bv, bv, time=0.2, reverse=True)
            anim.run_seconds(0.2)

            anim.run_seconds(0.1)

    anim.run_seconds(0.1)
    anim.finish_animation(start="pregraph_collect", end=None, current=None)


script.main = mol_main

with open(__file__.replace(".py", "_blend.py"), "w+") as f:
    f.write(script.to_script())

import os

d = """
import sys
sys.path.append('{}')
import importlib
import {} as blenderscript
import importlib
importlib.reload(blenderscript)
""".format(
    os.path.dirname(__file__.replace(".py", "_blend.py")),
    os.path.basename(__file__.replace(".py", "_blend.py")).replace(".py", ""),
)

print(d)
9
