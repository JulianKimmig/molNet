import bmesh
import bpy

from molNet.third_party.blender.tools import blender_basic_script, blender_function




@blender_basic_script
def create_plain_object(name, data=None):
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


@blender_function(dependencies=[create_plain_object])
def create_sphere(name="uvsphere", x=0, y=0, z=0, dia=1):
    mesh = bpy.data.meshes.new(name)
    uvsphere = create_plain_object(name, mesh)

    bm = bmesh.new()
    # bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=4, diameter=dia)
    bmesh.ops.create_cube(bm, size=dia * 2 * 1.162)
    bm.to_mesh(mesh)
    bm.free()


    uvsphere.add_modifier('spherification',Subsurface(levels=3,render_levels=6))

    mod = uvsphere.modifiers.new('spherification', 'SUBSURF')
    mod.levels = 3
    mod.render_levels = 6

    uvsphere.location = (x, y, z)

    return uvsphere


@blender_basic_script
def set_parent(child, parent):
    child.parent = parent


@blender_function(dependencies=[create_plain_object])
def create_text(text="lorem", name="font object", x=0, y=0, z=0,size=1):
    font_curve = bpy.data.curves.new(type="FONT", name="Font Curve")
    font_curve.body = text
    text = create_plain_object(name, font_curve)
    text.location = (x, y, z)
    text.data.size=size
    return text


@blender_basic_script
def connect_points(p1, p2, d=1, name="cylinder"):
    o = (p1 + p2) / 2
    bpy.ops.curve.primitive_bezier_curve_add()
    curve = bpy.context.object

    curve.data.dimensions = '3D'
    curve.data.fill_mode = 'FULL'
    curve.data.bevel_depth = d
    curve.data.bevel_resolution = 6
    # set first point to centre of sphere1
    curve.data.splines[0].bezier_points[0].co = p1 - o
    curve.data.splines[0].bezier_points[0].handle_left_type = 'VECTOR'
    # set second point to centre of sphere2
    curve.data.splines[0].bezier_points[1].co = p2 - o
    curve.data.splines[0].bezier_points[1].handle_left_type = 'VECTOR'
    curve.location = o
    return curve
