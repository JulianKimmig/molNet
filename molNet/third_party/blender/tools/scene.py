import bpy

from molNet.third_party.blender.tools import blender_basic_script


@blender_basic_script
def clear_scene(types_to_clear=["MESH", "CURVE", "SURFACE","EMPTY","CAMERA"]):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    for obj in bpy.context.scene.objects:
        if obj.type in types_to_clear:
            bpy.data.objects.remove(obj)
        #else:
        #    print(obj.type,obj)

@blender_basic_script
def add_light(type="SUN",name="sunlight",energy=4,location=(0,0,300)):
    light_data = bpy.data.lights.new(name=name, type=type)
    light_data.energy = energy

    # create new object with our light datablock
    light_object = bpy.data.objects.new(name="light_2.80", object_data=light_data)

    # link light object
    bpy.context.collection.objects.link(light_object)

    # make it active
    bpy.context.view_layer.objects.active = light_object

    #change location
    light_object.location = location

@blender_basic_script
def find_object_by_name(name,sub=True):
    obj = bpy.data.objects[name]
    return obj
