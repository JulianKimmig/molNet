import bpy

from molNet.third_party.blender.tools import blender_basic_script, blender_function



@blender_basic_script
def to_group(name, objs,new=False):
    '''
    named:   (string) name of group to use, or to create if not present
    objs:    a collection of object references
    '''
    groups = bpy.data.groups

    if new:
        group = groups.new(name)
    else:
        group = groups.get(name, groups.new(name))

    for obj in objs:
        if obj.name not in group.objects:
            group.objects.link(obj)

@blender_basic_script
def copy_ob(ob, parent,  collection=bpy.context.collection):
    # copy ob
    copy = ob.copy()
    copy.parent = parent
    copy.matrix_parent_inverse = ob.matrix_parent_inverse.copy()
    # copy particle settings
    for ps in copy.particle_systems:
        ps.settings = ps.settings.copy()
    collection.objects.link(copy)


    copy.animation_data_clear()
    if ob.data is not None:
        copy.data = ob.data.copy()
        for i,m in enumerate(list(ob.data.materials)):
            copy.data.materials[i]=m.copy()
            copy.data.materials[i].animation_data_clear()
    copy.animation_data_clear()
    return copy

@blender_function(dependencies=[copy_ob])
def tree_copy(ob, parent=None, levels=3):
    def recurse(ob, parent, depth):
        if depth > levels:
            return
        copy = copy_ob(ob, parent)

        for child in ob.children:
            recurse(child, copy, depth + 1)
        return copy
    return recurse(ob, ob.parent, 0)

