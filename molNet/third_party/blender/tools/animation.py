import numpy as np
import bpy

from molNet.third_party.blender.tools import blender_function
import numpy as np

@blender_function(dependencies=[])
class BlenderAnimationTracker:
    def __init__(self, fps=24):
        self.current_frame = 0
        self._max_frame = 0
        self._fps = fps

    def go_to_frame(self, f):
        self._max_frame = max(self._max_frame, f)
        self.current_frame = f

    def self_run_frames(self, f):
        self.go_to_frame(self.current_frame + np.ceil(f))

    def go_to_second(self, s):
        self.go_to_frame(self._fps * s)

    def run_seconds(self, s):
        self.self_run_frames(self._fps * s)


@blender_function(dependencies=[])
def wait(animation_tracker,s):
    animation_tracker.run_seconds(s)

@blender_function(dependencies=[])
def finish_animation(animation_tracker):
    bpy.context.scene.frame_end = animation_tracker._max_frame

@blender_function(dependencies=[])
def move_object(obj,x=0,y=0,z=0,delta=False,animator=None,time=0):
    if animator:
        obj.keyframe_insert(data_path="location", frame=animator.current_frame)
    vec = np.array([x,y,z],dtype=float)
    curr_loc=np.array(obj.location,dtype=float)
    if delta:
        obj.location = curr_loc + vec
        delta =  vec
    else:
        obj.location = vec
        delta =  np.array(obj.location)-curr_loc

    if animator:
        animator.run_seconds(time)
        obj.keyframe_insert(data_path="location", frame=animator.current_frame)

    return delta

@blender_function(dependencies=[])
def rotate_object(obj,x=0,y=0,z=0,delta=False,animator=None,time=0):
    if animator:
        obj.keyframe_insert(data_path="rotation_euler", frame=animator.current_frame)
    vec = np.array([x,y,z],dtype=float)*2*np.pi/360
    curr_rot =  np.array(obj.rotation_euler,dtype=float)
    if delta:
        obj.rotation_euler = curr_rot + vec
        delta =  vec
    else:
        obj.rotation_euler = vec
        delta =  np.array(obj.rotation_euler)-curr_rot

    if animator:
        animator.run_seconds(time)
        obj.keyframe_insert(data_path="rotation_euler", frame=animator.current_frame)


    return delta*360/(2*np.pi)

@blender_function(dependencies=[rotate_object,move_object])
def move_rotate_object(obj,x_move=0,y_move=0,z_move=0,x_rot=0,y_rot=0,z_rot=0,delta_move=False,delta_rot=False,animator=None,time=0):
    if animator:
        obj.keyframe_insert(data_path="location", frame=animator.current_frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=animator.current_frame)

    delta_r = rotate_object(obj=obj,x=x_rot,y=y_rot,z=z_rot,delta=delta_rot,animator=None)
    delta_m = move_object(obj=obj,x=x_move,y=y_move,z=z_move,delta=delta_rot,animator=None)

    if animator:
        animator.run_seconds(time)
        obj.keyframe_insert(data_path="location", frame=animator.current_frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=animator.current_frame)

    return delta_m,delta_r

@blender_function(dependencies=[move_object])
def move_objects(objs,x=0,y=0,z=0,delta=False,animator=None,time=0):
    if animator:
        for obj in objs:
            obj.keyframe_insert(data_path="location", frame=animator.current_frame)

    if delta or len(objs)<=1:
        for obj in objs:
            move_object(obj,x=x,y=y,z=z,delta=True)
    else:
        delta = move_object(objs[0],x=x,y=y,z=z,delta=False)
        for obj in objs[1:]:
            move_object(obj,*delta,delta=True)

    if animator:
        animator.run_seconds(time)
        for obj in objs:
            obj.keyframe_insert(data_path="location", frame=animator.current_frame)

@blender_function(dependencies=[])
def new_camera(location=(0,0,12),rotation_euler=(0,0,0),lens=18):
    scn = bpy.context.scene
    cam1 = bpy.data.cameras.new("Camera 1")
    cam1.lens = lens
    cam_obj1 = bpy.data.objects.new("Camera 1", cam1)
    cam_obj1.location = location
    cam_obj1.rotation_euler = rotation_euler
    scn.collection.objects.link(cam_obj1)
    bpy.context.scene.camera = cam_obj1
    return cam_obj1