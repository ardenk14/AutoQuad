import numpy as np
import bpy
import bpycv
import cv2
from mathutils import Vector
from mathutils import Euler
import copy

class DroneEnv():
    def __init__(self):
        try:
            cube = bpy.data.objects['Cube']
            bpy.data.objects.remove(cube, do_unlink=True)
        except:
            print("Object bpy.data.objects['Cube'] not found")
            
        #bpy.ops.outliner.orphans_purge()

        # Materials:
        self.red = bpy.data.materials.new("Red")
        self.red.diffuse_color = (1,0,0,0.8)
        self.green = bpy.data.materials.new("Green")
        self.green.diffuse_color = (0,1,0,0.8)
        self.blue = bpy.data.materials.new("Blue")
        self.blue.diffuse_color = (0,0,1,0.8)

        # Simulation parameters
        self.tracking_path = True

        #self.sphere1 = self.new_sphere((0,0,0), 0.3, "MySphere1")
        #self.sphere1.active_material = red
        self.sphere2 = self.new_sphere((2,0,0), 0.5, "MySphere2")

        bpy.ops.mesh.primitive_plane_add(size=100, location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0))
        current_name = bpy.context.selected_objects[0].name
        self.plane = bpy.data.objects[current_name]
        self.plane.name = "Plane"
        self.plane.data.name = "Plane" + "_mesh"

        self.camera = bpy.data.objects["Camera"]
        width = 1280
        height = 720
        #self.camera.location = Vector((0.17, -0.01, 0.0))
        #self.camera.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
        bpy.context.scene.frame_set(1)
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.samples = 1
        bpy.context.scene.render.resolution_y = 180#360
        bpy.context.scene.render.resolution_x = 320#640

        # Add drone
        self.t = None
        self.quad_state = None
        self.quad_pos = None
        self.quad_vel = None
        self.quad_quat = None
        self.quad_omega = None
        self.quad_euler = None
        #sDes_traj_all[i,:]   = traj.sDes
        self.ctrl_sDesCalc = None
        self.ctrl_w_cmd = None
        self.quad_wMotor = None
        self.quad_thr = None
        self.quad_tor = None

        self.quad = self.create_quad()
        self.quad.active_material = self.red

    def set_state(self, t, quad_state, quad_pos, quad_vel, quad_quat, quad_omega, quad_euler, ctrl_sDesCalc, ctrl_w_cmd, quad_wMotor, quad_thr, quad_tor):
        self.t = t
        self.quad_state = copy.deepcopy(quad_state)
        self.quad_pos = copy.deepcopy(quad_pos)
        self.quad_vel = copy.deepcopy(quad_vel)
        self.quad_quat = copy.deepcopy(quad_quat)
        self.quad_omega = copy.deepcopy(quad_omega)
        self.quad_euler = copy.deepcopy(quad_euler)
        #sDes_traj_all[i,:]   = traj.sDes
        self.ctrl_sDesCalc = copy.deepcopy(ctrl_sDesCalc)
        self.ctrl_w_cmd = copy.deepcopy(ctrl_w_cmd)
        self.quad_wMotor = copy.deepcopy(quad_wMotor)
        self.quad_thr = copy.deepcopy(quad_thr)
        self.quad_tor = copy.deepcopy(quad_tor)

        actual_pos = self.quad_pos
        actual_pos[-1] *= -1

        self.quad.location = actual_pos
        self.quad.rotation_euler = Euler(self.quad_euler, 'XYZ')

        if self.tracking_path is True:
            sphere = self.new_sphere(actual_pos, 0.01, str(t))
            sphere.active_material = self.blue


    def get_img(self):
        result = bpycv.render_data()
        rgb = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
        depth = result["depth"]
        return rgb

    def new_camera(self):
        pass

    def set_camera(self, pos):
        pass

    def create_quad(self):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.1,
            depth= 0.3,)
        current_name = bpy.context.selected_objects[0].name
        drone = bpy.data.objects[current_name]
        drone.name = "DRONE"
        drone.data.name = "DRONE" + "_mesh"
        return drone
    
    def new_sphere(self, mylocation, myradius, myname):
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=64, 
            ring_count=32, 
            radius=myradius, 
            location=mylocation)
        current_name = bpy.context.selected_objects[0].name
        sphere = bpy.data.objects[current_name]
        sphere.name = myname
        sphere.data.name = myname + "_mesh"
        return sphere