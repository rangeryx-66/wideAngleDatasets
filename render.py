import bpy
import math
import os
import json
from mathutils import Matrix, Vector
import mathutils
import random
import argparse

def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x = scene.render.resolution_x
    resolution_y = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width = camd.sensor_width

    if camd.sensor_fit == 'VERTICAL':
        s_u = resolution_x * scale / sensor_width
    else:
        s_u = resolution_y * scale / sensor_width

    alpha_u = f_in_mm * s_u
    u_0 = resolution_x * scale / 2
    v_0 = resolution_y * scale / 2
    skew = 0

    return Matrix(
        ((alpha_u, skew, u_0),
         (0, alpha_u, v_0),
         (0, 0, 1))
    )

def get_4x4_RT_matrix_from_blender(cam):
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    T_world2bcam = - (R_world2bcam @ location)

    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1))
    )

    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    return Matrix(
        ((R_world2cv[0][0], R_world2cv[0][1], R_world2cv[0][2], T_world2cv[0]),
         (R_world2cv[1][0], R_world2cv[1][1], R_world2cv[1][2], T_world2cv[1]),
         (R_world2cv[2][0], R_world2cv[2][1], R_world2cv[2][2], T_world2cv[2]),
         (0, 0, 0, 1))
    )

def calculate_scene_dimensions(obj):
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    min_coord = Vector((min(v.x for v in bbox),
                       min(v.y for v in bbox),
                       min(v.z for v in bbox)))
    max_coord = Vector((max(v.x for v in bbox),
                       max(v.y for v in bbox),
                       max(v.z for v in bbox)))
    
    return max_coord - min_coord

def setup_rendering(scene):
    scene.display_settings.display_device = 'None'
    scene.view_settings.view_transform = 'Standard'
    scene.sequencer_colorspace_settings.name = 'Raw'

    scene.render.engine = 'CYCLES'
    scene.view_layers["ViewLayer"].use_pass_mist = True
    scene.view_layers["ViewLayer"].use_pass_z = True
    scene.use_nodes = True
    scene.cycles.device = 'GPU'
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        
        print(d["name"])

    scene.cycles.samples = 256
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01


def load_environment_map(hdr_path, strength=1.0):
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # 清空现有节点
    nodes.clear()

    # 创建环境纹理节点
    env_texture = nodes.new('ShaderNodeTexEnvironment')
    env_texture.image = bpy.data.images.load(hdr_path)

    # 创建背景和输出节点
    background = nodes.new('ShaderNodeBackground')
    output = nodes.new('ShaderNodeOutputWorld')

    # 连接节点
    links.new(env_texture.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], output.inputs['Surface'])

    # 设置环境光强度
    background.inputs['Strength'].default_value = strength


# 修改setup_rendering_weather函数
def setup_rendering_weather(scene, scene_depth, weather, hdr_path):
    world = scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        scene.world = world

    # 加载环境贴图（统一使用传入的hdr_path）
    load_environment_map(hdr_path, strength=1.0)

    # 保留雾效设置
    mist_settings = world.mist_settings
    mist_settings.use_mist = True
    mist_settings.start = 0

    # 根据天气调整雾效参数（保留原有逻辑）
    if weather == 'foggy':
        mist_settings.depth = scene_depth * 0.6
        mist_settings.falloff = 'QUADRATIC'
    elif weather == 'rainy':
        mist_settings.depth = scene_depth * 2.0
        mist_settings.falloff = 'LINEAR'
    elif weather == 'cloudy':
        mist_settings.depth = scene_depth * 1.5
        mist_settings.falloff = 'LINEAR'
    else:  # sunny
        mist_settings.depth = scene_depth * 1.2
        mist_settings.falloff = 'LINEAR'


def render_views(filepath, savepath, height, hdr_path):
    bpy.ops.object.delete()

    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects

    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)

    imported_objects = [obj for obj in data_to.objects if obj.type == 'MESH']
    if imported_objects:
        bpy.context.view_layer.objects.active = imported_objects[0]
        bpy.ops.object.select_all(action='DESELECT')
        for obj in imported_objects:
            obj.select_set(True)
        bpy.ops.object.join()
    
    obj = bpy.context.active_object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
    
    scene_dim = calculate_scene_dimensions(obj)
    scene_depth = max(scene_dim.x, scene_dim.y, scene_dim.z)

    scene = bpy.context.scene
    setup_rendering(scene)
    center_place = obj.location
    print("center_place")
    print(center_place)
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0


    centers=[]
    viewpoints=[]
    for i in range(-2,2,1):
        for j in range(-2,2,1):
            new_center = mathutils.Vector((center_place.x +i*150+75, center_place.y +j*150+75, center_place.z)) 
            centers.append(new_center)

    for h in range(500, 900, 200):
        for center in centers:

            satelite_id=0
            location = (center.x, center.y, center.z + h)
            rotation = (0, 0, 0)
            viewpoints.append({
                "type":"satelite",
                "location": location,
                "rotation": rotation,
                "height": h,
                "view_id": len(viewpoints),
                "parent_view_id": -1,
            })
            satelite_id=len(viewpoints)-1

            sensor_width = 36  # mm
            focal_length = 35  # mm
            aspect_ratio = scene.render.resolution_y / scene.render.resolution_x
            sensor_height = sensor_width * aspect_ratio

            # 计算地面覆盖范围
            ground_width = (sensor_width * h) / focal_length
            ground_height = (sensor_height * h) / focal_length

            # 计算四个角落的坐标（相对于场景中心）
            corners=[]
            for i in range(8):
                corners.append((-ground_width/2+i*ground_width/8,ground_height/2))
                corners.append((-ground_width/2+i*ground_width/8,-ground_height/2))
                corners.append((ground_width/2,-ground_height/2+i*ground_height/8))
                corners.append((-ground_width/2,-ground_height/2+i*ground_height/8))
                

            # 生成四个低空无人机视角
            for dx, dy in corners:
                x = center.x + dx
                y = center.y + dy
                z = center.z + height

                # 计算相机旋转四元数
                yaw = math.atan2(dy, dx)+math.pi/2 #注意这里的旋转的z的角度和数学上的旋转符号是相反的
                viewpoints.append({
                        "type":"drone",
                        "location": (x, y, z),
                        "rotation": (math.pi/2-0.7, 0, yaw),
                        "height":height,
                        "parent_view_id": satelite_id,
                        "view_id": len(viewpoints),
                    })
        
        

    camera_params = []
    weathers=["sunny","cloudy","rainy","foggy"]
    for i, viewpoint in enumerate(viewpoints):
        
        weather_idx = random.randint(0,3)
        # weather_idx=i
        weather=weathers[weather_idx]
        # weather_idx=0
        setup_rendering_weather(scene, scene_depth, weather, hdr_path)
        cam = bpy.data.cameras.new(f"Camera_{i}")
        cam.lens = 35
        cam.clip_start = 0.1
        cam.clip_end = scene_depth * 2
        cam.show_mist = True
        cam_obj = bpy.data.objects.new(f"Camera_{i}", cam)
        cam_obj.location = viewpoint["location"]
        cam_obj.rotation_euler = viewpoint["rotation"]


        bpy.context.collection.objects.link(cam_obj)


        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = os.path.join(savepath, f"render_{i}.png")
        scene.camera = cam_obj

        tree = scene.node_tree
        tree.nodes.clear()
                
        rl_node = tree.nodes.new('CompositorNodeRLayers')

        normalize_node = tree.nodes.new('CompositorNodeNormalize')
        invert_node = tree.nodes.new('CompositorNodeInvert')
        invert_node.inputs['Color'].default_value = (1, 1, 1, 1)

        depth_exr_node = tree.nodes.new('CompositorNodeOutputFile')
        depth_exr_node.base_path = savepath+'/depthexr/'
        depth_exr_node.file_slots[0].path = f"raw_depth_{i}_"
        depth_exr_node.format.file_format = 'OPEN_EXR'
        depth_exr_node.format.color_depth = '32'
        depth_exr_node.format.color_mode = 'RGBA' 

        depth_file_node = tree.nodes.new('CompositorNodeOutputFile')
        depth_file_node.base_path = savepath+'/depthmap/'
        depth_file_node.file_slots[0].path = f"depth_{i}_"
        depth_file_node.format.file_format = 'PNG'
        depth_file_node.format.color_depth = '16'
        depth_file_node.format.color_mode = 'BW'

        image_file_node = tree.nodes.new('CompositorNodeOutputFile')
        image_file_node.base_path = savepath+'/rgb/'
        image_file_node.file_slots[0].path = f"color_{i}_"

        tree.links.new(rl_node.outputs['Depth'], normalize_node.inputs['Value'])
        tree.links.new(normalize_node.outputs['Value'], invert_node.inputs['Color'])
        tree.links.new(invert_node.outputs['Color'], depth_file_node.inputs[0])
        tree.links.new(rl_node.outputs['Image'], image_file_node.inputs[0])
        tree.links.new(rl_node.outputs['Depth'], depth_exr_node.inputs[0])

        bpy.ops.render.render(write_still=True)

        K = get_calibration_matrix_K_from_blender(cam)
        RT = get_4x4_RT_matrix_from_blender(cam_obj)
        
        camera_params.append({
            "view_id": i,
            "type": viewpoint["type"],
            "parent_view_id": viewpoint.get("parent_view_id", None),
            "K": [list(row) for row in K],
            "RT": [list(row) for row in RT],
            "weather":weather,
            "location":viewpoint["location"],
            "rotation":viewpoint["rotation"],
        })

    with open(os.path.join(savepath, "camera_params.json"), "w") as f:
        json.dump(camera_params, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render 3D model views with Blender')
    parser.add_argument('--filepath', type=str, required=True,
                        help='Path to input .blend file')
    parser.add_argument('--savepath', type=str, required=True,
                        help='Path to save rendered outputs')
    parser.add_argument('--height', type=int, required=True,
                        help='Drone camera height setting')
    parser.add_argument('--environment_map', type=str, required=True,
                        help='Path to HDR environment map')

    args = parser.parse_args()

    # 修改函数调用
    render_views(
        filepath=args.filepath,
        savepath=args.savepath,
        height=args.height,
        hdr_path=args.environment_map  # 新增参数
    )
# blender --background --python render.py -- --filepath model.blend --savepath ./output --height 200 --environment_map /path/to/env.hdr
