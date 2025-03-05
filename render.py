import bpy
import math
import os
import json
from mathutils import Matrix, Vector
import mathutils
import random
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

def setup_rendering_weather(scene, scene_depth,weather):
    world = scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.clear()
    
    bg_node = world.node_tree.nodes.new('ShaderNodeBackground')
    output_node = world.node_tree.nodes.new('ShaderNodeOutputWorld')
    world.node_tree.links.new(bg_node.outputs[0], output_node.inputs[0])

    mist_settings = world.mist_settings
    mist_settings.use_mist = True
    mist_settings.start = 0

    # 根据天气设置雾效参数
    if weather == 'foggy':
        mist_settings.depth = scene_depth * 0.6
        mist_settings.falloff = 'QUADRATIC'
        bg_node.inputs[0].default_value = (0.7, 0.7, 0.7, 1)  # 灰白色雾
    elif weather == 'rainy':
        mist_settings.depth = scene_depth * 2.0
        mist_settings.falloff = 'LINEAR'
        bg_node.inputs[0].default_value = (0.5, 0.5, 0.6, 1)  # 冷色调雾
    elif weather == 'cloudy':
        mist_settings.depth = scene_depth * 1.5
        mist_settings.falloff = 'LINEAR'
        bg_node.inputs[0].default_value = (0.8, 0.85, 0.9, 1)  # 淡蓝色雾
    else:  # sunny
        mist_settings = world.mist_settings
        mist_settings.use_mist = True
        mist_settings.start = 0
        mist_settings.depth = scene_depth * 1.2
        mist_settings.falloff = 'LINEAR'  # 使用线性衰减


def create_lighting(weather,location):
    # 删除所有现有光源
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)
        
    world = bpy.context.scene.world
    bg_node = world.node_tree.nodes['Background']

    if weather == 'sunny':
        # 设置背景为明亮的蓝色天空
        # 强日光设置
        # sun = bpy.data.lights.new(name="Sun", type='SUN')
        # sun.energy = 4.0
        # sun.color = (1.0, 0.95, 0.9)
        # sun_obj = bpy.data.objects.new(name="Sun", object_data=sun)
        # bpy.context.collection.objects.link(sun_obj)
        # sun_obj.rotation_euler = (math.radians(60), 0, math.radians(45))
        bg_node.inputs[1].default_value = 3.0  # 环境光强度

    elif weather == 'rainy':
        bg_node.inputs[0].default_value = (0.5, 0.5, 0.55, 1)
        bg_node.inputs[1].default_value = 3
        area = bpy.data.lights.new(name="RainLight", type='AREA')
        area.energy = 500
        area.size = 30
        area.color = (0.5, 0.55, 0.65) 
        area_obj = bpy.data.objects.new(name="RainLight", object_data=area)
        bpy.context.collection.objects.link(area_obj)
        area_obj.location = (0, 0, 20)
        area_obj.rotation_euler = (0, 0, 0)


    elif weather == 'cloudy':

        bg_node.inputs[1].default_value = 2.5

        bg_node.inputs[0].default_value = (0.4, 0.5, 0.6, 1)
        

        area = bpy.data.lights.new(name="CloudLight", type='AREA')
        area.energy = 500
        area.size = 20

        area.color = (0.6, 0.65, 0.75)
        area_obj = bpy.data.objects.new(name="CloudLight", object_data=area)
        bpy.context.collection.objects.link(area_obj)
        
        area_obj.location = (0, -10, 15)
        area_obj.rotation_euler = (math.radians(45), 0, 0)  # 调整角度，避免太直射


    elif weather == 'foggy':
        # 雾天设置
        bg_node.inputs[1].default_value = 1.2
        bg_node.inputs[0].default_value = (0.6, 0.6, 0.6, 1)
        sun = bpy.data.lights.new(name="FogSun", type='SUN')
        sun.energy = 1.0
        sun.color = (0.9, 0.9, 0.8)
        sun_obj = bpy.data.objects.new(name="FogSun", object_data=sun)
        bpy.context.collection.objects.link(sun_obj)
        sun_obj.rotation_euler = (math.radians(30), 0, math.radians(45))

def render_views(filepath, savepath,height):
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
        
            # 添加高空卫星视角
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
        setup_rendering_weather(scene,scene_depth,weather)
        cam = bpy.data.cameras.new(f"Camera_{i}")
        cam.lens = 35
        cam.clip_start = 0.1
        cam.clip_end = scene_depth * 2
        cam.show_mist = True
        cam_obj = bpy.data.objects.new(f"Camera_{i}", cam)
        cam_obj.location = viewpoint["location"]
        cam_obj.rotation_euler = viewpoint["rotation"]


        bpy.context.collection.objects.link(cam_obj)

        create_lighting(weather,center_place)


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
    # render_views(
    #     filepath='/root/autodl-tmp/bigBuildingstore/城区城市街道楼群带道路线_爱给网_aigei_com/City.blend',
    #     savepath='/root/autodl-tmp/wideAngleDatasets/sunny/',
    #     weather='sunny'  # 可选：sunny, cloudy, rainy, foggy
    # )
    render_views(
        filepath='/root/autodl-tmp/bigBuildingstore/paris.blend',
        savepath='/root/autodl-tmp/wideAngleDatasets/pairs' ,
        height=200
    )
    # render_views(
    #     filepath='/root/autodl-tmp/bigBuildingstore/乡镇城乡结合部城区城市街道楼群_爱给网_aigei_com/乡镇.blend',
    #     savepath='wideAngleDatasets/乡镇' ,
    #     height=30
    # )
    # render_views(
    #     filepath='/root/autodl-tmp/bigBuildingstore/城区城市街道楼群带道路线_爱给网_aigei_com/City.blend',
    #     savepath='/root/autodl-tmp/wideAngleDatasets/foggy/',
    #     weather='foggy'  # 可选：sunny, cloudy, rainy, foggy
    # )
    # render_views(
    #     filepath='/root/autodl-tmp/bigBuildingstore/城区城市街道楼群带道路线_爱给网_aigei_com/City.blend',
    #     savepath='/root/autodl-tmp/wideAngleDatasets/cloudy/',
    #     weather='cloudy'  # 可选：sunny, cloudy, rainy, foggy
    # )