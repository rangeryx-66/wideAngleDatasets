#在blender中批量导入模型
#这个脚本在blender的script模块里面使用
import bpy
import os

root_dir = "C:/Path/To/Your/Main/Folder"

def import_obj(filepath):
    bpy.ops.import_scene.obj(
        filepath=filepath,
        axis_forward='-Z',
        axis_up='Y',
        use_edges=True,
        use_smooth_groups=True,
        use_split_objects=False,
        use_split_groups=False,
        use_groups_as_vgroups=False,
        use_image_search=True,
        split_mode='OFF'
    )


for root, dirs, files in os.walk(root_dir):

    obj_files = [f for f in files if f.lower().endswith(".obj")]


    if len(obj_files) == 0:
        continue


    for obj_file in obj_files:
        full_path = os.path.join(root, obj_file)

        bpy.ops.object.select_all(action='DESELECT')

        import_obj(full_path)
        print(f"已从 [{os.path.basename(root)}] 导入: {obj_file}")

        for obj in bpy.context.selected_objects:
            obj.name = f"{os.path.basename(root)}_{obj.name}"

print("所有子文件夹中的OBJ文件已导入完毕")