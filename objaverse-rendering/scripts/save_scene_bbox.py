
import math
import bpy
from mathutils import Vector
import os
import numpy as np

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")



def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    # return Vector(bbox_min), Vector(bbox_max)
    return bbox_min, bbox_max


if __name__ == "__main__":

    ids = ['11581', '11586', '11691', '11778', '11876', '11945', '10040', '10098', '10101', '10383', '10306', '10626', '9992', '12073', '11242', '11586', '9968', '11477', '11429', '11156', '10885', '11395', '11075']
    data_path = '/home/zubairirshad/SAPIEN/partnet-mobility-dataset'

    out_box_path = '/home/zubairirshad/zero123/objaverse-rendering/bbox_90.json'
    
    all_bboxes = {}
    for id in ids:
        object_file = os.path.join(data_path, id, 'textured_objs', '90', '90.obj')
        reset_scene()

        load_object(object_file)
        bbox_min, bbox_max = scene_bbox()
        all_bboxes[id] = [np.array(bbox_min).tolist(), np.array(bbox_max).tolist()]

    import json
    with open(out_box_path, 'w') as outfile:
        json.dump(all_bboxes, outfile)