#generate camera paths for all ids in advance

import numpy as np

def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
#         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def generate_camera_paths(num_ids=20):

    num_images = 12
    all_xyz = []
    all_camera_paths = {}

    for i in range(num_ids):
        for j in range(num_images):
            x, y, z = sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
            all_xyz.append([x, y, z])

        all_xyz = np.array(all_xyz)
        all_camera_paths[i] = all_xyz.tolist()

    return all_camera_paths