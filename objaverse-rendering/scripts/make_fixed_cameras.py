import numpy as np
import os
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

ids = ['11581', '11586', '11691', '11778', '11854', '11876', '11945', '10040', '10098', '10101', '10383', '10306', '10626', '9992', '12073', '11242', '11586', '9968', '11477', '11429', '11156', '10885', '11395', '11075']

# '10383', '10306', '10626', '9992', '12073', '11242', '11586', '9968', '11477', '11429', '11156', '10885', '11395', '11075'
num_images = 25
for j in range(len(ids)):
    all_xyz = []
    for i in range(num_images):
        x, y, z = sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
        all_xyz.append([x, y, z])

    all_xyz = np.array(all_xyz)
    print("all_xyz", all_xyz.shape)
    #save this as .json file
    import json
    save_path = '/home/zubairirshad/zero123/objaverse-rendering/camera_positions'
    save_path = os.path.join(save_path, 'positions_partnet_fixed'+ ids[j]+'.json')
    with open(save_path, 'w') as outfile:
        json.dump(all_xyz.tolist(), outfile)