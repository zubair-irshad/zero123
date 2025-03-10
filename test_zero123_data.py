import random
import numpy as np
# data_folder = '/home/zubairirshad/zero123/objaverse-rendering/partnet_mobility_fixed_camera/11581/20/20'

data_folder = '/home/zubairirshad/zero123/objaverse-rendering/partnet_mobility_fixed_camera/11581'

import os
import torch
import math

def cartesian_to_spherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    z = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[:,1], xyz[:,0])
    return np.array([theta, azimuth, z])


def get_T_sapien(target_RT, cond_RT):
    # R, T = target_RT[:3, :3], target_RT[:, -1]
    # T_target = -R.T @ T

    # R, T = cond_RT[:3, :3], cond_RT[:, -1]
    # T_cond = -R.T @ T

    R, T = target_RT[:3, :3], target_RT[:3, 3]
    T_target = T

    R, T = cond_RT[:3, :3], cond_RT[:3, 3]
    T_cond = T

    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])
    
    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond
    
    d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
    return d_T

def get_T(target_RT, cond_RT):
    R, T = target_RT[:3, :3], target_RT[:, -1]
    T_target = -R.T @ T

    R, T = cond_RT[:3, :3], cond_RT[:, -1]
    T_cond = -R.T @ T

    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])
    
    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond
    
    d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
    return d_T

total_view = 25

index_to_angles = {0:0, 1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60, 7: 70, 8: 80, 9: 90}


for i in range(10):
    # index_target, index_cond = random.sample(range(total_view), 2) # without replacement
    # target_RT = np.load(os.path.join(data_folder, '%03d.npy' % index_target))
    # cond_RT = np.load(os.path.join(data_folder, '%03d.npy' % index_cond))
    
    total_angles = 10
    angle_target, angle_cond = random.sample(range(total_angles), 2) # without replacement))

    angle_target = index_to_angles[angle_target]
    angle_cond = index_to_angles[angle_cond]
    index_img = random.sample(range(total_view), 1) # without replacement

    print("index_img", index_img)
    target_RT = np.load(os.path.join(data_folder, str(angle_target), str(angle_target), '%03d.npy' % index_img[0]))
    cond_RT = np.load(os.path.join(data_folder, str(angle_cond), str(angle_cond), '%03d.npy' % index_img[0]))


    # T = get_T_sapien(target_RT, cond_RT)
    T = get_T(target_RT, cond_RT)
    
    # a = T[:, None, :]
    print("T", T.shape)

    print("T", T)