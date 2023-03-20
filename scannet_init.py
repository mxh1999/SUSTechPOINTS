import open3d as o3d
import os
import numpy as np

def get_prerot_mat(data_dir, scene):
    with open(os.path.join(data_dir, f'{scene}.txt'),'r') as f:
        lines = f.readlines()
    # test set data doesn't have align_matrix
    axis_align_matrix = np.eye(4)
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [
                float(x)
                for x in line.rstrip().strip('axisAlignment = ').split(' ')
            ]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    return axis_align_matrix

def scannet_init(out_dir, data_dir, scene):
    pcd = o3d.io.read_point_cloud(os.path.join(data_dir, f'{scene}_vh_clean.ply'), format='ply')
    rot_matrix = get_prerot_mat(data_dir, scene)
    pcd.transform(rot_matrix)
    os.makedirs(os.path.join(out_dir, 'lidar'))
    o3d.io.write_point_cloud(os.path.join(out_dir, 'lidar', 'main.pcd'), pcd)
    