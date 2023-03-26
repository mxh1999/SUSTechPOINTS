import open3d as o3d
import os
import numpy as np
import json
from . import load_scannet_data

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

DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191])


def export_one_scan(scan_name,
                    out_dir,
                    label_map_file,
                    data_dir,
                    test_mode=False):
    # data = SensorData(osp.join(scannet_dir, scan_name, scan_name + '.sens'), -1)
    mesh_file = os.path.join(data_dir, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(data_dir, scan_name + '.aggregation.json')
    seg_file = os.path.join(data_dir, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(data_dir, f'{scan_name}.txt')
    mesh_vertices, semantic_labels, instance_labels, unaligned_bboxes, \
        aligned_bboxes, instance2semantic, axis_align_matrix = load_scannet_data.export(
            mesh_file, agg_file, seg_file, meta_file, label_map_file, None,
            test_mode)

    if not test_mode:
        mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
        mesh_vertices = mesh_vertices[mask, :]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        num_instances = len(np.unique(instance_labels))
        print(f'Num of instances: {num_instances}')

        bbox_mask = np.in1d(unaligned_bboxes[:, -1], OBJ_CLASS_IDS)
        unaligned_bboxes = unaligned_bboxes[bbox_mask, :]
        bbox_mask = np.in1d(aligned_bboxes[:, -1], OBJ_CLASS_IDS)
        aligned_bboxes = aligned_bboxes[bbox_mask, :]
        assert unaligned_bboxes.shape[0] == aligned_bboxes.shape[0]
        print(f'Num of care instances: {unaligned_bboxes.shape[0]}')
    
    os.makedirs(os.path.join(out_dir, 'label'))
    
    res = []
    for i in range(aligned_bboxes.shape[0]):
        x,y,z,dx,dy,dz, label = aligned_bboxes[i]
        res.append({"obj_id" : str(i+1),
                    "obj_type": str(label),
                    "psr": {
                        "position": {"x": x, "y": y, "z": z},
                        "rotation": {"x": 0, "y": 0, "z": 0},
                        "scale": {"x": dx, "y": dy, "z": dz}
                    }})
    with open(os.path.join(out_dir, 'label', 'main.json'),'w') as f:
        json.dump(res, f)

def scannet_init(out_dir, in_dir, scene):
    data_dir = os.path.join(in_dir, scene)
    pcd = o3d.io.read_point_cloud(os.path.join(data_dir, f'{scene}_vh_clean.ply'), format='ply')
    rot_matrix = get_prerot_mat(data_dir, scene)
    pcd.transform(rot_matrix)
    os.makedirs(os.path.join(out_dir, 'lidar'))
    o3d.io.write_point_cloud(os.path.join(out_dir, 'lidar', 'main.pcd'), pcd)
    export_one_scan(scan_name=scene, out_dir=out_dir, label_map_file=os.path.join(in_dir, 'scannetv2-labels.combined.tsv'), data_dir=data_dir)