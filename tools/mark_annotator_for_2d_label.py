





import os
import json
import numpy as np
import math
import pypcd.pypcd as pypcd
import argparse
import re
from utils import proj_pts3d_to_img, read_scene_meta, get_calib_for_frame, box3d_to_corners, crop_pts, gen_2dbox_for_obj_pts

parser = argparse.ArgumentParser(description='generate 2d boxes by 3d boxes')        
parser.add_argument('data_folder', type=str, default='./data', help="")
parser.add_argument('--scenes', type=str, default='.*', help="")
parser.add_argument('--frames', type=str, default='.*', help="")
parser.add_argument('--camera_types', type=str, default='aux_camera', help="")
parser.add_argument('--camera_names', type=str, default='front', help="")
parser.add_argument('--save', type=str, default='no', help="")
args = parser.parse_args()


all_scenes = os.listdir(args.data_folder)
scenes = list(filter(lambda s: re.fullmatch(args.scenes, s), all_scenes))
scenes.sort()
#print(list(scenes))


data_folder = args.data_folder
camera_types  = args.camera_types.split(",")
camera_names  = args.camera_names.split(",")

def prepare_dirs(path):
    if not os.path.exists(path):
            os.makedirs(path)

def area(r):
    return (r['x2'] - r['x1']) * (r['y2'] - r['y1'])

def intersect(a,b):
    x1 = max(a['x1'], b['x1'])
    y1 = max(a['y1'], b['y1'])

    x2 = min(a['x2'], b['x2'])
    y2 = min(a['y2'], b['y2'])

    return {
        'x1': x1,
        'x2': x2,
        'y1': y1,
        'y2': y2
    }

def iou(a,b):
    i = intersect(a, b)

    if i['x2'] < i['x1'] or i['y2'] < i['y1']:
        return 0
    else:
        return area(i)/(area(a)+area(b) - area(i))

def same_rect(a,b):
    
    for k in ['x1','x2','y1','y2']:
        if a[k] - b[k] > 2 or a[k] - b[k] < -2:
            return False
    return True
def load_points(scene, frame):
    # load lidar points
    lidar_file = os.path.join(scene, 'lidar', frame+".pcd")
    pc = pypcd.PointCloud.from_path(lidar_file)
    
    pts =  np.stack([pc.pc_data['x'], 
                    pc.pc_data['y'], 
                    pc.pc_data['z']],
                axis=-1)
    pts = pts[(pts[:,0]!=0) | (pts[:,1]!=0) | (pts[:,2]!=0)]

    return pts


def proc_frame_camera(scene, meta, frame, camera_type, camera, extrinsic, intrinsic, objs):

    #print(camera_type, camera)

    pts = None

    label2d_file = os.path.join(scene, 'label_fusion', camera_type, camera, frame+".json")

    if os.path.exists(label2d_file):
        with open(label2d_file) as f:
            label = json.load(f)
    else:
        print("label doesn't exist", frame)
        return

    #print(label)
    for o in label['objs']:


        # if not 'annotator' in  o:
        #     continue

        # if o['annotator'] != '3dbox':
        #     continue
        
        if not 'obj_id' in o:
            continue

        box3d = None
        for i in objs:
             if i['obj_id'] == o['obj_id']:
                box3d = i
                break
        
        if not box3d:
            print('obj not found in 3d labels', o['obj_id'])
            continue
        # #print(o['obj_id'])
        # corners = box3d_to_corners(box3d)
        # #print(corners)
        # corners_img = proj_pts3d_to_img(corners, extrinsic, intrinsic, meta[camera_type][camera]['width'], meta[camera_type][camera]['height']) 
        # #print(corners_img)

        #if corners_img.shape[0] < 8:
        if True:
            #o['inside_corners'] = corners_img.shape[0]
            #print(frame, o['obj_id'], 'object stretch out camera')

            if pts is None:
                pts = load_points(scene, frame)
            
            box3d_pts = crop_pts(pts, box3d)
            box2d = gen_2dbox_for_obj_pts(box3d_pts, extrinsic, intrinsic, meta[camera_type][camera]['width'], meta[camera_type][camera]['height'])

            if box2d is not None:
                iou_score = iou(box2d, o['rect'])
                if iou_score < 0.5:
                    print(frame, o['obj_id'], iou_score) #, box2d, o['rect'])
                    o['annotator'] = '3dbox'
                    o['rect'] = box2d
            else:
                print(frame, o['obj_id'], 'gen 2dbox failed.')

        if True:
            if o['rect']['x2'] - o['rect']['x1'] < meta[camera_type][camera]['width'] * 0.005:
                print(frame, o['obj_id'], 'rect too small')
            if o['rect']['y2'] - o['rect']['y1'] < meta[camera_type][camera]['width'] * 0.005:
                print(frame, o['obj_id'], 'rect too small')


        # if corners_img.shape[0] == 0:
        #     print("rect points all out of image", o['obj_id'])
        #     continue        
        
        # corners_img = corners_img[:, 0:2]
        # p1 = np.min(corners_img, axis=0)
        # p2 = np.max(corners_img, axis=0)

        # rect = {
        #         "x1": p1[0],
        #         "y1": p1[1],
        #         "x2": p2[0],
        #         "y2": p2[1]
        #     }

        # if same_rect(o['rect'], rect):
        #     print('rect generated by 3d corners', o['obj_id'])
        #     o['annotator'] = '3dbox_corners'
        # else:
        #     #print('rect annotated by human', o['obj_id'])
        #     if 'annotator' in o:
        #         o.pop('annotator')

        
    #print(label)
    if args.save == 'yes':
        with open(label2d_file, 'w') as f:
            json.dump(label,f,indent=2)
        

def proc_frame(scene, meta, frame):

    #print(frame)

    # load 3d boxes
    label_3d_file = os.path.join(scene, 'label', frame+".json")
    if not os.path.exists(label_3d_file):
        print("label3d for", frame, 'does not exist')
        return

    with open(label_3d_file) as f:
        try:
            label_3d = json.load(f)
        except:
            print("error loading", label_3d_file)
            return

    boxes = label_3d
    #print(boxes)
    if 'objs' in boxes:
        boxes = boxes['objs']
        
    for camera_type in camera_types:
        for camera in camera_names:
            (extrinsic,intrinsic) = get_calib_for_frame(scene, meta, camera_type, camera, frame)
            proc_frame_camera(scene, meta, frame, camera_type, camera, extrinsic, intrinsic, boxes)
            




def proc_scene(scene):
    print(scene)
    scene_path = os.path.join(data_folder, scene)
    meta = read_scene_meta(scene_path)
    
    for frame in meta['frames']:
        if re.fullmatch(args.frames, frame):
            proc_frame(scene_path, meta, frame)


for s in scenes:
    proc_scene(s)
