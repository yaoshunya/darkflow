"""
Ingestor for KITTI formats.

http://www.cvlibs.net/datasets/kitti/eval_object.php

Per devkit docs:

All values (numerical or strings) are separated via spaces,
each row corresponds to one object. The 15 columns represent:

#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
0：物体の種類（Car, Van, Truck, Pedestrian, Person_sitting, Cyclist）
1：物体の画像からはみ出している割合（0は完全に見えている、1は完全にはみ出している）
2：オクルージョン状態（0：完全に見える、1：部分的に隠れている、2：大部分が隠れている、3：不明）
3：カメラから見た物体の向きα[-pi, pi]
4：2D bounding boxのminx
5：2D bounding boxのminy
6：2D bounding boxのmaxx
7：2D bounding boxのmaxy
8：3D object dimensionsの高さ（height）
9：3D object dimensionsの幅（width）
10：3D object dimensionsの奥行き（length）
11：3D 物体のx座標
12：3D 物体のy座標
13：3D 物体のz座標
14：カメラ座標系での物体の向きrotation_y [-pi..pi]


"""

import csv
import os
from PIL import Image
import shutil
import pdb
from converter import Ingestor, Egestor
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import final_prepare as fi
import numpy as np


class KITTIIngestor(Ingestor):
    def validate(self, path):
        expected_dirs = [
            'training/sphere_data',
            'training/label_2'
        ]
        for subdir in expected_dirs:
            if not os.path.isdir(f"{path}/{subdir}"):
                return False, f"Expected subdirectory {subdir} within {path}"
        if not os.path.isfile(f"{path}/train.txt"):
            return False, f"Expected train.txt file within {path}"
        return True, None

    def ingest(self, path):
        image_ids = self._get_image_ids(path)
        image_ext = 'png'
        if len(image_ids):
            first_image_id = image_ids[0]
            image_ext = self.find_image_ext(path, first_image_id)
        return [self._get_image_detection(path, image_name, image_ext=image_ext) for image_name in image_ids[:]]

    def find_image_ext(self, root, image_id):
        for image_ext in ['png', 'jpg']:
            if os.path.exists(f"{root}/training/image_2/{image_id}.{image_ext}"):
                return image_ext
        raise Exception(f"could not find jpg or png for {image_id} at {root}/training/image_2")

    def _get_image_ids(self, root):
        path = f"{root}/train.txt"
        with open(path) as f:
            return f.read().strip().split('\n')

    def _get_image_detection(self, root, image_id, *, image_ext='png'):
        detections_fpath = f"{root}/training/label_2/{image_id}.txt"
        detections = self._get_detections(detections_fpath)
        #detections = [det for det in detections if det['left'] < det['right'] and det['top'] < det['bottom']]
        image_path = f"{root}/training/image_2/{image_id}.{image_ext}"
        image_width, image_height = _image_dimensions(image_path)
        #pdb.set_trace()
        return {
            'image': {
                'id': image_id,
                'path': image_path,
                'segmented_path': None,
                'width': image_width,
                'height': image_height
            },
            'detections': detections
        }

    def prepare_mask():
    	return np.zeros((375,1242),dtype=int)




    def _get_detections(self, detections_fpath):
        detections = []
        with open(detections_fpath) as f:
            f_csv = csv.reader(f, delimiter=' ')
            for row in f_csv:
                x1, y1, x2, y2, height, width, length, X, Y, Z, rotation_y = map(float, row[4:15])
                label = row[0]
                """
                minx=int(x1)
                maxx=int(x2)
                miny=int(y1)
                maxy=int(y2)
                mask_prepare = np.zeros((375,1242),dtype=int)
                mask_prepare[miny:maxy,minx:maxx]=255
                mask_parts = np.array([])
                grid = fi.get_projection_grid(b=500)
                rot = fi.rand_rotation_matrix(deflection=1.0)
                grid = fi.rotate_grid(rot,grid)
                #pdb.set_trace()
                mask_parts = fi.project_2d_on_sphere(mask_prepare,grid)
                #pdb.set_trace()
                """
                detections.append({
                    'label': label,
                    'left': x1,
                    'right': x2,
                    'top': y1,
                    'bottom': y2,
                    #'mask':mask_parts
                })
                print(detections_fpath)
        return detections


def _image_dimensions(path):
    with Image.open(path) as image:
        return image.width, image.height

DEFAULT_TRUNCATED = 0.0 # 0% truncated
DEFAULT_OCCLUDED = 0    # fully visible

class KITTIEgestor(Egestor):

    def expected_labels(self):
        return {
            'Car': [],
            'Cyclist': ['biker'],
            'Misc': [],
            'Pedestrian': ['person'],
            'Person_sitting': [],
            'Tram': [],
            'Truck': [],
            'Van': [],
        }

    def egest(self, *, image_detections, root):
        images_dir = f"{root}/training/image_2"
        os.makedirs(images_dir, exist_ok=True)
        labels_dir = f"{root}/training/label_2"
        os.makedirs(labels_dir, exist_ok=True)

        id_file = f"{root}/train.txt"

        for image_detection in image_detections:
            image = image_detection['image']
            image_id = image['id']
            src_extension = image['path'].split('.')[-1]
            shutil.copyfile(image['path'], f"{images_dir}/{image_id}.{src_extension}")

            with open(id_file, 'a') as out_image_index_file:
                out_image_index_file.write(f'{image_id}\n')

            out_labels_path = f"{labels_dir}/{image_id}.txt"
            with open(out_labels_path, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)

                for detection in image_detection['detections']:
                    kitti_row = [-1] * 15
                    kitti_row[0] = detection['label']
                    kitti_row[1] = DEFAULT_TRUNCATED
                    kitti_row[2] = DEFAULT_OCCLUDED
                    x1 = detection['left']
                    x2 = detection['right']
                    y1 = detection['top']
                    y2 = detection['bottom']
                    kitti_row[4:8] = x1, y1, x2, y2
                    #pdb.set_trace()
                    csvwriter.writerow(kitti_row)
