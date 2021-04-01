# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley


import datetime
import json
import multiprocessing as mp
import numpy as np
import os
import time

import cv2
import h5py
from tqdm import tqdm

from .multithread import crop
from . import paths as p
from .segmentation import RobotSegmenter
from .utils import workerCount


def save_video(path, img_arr):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path,fourcc, 15, (img_arr.shape[2],img_arr.shape[1]))
    for img in img_arr:
        out.write(img)
    out.release()


class Builder():
    def __init__(self):
        self.build_start_time = time.time()

    def build_full(self, data_path, dataset_ver, name = None):
        self._set_dest_path(data_path, name)
        self._get_filepaths_from_data_dir(data_path)
        self._load_json_data()
        self._load_imgs_and_depthmaps()
        self._segment_images_and_maps()
        self._save_reference_videos()
        return self._save_full(dataset_ver)

    def build_subset(self, src, sub_type, idxs):
        self._read_full(src)
        dst = src.replace('.h5',f'_{sub_type}.h5')
        self._write_subset(dst, sub_type, idxs)



    def _set_dest_path(self, data_path, name):
        if name is None:
            name = os.path.basename(os.path.normpath(data_path))
        self.dest_path = os.path.join(p.DATASETS, name)
        self.name = name
        if not os.path.isdir(self.dest_path):
            os.mkdir(self.dest_path)

    def _get_filepaths_from_data_dir(self, data_path):
        self.jsons_p = [os.path.join(r,x) for r,d,y in os.walk(data_path) for x in y if x.endswith('.json')]
        self.maps_p = [os.path.join(r,x) for r,d,y in os.walk(data_path) for x in y if x.endswith('.npy')]
        self.imgs_p = [os.path.join(r,x) for r,d,y in os.walk(data_path) for x in y if x.endswith('.png')]

        self.jsons = [x.replace(data_path,'') for x in self.jsons_p]
        self.maps = [x.replace(data_path,'') for x in self.maps_p]
        self.imgs = [x.replace(data_path,'') for x in self.imgs_p]

        # Make sure overall dataset length is the same for each file type
        self.length = len(self.imgs)
        assert len(self.jsons) == len(self.maps) == self.length, "Unequal number of images, jsons, or maps"


    def _load_json_data(self):
        self.ang_arr = np.zeros((self.length, 6), dtype=float)
        self.pos_arr = np.zeros((self.length, 6, 3), dtype=float)
        depth_scale = set()
        intrin_depth = set()
        intrin_color = set()

        # Parse JSONs
        for idx, path in tqdm(zip(range(self.length), self.jsons_p), total=self.length, desc="Parsing JSONs"):
            with open(path, 'r') as f:
                d = json.load(f)

            depth_scale.add(d['realsense_info'][0]['depth_scale'])
            intrin_depth.add(d['realsense_info'][0]['intrin_depth'])
            intrin_color.add(d['realsense_info'][0]['intrin_color'])

            d = d['objects'][0]['joints']

            for sub_idx in range(6):
                self.ang_arr[idx,sub_idx] = d[sub_idx]['angle']
                self.pos_arr[idx,sub_idx] = d[sub_idx]['position']

        assert len(depth_scale) == len(intrin_depth) ==  len(intrin_color) == 1,f'Camera settings must be uniform over the dataset.'

        self.depth_scale = depth_scale.pop()
        self.intrin_depth = intrin_depth.pop()
        self.intrin_color = intrin_color.pop()


    def _load_imgs_and_depthmaps(self):
        img = cv2.imread(self.imgs_p[0])
        self.img_height, self.img_width = img.shape[0:2]

        # Create image array
        self.orig_img_arr = np.zeros((self.length, self.img_height, self.img_width, 3), dtype=np.uint8)
        self.depthmap_arr = np.zeros((self.length, self.img_height, self.img_width), dtype=np.float64)

        for idx, path in tqdm(zip(range(self.length), self.imgs_p),total=self.length,desc="Parsing 2D Images"):
            self.orig_img_arr[idx] = cv2.imread(path)
        for idx, path in tqdm(zip(range(self.length), self.maps_p),total=self.length,desc="Parsing Depthmaps"):
            self.depthmap_arr[idx] = np.load(path)

        self.depthmap_arr *= self.depth_scale

    
    def _segment_images_and_maps(self):
        segmenter = RobotSegmenter()
        self.segmented_img_arr = np.zeros((self.length, segmenter.height(), segmenter.width(), 3), dtype=np.uint8)
        self.pointmap = np.zeros((self.length, segmenter.height(), segmenter.width(), 3), dtype=np.float64)
        self.mask_arr = np.zeros((self.length, self.img_height, self.img_width), dtype=bool)
        self.rois = np.zeros((self.length, 4))

        # Segment images
        for idx in tqdm(range(self.length),desc="Segmenting Images",colour='red'):
            self.mask_arr[idx], self.rois[idx] = segmenter.segmentImage(self.orig_img_arr[idx])
        self.rois = self.rois.astype(int)

        del segmenter

        batch_size = 100
        print(f"Using Crop Pool of size {batch_size} with {workerCount()} workers.")
        with tqdm(total=self.length, desc="Cropping Data") as pbar:
            for start in range(0,self.length, batch_size):

                if start + batch_size >= self.length:
                    batch_size = self.length - start

                # Make iterable for pool
                crop_inputs = []
                for idx in range(start,start+batch_size):
                    crop_inputs.append((self.depthmap_arr[idx], self.orig_img_arr[idx], self.mask_arr[idx], self.rois[idx]))

                # Run pool to segment PLYs
                with mp.Pool(workerCount()) as pool:
                    crop_outputs = pool.starmap(crop, crop_inputs)

                for idx in range(start,start+batch_size):
                    self.segmented_img_arr[idx] = crop_outputs[idx-start][0]
                    self.pointmap[idx] = crop_outputs[idx-start][1]
                pbar.update(batch_size)

    def _save_reference_videos(self):
        save_video(os.path.join(self.dest_path,"og_vid.avi"), self.orig_img_arr)
        save_video(os.path.join(self.dest_path,"seg_vid.avi"), self.segmented_img_arr)

    def _save_full(self, ver):
        dest = os.path.join(self.dest_path, self.name + '.h5')
        with tqdm(total=10, desc="Writing Dataset") as pbar:
            with h5py.File(dest,'a') as file:
                file.attrs['name'] = self.name
                file.attrs['version'] = ver
                file.attrs['length'] = self.length
                file.attrs['build_date'] = str(datetime.datetime.now())
                file.attrs['compile_date'] = str(datetime.datetime.now())
                file.attrs['compile_time'] = time.time() - self.build_start_time
                file.attrs['type'] = 'full'
                file.attrs['original_resolution'] = self.orig_img_arr[0].shape
                file.attrs['segmented_resolution'] = self.segmented_img_arr[0].shape
                file.attrs['depth_intrinsics'] = self.intrin_depth
                file.attrs['color_intrinsics'] = self.intrin_color
                file.attrs['depth_scale'] = self.depth_scale
                file.create_dataset('angles', data = self.ang_arr, compression="gzip")
                pbar.update(1)
                file.create_dataset('positions', data = self.pos_arr, compression="gzip")
                pbar.update(1)
                coord_grop = file.create_group('coordinates')
                dm = coord_grop.create_dataset('depthmaps', data = self.depthmap_arr, compression="gzip")
                pbar.update(1)
                dm.attrs['depth_scale'] = self.depth_scale
                coord_grop.create_dataset('pointmaps', data = self.pointmap, compression="gzip")
                pbar.update(1)
                img_grp = file.create_group('images')
                img_grp.create_dataset('original', data = self.orig_img_arr, compression="gzip")
                pbar.update(1)
                img_grp.create_dataset('segmented', data = self.segmented_img_arr, compression="gzip")
                pbar.update(1)
                img_grp.create_dataset('rois', data = self.rois, compression="gzip")
                pbar.update(1)
                path_grp = file.create_group('paths')
                path_grp.create_dataset('jsons', data = np.array(self.jsons, dtype=h5py.string_dtype()), compression="gzip")
                pbar.update(1)
                path_grp.create_dataset('depthmaps', data = np.array(self.maps, dtype=h5py.string_dtype()), compression="gzip")
                pbar.update(1)
                path_grp.create_dataset('images', data = np.array(self.imgs, dtype=h5py.string_dtype()), compression="gzip")
                pbar.update(1)

        return dest

    def _read_full(self, path):
        with h5py.File(path,'r') as file:
            self.attrs = file.attrs
            self.name = file.attrs['name']
            self.length = file.attrs['length']

            self.intrin_depth = file.attrs['depth_intrinsics']
            self.intrin_color = file.attrs['color_intrinsics']
            self.depth_scale = file.attrs['depth_scale']
            self.ang_arr = file['angles']
            self.pos_arr = file['positions']
            self.depthmap_arr = file['coordinates/depthmaps']
            self.pointmap = file['coordinates/pointmaps']

            self.orig_img_arr = file['images/original']
            self.segmented_img_arr = file['images/segmented']
            self.rois = file['images/rois']

            self.jsons = file['paths/jsons']
            self.maps = file['paths/depthmaps']
            self.imgs = file['paths/images']

    def _write_subset(self,path,sub_type,idxs):
        """Create a derivative dataset from a full dataset, using a subset of the data."""

        with h5py.File(path,'a') as file:
            file.attrs = self.attrs
            file.attrs['length'] = len(idxs)
            file.attrs['compile_date'] = str(datetime.datetime.now())
            file.attrs['compile_time'] = 0
            file.attrs['type'] = sub_type
            file.create_dataset('angles', data = self.ang_arr[idxs], compression="gzip")
            file.create_dataset('positions', data = self.pos_arr[idxs], compression="gzip")
            coord_grop = file.create_group('coordinates')
            dm = coord_grop.create_dataset('depthmaps', data = self.depthmap_arr[idxs], compression="gzip")
            dm.attrs['depth_scale'] = self.depth_scale
            coord_grop.create_dataset('pointmaps', data = self.pointmap[idxs], compression="gzip")
            img_grp = file.create_group('images')
            img_grp.create_dataset('original', data = self.orig_img_arr[idxs], compression="gzip")
            img_grp.create_dataset('segmented', data = self.segmented_img_arr[idxs], compression="gzip")
            img_grp.create_dataset('rois', data = self.rois[idxs], compression="gzip")
            path_grp = file.create_group('paths')
            path_grp.create_dataset('jsons', data = np.array(self.jsons[idxs], dtype=h5py.string_dtype()), compression="gzip")
            path_grp.create_dataset('depthmaps', data = np.array(self.maps[idxs], dtype=h5py.string_dtype()), compression="gzip")
            path_grp.create_dataset('images', data = np.array(self.imgs[idxs],dtype=h5py.string_dtype()), compression="gzip")

        
    def weld(self, path_a, path_b, dst_dir, name):
        a = h5py.File(path_a,'r')
        b = h5py.File(path_b,'r')
        dst = h5py.File(path_a,'r')

        a_attrs = a.attrs
        b_attrs = b.attrs

        for attribute in ['version','original_resolution','segmented_resolution','depth_intrinsics','color_intrinsics','depth_scale']:
            assert a_attrs[attribute] == b_attrs[attribute], f"{attribute} must be equal to join datasets"

        a_len = a.attrs['length']
        b_len = b.attrs['length']
        self.length = a_len + b_len

        self.name = name
        self.dest_path = dst_dir

        self.ang_arr = np.vstack((a['angles'],b['angles']))
        self.pos_arr = np.vstack((a['positions'],b['positions']))
        self.depthmap_arr = np.vstack((a['coordinates/depthmaps'],b['coordinates/depthmaps']))
        self.pointmap = np.vstack((a['coordinates/pointmaps'],b['coordinates/pointmaps']))
        self.orig_img_arr = np.vstack((a['images/original'],b['images/original']))
        self.segmented_img_arr = np.vstack((a['images/segmented'],b['images/segmented']))
        self.rois = np.vstack((a['images/rois'],b['images/rois']))
        self.jsons = np.vstack((a['paths/jsons'],b['paths/jsons']))
        self.maps = np.vstack((a['paths/depthmaps'],b['paths/depthmaps']))
        self.imgs = np.vstack((a['paths/images'],b['paths/images']))

        self._save_full()