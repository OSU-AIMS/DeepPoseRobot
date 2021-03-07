import os
import cv2
import robotpose.utils as utils
import numpy as np
from tqdm import tqdm
import json
import pyrealsense2 as rs
import open3d as o3d
import pickle
from robotpose import paths as p
from .segmentation import RobotSegmenter
import time
import datetime
from deepposekit.io import initialize_dataset


dataset_version = 1.0
"""
Version 1.0: 3/7/2021
    Began versioning.
    Compatible versions should include the same integer base (eg 1.0 and 1.4).
    Backwards-Incompatiblity should be marked by a new integer base (eg going from 1.4 to 2.0)

"""



def build(data_path, dest_path = None):

    build_start_time = time.time()

    if dest_path is None:
        name = os.path.basename(os.path.normpath(data_path))
        dest_path = os.path.join(p.datasets, name)

    # Make dataset folder if it does not already exist
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)

    # Build lists of files
    jsons = [x for x in os.listdir(data_path) if x.endswith('.json')]
    plys = [x for x in os.listdir(data_path) if x.endswith('full.ply')]
    imgs = [x for x in os.listdir(data_path) if x.endswith('og.png')]

    # Make sure overall dataset length is the same for each file type
    length = len(imgs)
    assert len(jsons) == len(plys) == length, "Unequal number of images, jsons, or plys"


    """
    Parse Images
    """

    # Get image dims
    img = cv2.imread(os.path.join(data_path,imgs[0]))
    img_height = img.shape[0]
    img_width = img.shape[1]

    # Create image array
    orig_img_arr = np.zeros((length, img_height, img_width, 3), dtype=np.uint8)

    # Get paths for each image
    orig_img_path = [os.path.join(data_path, x) for x in imgs]

    # Store images in array
    for idx, path in tqdm(zip(range(length), orig_img_path),desc="Parsing 2D Images"):
        orig_img_arr[idx] = cv2.imread(path)

    # Save array
    np.save(os.path.join(dest_path, 'og_img.npy'), orig_img_arr)

    # Save as a video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(dest_path,"og_vid.avi"),fourcc, 15, (img_width,img_height))
    for idx in range(length):
        out.write(orig_img_arr[idx])
    out.release()

    segmenter = RobotSegmenter()
    segmented_img_arr = np.zeros((length, segmenter.height(), segmenter.width(), 3), dtype=np.uint8)
    ply_data = []

    times = np.array([0,0,0,0,0,0,0], dtype=np.float64)
    # Segment images and PLYS
    for idx in tqdm(range(length),desc="Segmenting"):
        ply_path = os.path.join(data_path,plys[idx])
        segmented_img_arr[idx,:,:,:], ply = segmenter.segmentImage(orig_img_arr[idx], ply_path)
        ply_data.append(ply)


    # Save segmented image array
    np.save(os.path.join(dest_path, 'seg_img.npy'), segmented_img_arr)

    # Save as a video (just for reference)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(dest_path,"seg_vid.avi"),fourcc, 15, (segmenter.width(),segmenter.height()))
    for idx in range(length):
        out.write(segmented_img_arr[idx])
    out.release()

    """
    Process PLY data
    """

    ply_data_nd = np.array(ply_data, dtype=object)
    np.save(os.path.join(dest_path,'ply.npy'),ply_data_nd)


    """
    Parse JSONs
    """
    json_path = [os.path.join(data_path, x) for x in jsons]
    json_arr = np.zeros((length, 6), dtype=float)

    for idx, path in tqdm(zip(range(length), json_path), desc="Parsing JSON Joint Angles"):
        # Open file
        with open(path, 'r') as f:
            d = json.load(f)
        d = d['objects'][0]['joint_angles']

        # Put data in array
        for sub_idx in range(6):
            json_arr[idx,sub_idx] = d[sub_idx]['angle']

    # Save JSON data as npy
    np.save(os.path.join(dest_path, 'ang.npy'), json_arr)

    """
    Write dataset info file
    """
    # Make json info file
    info = {
        "ds_ver": dataset_version,
        "name": os.path.basename(os.path.normpath(dest_path)),
        "frames": length,
        "build_time": time.time() - build_start_time,
        "last_build": str(datetime.datetime.now())
    }

    with open(os.path.join(dest_path,'ds.json'),'w') as file:
        json.dump(info, file)





class Dataset():
    def __init__(self, name, skeleton=None, load_seg = True, load_og = False, primary = "seg"):
        
        self.load_seg = load_seg
        self.load_og = load_og
        # Search for dataset with correct name
        datasets = [ f.path for f in os.scandir(p.datasets) if f.is_dir() ]
        names = [ os.path.basename(os.path.normpath(x)) for x in datasets ]
        ds_found = False
        for ds, nm in zip(datasets, names):
            if name in nm:
                if self.validate(ds):
                    self.path = ds
                    self.name = nm
                    ds_found = True
                    break
                else:
                    print("\nDataset Incomplete.")
                    print(f"Recompiling:\n")
                    self.build(os.path.join(os.path.join(p.datasets,'raw'),nm))
                    self.path = ds
                    self.name = nm
                    ds_found = True
                    break


        # If no dataset was found, try to find one to build
        if not ds_found:
            datasets = [ f.path for f in os.scandir(os.path.join(p.datasets,'raw')) if f.is_dir() ]
            names = [ os.path.basename(os.path.normpath(x)) for x in datasets ]
            for ds, nm in zip(datasets, names):
                if name in nm:
                    print("\nNo matching compiled dataset found.")
                    print(f"Compiling from {ds}:\n")
                    self.build(ds)
                    self.path = os.path.join(p.datasets, nm)
                    self.name = nm
                    ds_found = True
                    break
        
        # Make sure a dataset was found
        assert ds_found, f"No matching dataset found for '{name}'"

        # Load dataset
        self.load(skeleton)

        # Set paths, resolution
        if self.load_og:
            self.og_vid_path = os.path.join(self.path, 'og_vid.avi')
            self.resolution_og = self.og_img.shape[1:3]
            self.resolution = self.resolution_og
        if self.load_seg:
            self.seg_vid_path = os.path.join(self.path, 'seg_vid.avi')
            self.resolution_seg = self.seg_img.shape[1:3]
            self.resolution = self.resolution_seg


        # Set primary image and video types
        if self.load_seg and not self.load_og:
            primary = "seg"
        if self.load_og and not self.load_seg:
            primary = "og"

        if primary == "og":
            self.img = self.og_img
            self.vid = self.og_vid
            self.vid_path = self.og_vid_path
        else:
            self.img = self.seg_img
            self.vid = self.seg_vid
            self.vid_path = self.seg_vid_path
            if primary != "seg":
                print("Invalid primary media type selected.\nUsing seg.")



    def load(self, skeleton=None):
        print("\nLoading Dataset...")
        # Read into JSON to get dataset settings
        with open(os.path.join(self.path, 'ds.json'), 'r') as f:
            d = json.load(f)

        self.length = d['frames']

        # Read in og images
        if self.load_og:
            print("Reading orig images...")
            self.og_img = np.load(os.path.join(self.path, 'og_img.npy'))
            self.og_vid = cv2.VideoCapture(os.path.join(self.path, 'og_vid.avi'))

        # Read in seg images
        if self.load_seg:
            print("Reading segmented images...")
            self.seg_img = np.load(os.path.join(self.path, 'seg_img.npy'))
            self.seg_vid = cv2.VideoCapture(os.path.join(self.path, 'seg_vid.avi'))
            

        # Read angles
        print("Reading joint angles...")
        self.angles = np.load(os.path.join(self.path, 'ang.npy'))

        # Read in point data
        print("Reading 3D data...")
        ply_in = np.load(os.path.join(self.path, 'ply.npy'),allow_pickle=True)
        self.ply = []
        for entry in ply_in:
            entry = np.array(entry)
            self.ply.append(entry)

        # Set deeppose dataset path
        self.deepposeds_path = os.path.join(self.path,'deeppose.h5')

        # If a skeleton is set, change paths accordingly
        if skeleton is not None:
            self.setSkeleton(skeleton)
        
        print("Dataset Loaded.\n\n")


    def validate(self, path):
        ang = os.path.isfile(os.path.join(path,'ang.npy'))
        ds = os.path.isfile(os.path.join(path,'ds.json'))
        ply = os.path.isfile(os.path.join(path,'ply.npy'))
        seg_img = os.path.isfile(os.path.join(path,'seg_img.npy'))
        og_img = os.path.isfile(os.path.join(path,'og_img.npy'))
        seg_vid = os.path.isfile(os.path.join(path,'seg_vid.avi'))
        og_vid = os.path.isfile(os.path.join(path,'og_vid.avi'))
        
        with open(os.path.join(path, 'ds.json'), 'r') as f:
            d = json.load(f)

        try:
            if int(d['ds_ver']) != int(dataset_version):
                print(f"Dataset Out of Date:\n\tDataset version:{d['ds_ver']}\n\tCurrent version: {dataset_version}")
                return False
        except KeyError:
            print(f"Dataset Out of Date:\n\tDataset version: Unversioned\n\tCurrent version: {dataset_version}")
            return False
            

        return ang and ds and ply and seg_img and og_img and seg_vid and og_vid


    def build(self,data_path):
        build(data_path)

    def setSkeleton(self,skeleton_name):
        for file in [x for x in os.listdir(p.skeletons) if x.endswith('.csv')]:
            if skeleton_name in os.path.splitext(file)[0]:
                self.skeleton = os.path.splitext(file)[0]
                self.skeleton_path = os.path.join(p.skeletons, file)
                self.deepposeds_path = self.deepposeds_path.replace('.h5','_'+os.path.splitext(file)[0]+'.h5')

    def makeDeepPoseDS(self, force=False):
        if force:
            initialize_dataset(
                images=self.seg_img,
                datapath=self.deepposeds_path,
                skeleton=self.skeleton_path,
                overwrite=True # This overwrites the existing datapath
            )
            return

        if not os.path.isfile(self.deepposeds_path):
            initialize_dataset(
                images=self.seg_img,
                datapath=self.deepposeds_path,
                skeleton=self.skeleton_path,
                overwrite=False # This overwrites the existing datapath
            )
        else:
            print("Using Precompiled Deeppose Dataset")


    def __len__(self):
        if self.length is None:
            return 0
        else:
            return self.length  

    def __repr__(self):
        return f"RobotPose dataset of {self.length} frames. Using skeleton {self.skeleton}"

    def og(self):
        return self.load_og

    def seg(self):
        return self.load_seg
