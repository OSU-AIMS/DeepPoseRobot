# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import argparse
import os
import random
import requests
import shutil

from pixellib.custom_train import instance_custom_training

from robotpose import Dataset
from robotpose.paths import Paths as p


def train(dataset, batch, valid, classes):
    ds = Dataset(dataset)

    print("Splitting up data...")
    # Split set into validation and train
    for folder in ['train', 'test']:
        path = os.path.join(ds.seg_anno_path, folder)
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)

    jsons = [x for x in os.listdir(ds.seg_anno_path) if x.endswith('.json') and 'test' not in x and 'train' not in x]
    random.shuffle(jsons)

    valid_size = int(len(jsons) * valid)
    valid_list = jsons[:valid_size]
    train_list = jsons[valid_size:]

    for lst, folder in zip([valid_list, train_list],['test','train']):
        path = os.path.join(ds.seg_anno_path, folder)
        for file in lst:
            shutil.copy2(os.path.join(ds.seg_anno_path, file), os.path.join(path, file))
            shutil.copy2(os.path.join(ds.seg_anno_path, file.replace('.json','.png')), os.path.join(path, file.replace('.json','.png')))

    print("Data Split.")
    default_model_path = r'models/segmentation/mask_rcnn_coco.h5'

    if not os.path.isfile(default_model_path):
        print("Base model not found.\nDownloading...")
        url = "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5"
        r = requests.get(url, allow_redirects=True)
        open(default_model_path, 'wb').write(r.content)

    train_maskrcnn = instance_custom_training()
    train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes = classes, batch_size = batch)
    train_maskrcnn.load_pretrained_model(default_model_path)


    pth = os.path.abspath(p().SEG_MODELS)     
    if classes > 1:
        pth = os.path.join(os.path.abspath(p().SEG_MODELS)  ,'multi')
    #Train
    train_maskrcnn.load_dataset(ds.seg_anno_path)
    train_maskrcnn.train_model(num_epochs = 300, augmentation=True,  path_trained_models = pth)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default="set6", help="The dataset to load to annotate. Can be a partial name.")
    parser.add_argument('-batch',type=int, choices=[1,2,4,8,12,16], default=2, help="Batch size for training")
    parser.add_argument('-valid',type=float, default=.2, help="Validation size for training")
    parser.add_argument('-classes',type=int, default=1, help="Class number for training")
    args = parser.parse_args()

    train(args.dataset, args.batch, args.valid, args.classes)