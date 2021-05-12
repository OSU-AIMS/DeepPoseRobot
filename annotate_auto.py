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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf

from robotpose.autoAnnotate import AutomaticKeypointAnnotator, AutomaticSegmentationAnnotator
from robotpose.render import DatasetRenderer


def label(args):
    rend = DatasetRenderer(args.dataset, args.skeleton)
    if not args.no_key:
        key = AutomaticKeypointAnnotator(args.dataset, args.skeleton, renderer = rend, preview = not args.no_preview)
        key.run()
        del key
    if not args.no_seg:
        if args.per_joint:
            mode = 'seg'
        else:
            mode = 'seg_full'
        seg = AutomaticSegmentationAnnotator(args.dataset, args.skeleton, renderer = rend, preview = not args.no_preview, mode=mode)
        seg.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default="set10", help="The dataset to load to annotate. Can be a partial name.")
    parser.add_argument('skeleton', type=str, default="B", help="The skeleton to use for annotation.")
    parser.add_argument('-no_preview', action="store_true", help="Disables preview.")
    parser.add_argument('-no_seg', action="store_true", help="Disables segmentation annotation.")
    parser.add_argument('-no_key', action="store_true", help="Disables keypoint annotation.")
    parser.add_argument('-per_joint', action="store_true", help="Labels segmentation per-joint.")
    args = parser.parse_args()
    label(args)