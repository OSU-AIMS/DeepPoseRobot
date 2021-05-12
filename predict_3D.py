# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from deepposekit.models import load_model
from deepposekit.io import VideoReader
import cv2
import matplotlib.pyplot as plt
import numpy as np
from robotpose.utils import *
from robotpose.paths import Paths as p
from robotpose import Dataset
from robotpose.utils import reject_outliers_iqr
from robotpose.turbo_colormap import color_array
from tqdm import tqdm
import json

from robotpose.angle_prediction import Predictor

setMemoryGrowth()

predict = False
save = False
skele = 'E'
ds = Dataset('set10',skele)

if predict:
    print("Predicting...")
    # Load model, make predictions
    model = load_model(os.path.join(os.getcwd(),fr'models\set10__{skele}__StackedDensenet.h5'))
    reader = VideoReader(ds.seg_vid_path)
    predictions = model.predict(reader)
    print("Finished Predicting.")

    if save:
        np.save(f'output/predictions_{skele}.npy',np.array(predictions))
        print("Predictions saved")
else:
    predictions = np.load(f'output/predictions_{skele}.npy')

# Load video capture and make output
cap = cv2.VideoCapture(ds.seg_vid_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(p().VIDEO.replace(".avi","_overlay.avi"),fourcc, 20, (ds.seg_resolution[1]*2,ds.seg_resolution[0]))

ret, image = cap.read()

frame_height = image.shape[0]
frame_width = image.shape[1]

tim = Predictor(skele)
print("Copying pointmaps...")
pointmaps = np.copy(ds.pointmaps)
print("Pointmaps Copied.")
with tqdm(total=ds.length) as pbar:
    i = 0
    while ret:
        over = np.zeros((ds.seg_resolution[0],ds.seg_resolution[1],3),dtype=np.uint8)

        tim.load(predictions[i], pointmaps[i])
        pred = tim.predict()

        # Put depth info on overlay
        over = color_array(pointmaps[i,...,2])
        #Visualize lines
        image = tim.visualize(image)
        over = tim.visualize(over)

        dual = np.zeros((frame_height,frame_width*2,3),dtype=np.uint8)
        dual[:,0:frame_width] = image
        dual[:,frame_width:frame_width*2] = over

        out.write(dual)
        cv2.imshow("Angle Predictions",dual)
        cv2.waitKey(1)
        i+=1
        ret, image = cap.read()
        pbar.update(1)

cv2.destroyAllWindows()
cap.release()
out.release()

a = Grapher(['S','L','U'],tim.prediction_history,np.copy(ds.angles),tim.full_prediction_history)
a.plot()
a.plot(20)
a.plotJoint('S',20)
a.plotJoint('L',20)
a.plotJoint('U',20)