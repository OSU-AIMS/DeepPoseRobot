# DeepPoseRobot, an implementation of DeepPoseKit
<p align="center">
<img src="https://github.com/AdamExley/DeepPoseRobot/blob/main/assets/video_overlay_rebound.gif" height="200px">
<img src="https://github.com/AdamExley/DeepPoseRobot/blob/segmentation_dataset/assets/video_overlay_new.gif" height="200px">
</p>

This is an adaptation of both [DeepPoseKit](deepposekit.org)  and [PixelLib](https://github.com/ayoolaolafenwa/PixelLib) to predict robot joint angles.

Visualization uses the [Turbo Colormap](https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html).

The robot is isolated from the background using PixelLib and then the keypoint locations of the robot are predicted using a DeepPoseKit model.


# Installation

This requires [Tensorflow](https://github.com/tensorflow/tensorflow) for both segmentation and pose estimation. [Tensorflow](https://github.com/tensorflow/tensorflow) should be manually installed, along with CUDA and cuDNN as follows:

- [Tensorflow Installation Instructions](https://www.tensorflow.org/install)
- Any Tensorflow version >=2.0.0 should be compatible.
    - Tensorflow-gpu 2.0.0 is currently the only tested version.

## Installing with Anaconda on Windows

To use DeepPoseKit on Windows, you must first manually install `Shapely`, one of the dependencies for the [imgaug package](https://github.com/aleju/imgaug):
```bash
conda install -c conda-forge shapely
```

Install requirements with pip:
```bash
pip install --update --r requirements.txt
```
Sometimes Pixellib will not work after all installations have been complted. To fix this error, upgrade and downgrade Tensorflow.
```bash
pip install --update tensorflow-gpu
pip install --update tensorfow-gpu==2.0.0
```


# License

Released under a Apache 2.0 License. See [LICENSE](https://github.com/jgraving/deepposekit/blob/master/LICENSE) for details.
