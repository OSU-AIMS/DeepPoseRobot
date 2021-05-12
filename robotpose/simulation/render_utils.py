import numpy as np
import os
import json

import pyrender
import trimesh

from ..paths import Paths as p
from ..urdf import URDFReader


MESH_CONFIG = os.path.join(p().DATASETS,'mesh_config.json')

DEFAULT_COLORS = [
    [0  , 0  , 85 ],[0  , 0  , 170],[0  , 0  , 255],
    [0  , 85 , 0  ],[0  , 170, 0  ],[0  , 255, 0  ],
    [85 , 0  , 0  ],[170, 0  , 0  ],[255, 0  , 0  ],
    [0  , 85 , 85 ],[85 , 0  , 85 ],[85 , 85  , 0 ],
    [0  , 170, 170],[170, 0  , 170],[170, 170 , 0 ],
    [0  , 255, 255],[255, 0  , 255],[255, 255 , 0 ],
    [170, 85 , 85 ],[85 , 170, 85 ],[85 , 85 , 170],
    [255, 85 , 85 ],[85 , 255, 85 ],[85 , 85 , 255],
    [255, 170, 170],[170, 255, 170],[170, 170, 255],
    [85 , 170, 170],[170, 85 , 170],[170, 170, 85 ],
    [85 , 255, 255],[255, 85 , 255],[255, 255, 85 ],
    [85 , 170, 255],[255, 85 , 170],[170, 255, 85 ],
    [85 , 85 , 85]
]



class MeshLoader():

    def __init__(self):

        if not os.path.isfile(MESH_CONFIG):
            info = {}
            default_pose = [0,0,0,0,0,-np.pi/2]
            links = ['BASE','S','L','U','R','B','T']
            for link in links:
                info[link] = {"pose":default_pose}
            with open(MESH_CONFIG,'w') as f:
                json.dump(info, f, indent=4)

        with open(MESH_CONFIG,'r') as f:
            d = json.load(f)

        urdf_reader = URDFReader()

        self.name_list = urdf_reader.mesh_names
        self.mesh_list = urdf_reader.meshes[:-1]
        self.pose_list = [d[x]['pose'] for x in d.keys()]

    def load(self):
        self.meshes = []
        for file, pose in zip(self.mesh_list, self.pose_list):
            tm = trimesh.load(file)
            self.meshes.append(pyrender.Mesh.from_trimesh(tm,smooth=True, poses=makePose(*pose)))

    def getMeshes(self):
        return self.meshes

    def getNames(self):
        return self.name_list



def cameraFromIntrinsics(rs_intrinsics):
    """Returns Pyrender Camera.
    Makes a Pyrender camera from realsense intrinsics
    """
    return pyrender.IntrinsicsCamera(cx=rs_intrinsics.ppx, cy=rs_intrinsics.ppy, fx=rs_intrinsics.fx, fy=rs_intrinsics.fy)


def angToPoseArr(yaw,pitch,roll, arr = None):
    """Returns 4x4 pose array.
    Converts rotations to a pose array
    """
    # Takes pitch, roll, yaw and converts into a pose arr
    angs = np.array([yaw,pitch,roll])
    c = np.cos(angs)
    s = np.sin(angs)
    if arr is None:
        pose = np.zeros((4,4))
    else:
        pose = arr

    pose[0,0] = c[0] * c[1]
    pose[1,0] = c[1] * s[0]
    pose[2,0] = -1 * s[1]

    pose[0,1] = c[0] * s[1] * s[2] - c[2] * s[0]
    pose[1,1] = c[0] * c[2] + np.prod(s)
    pose[2,1] = c[1] * s[2]

    pose[0,2] = s[0] * s[2] + c[0] * c[2] * s[1]
    pose[1,2] = c[2] * s[0] * s[1] - c[0] * s[2]
    pose[2,2] = c[1] * c[2]

    pose[3,3] = 1.0

    return pose
    

def translatePoseArr(x,y,z, arr = None):
    """Returns 4x4 pose array.
    Translates a pose array
    """
    if arr is None:
        pose = np.zeros((4,4))
    else:
        pose = arr

    pose[0,3] = x
    pose[1,3] = y
    pose[2,3] = z

    return pose


def makePose(x,y,z,pitch,roll,yaw):
    """Returns 4x4 pose array.
    Makes pose array given positon and angle
    """
    pose = angToPoseArr(yaw,pitch,roll)
    pose = translatePoseArr(x,y,z,pose)
    return pose



def posesFromData(ang, pos):
    """
    Returns Zx6x4x4 array of pose arrays
    """
    #Make arr in x,y,z,roll,pitch,yaw format
    coord = np.zeros((pos.shape[0],6,6))
    # Yaw of S-R is just S angle
    for idx in range(1,5):
        coord[:,idx,5] = ang[:,0]

    coord[:,1:6,0:3] = pos[:,:5]    # Set xyz translations

    coord[:,2,3] = ang[:,1]             # Pitch of L
    coord[:,3,3] = ang[:,1] - ang[:,2]  # Pitch of U
    coord[:,4,3] = ang[:,1] - ang[:,2]  # Pitch of R       

    coord[:,4,4] = -ang[:,3]    # Roll of R


    poses = np.zeros((coord.shape[0],6,4,4))    # X frames, 6 joints, 4x4 pose for each
    for idx in range(coord.shape[0]):
        for sub_idx in range(5):
            poses[idx,sub_idx] = makePose(*coord[idx,sub_idx])

    # Determine BT with vectors because it's easier

    bt = pos[:,5] - pos[:,4]    # Vectors from B to T

    y = poses[:,4,:3,1] # Y Axis is common with R joint
    z = np.cross(bt, y)

    z = z/np.vstack([np.linalg.norm(z, axis=-1)]*3).transpose()
    x = bt/np.vstack([np.linalg.norm(bt, axis=-1)]*3).transpose()

    b_poses = np.zeros((pos.shape[0],4,4))

    b_poses[:,:3,0] = x # X Unit
    b_poses[:,:3,1] = y # Y Unit
    b_poses[:,:3,2] = z # Z Unit
    b_poses[:,3,3] = 1
    b_poses[:,:3,3] = pos[:,4] # XYZ Offset

    poses[:,-1,:] = b_poses

    return poses



def setPoses(scene, nodes, poses):
    """
    Set all the poses of objects in a scene
    """
    for node, pose in zip(nodes,poses):
        scene.set_pose(node,pose)