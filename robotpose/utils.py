# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley


import os
import string

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

import multiprocessing as mp


def setMemoryGrowth():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def workerCount():
    cpu_count = mp.cpu_count()
    return int(min(cpu_count - 2, .85 * cpu_count))


def expandRegion(image, size, iterations = 1):
    kern = np.ones((size,size), dtype=np.uint8)
    return cv2.dilate(image, kern, iterations = iterations)




def XYangle(x, y, lims=None):
    """
    Returns the angle between the input point and the +x axis 
    """
    # Get angle
    ang = np.arctan(y/x)

    # If in Quad II/III, make the angle obtuse 
    if x < 0:
        ang += np.pi

    # Apply any custom limits to the angle
    if lims is not None:
        if ang > max(lims):
            ang -= 2 * np.pi
        elif ang < min(lims):
            ang += 2* np.pi
    
    return ang


def XYZangle(start, end, lims = None):
    """
    Rotates vector from start to end into the XY plane and uses XY angle to calculate angle
    """
    # Find link vector from start and end points
    vec = np.subtract(end, start)

    # Rotate the reference XY plane to contain the vector
    rotated_y = vec[1]
    rotated_x = np.sqrt(vec[0] ** 2 + vec[2] ** 2) * abs(vec[0]) / vec[0]

    # Find 2D angle to X axis
    return XYangle(rotated_x, rotated_y, lims)



def reject_outliers_std(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def outlier_min_max(data, iqr_mult = 1.5):
    data = reject_outliers_iqr(data, iqr_mult)
    return np.min(data), np.max(data)

def reject_outliers_iqr(data, iqr_mult = 1.5):
    percentiles = np.percentile(data, [75, 25])
    iqr = np.subtract(*percentiles)
    max = percentiles[0] + iqr_mult * iqr
    min = percentiles[1] - iqr_mult * iqr
    data = data[data >= min]
    data = data[data <= max]
    return data



class Timer():
    def __init__(self):
        self.data = {}
        self.start()
    
    def start(self):
        self.start_time = time.time()

    def split(self,split_name):
        try:
            self.data[split_name] += time.time() - self.start_time
        except KeyError:
            self.data[split_name] = time.time() - self.start_time
        self.start()

    def __repr__(self):
        out = "Times:"
        tot = 0
        for item in self.data:
            tot += self.data[item]
        for item in self.data:
            out += f"\n\t{item}: {self.data[item]:.3f}s {(self.data[item] * 100/tot):.2f}%"
        return out







class Grapher():

    def __init__(self, joints, prediction_history, ds_angles = None, joint_predictions = None):
        self.joints = joints
        self._convertToMatrix(prediction_history, joints)
        self.compare = ds_angles is not None
        if ds_angles is not None:
            self._cropComparison(ds_angles,joints,prediction_history)
        if joint_predictions is not None:
            self._convertJointToMatricies(joint_predictions,joints)


    
    def plot(self,ylim=None):
        self._plotWithComparison(ylim)

    def plotJoint(self,joint,ylim=None):
        self._plotSingleJoint(joint,ylim)


    def _convertToMatrix(self, prediction_history, joints):
        # Convert into a L X N matrix (N is # of joint, L is prediction length)
        self.angles = np.zeros((len(prediction_history),len(joints)))
        self.percent_estimated = np.copy(self.angles)

        for idx in range(len(prediction_history)):
            for joint, subidx in zip(joints, range(len(joints))):
                self.angles[idx,subidx] = prediction_history[idx][joint]['val']
                self.percent_estimated[idx,subidx] = prediction_history[idx][joint]['percent_est']

        self.angles = np.degrees(self.angles)


    def _convertJointToMatricies(self, joint_predictions, joints):
        self.joint_data = {}
        for joint in joints:
            self.joint_data[joint] = {
                "values": np.array(joint_predictions[joint]["values"]),
                "multipliers": np.array(joint_predictions[joint]["multipliers"]),
                "estimated": np.array(joint_predictions[joint]["estimated"]),
            }


    def _cropComparison(self, ds_angles, joints, prediction_history):
        ang = ['S','L','U','R','B','T']
        l = len(prediction_history)
        self.real_angles = np.zeros((len(prediction_history),len(joints)))
        for joint, idx in zip(joints,range(len(joints))):
            self.real_angles[:,idx] = ds_angles[:l,ang.index(joint)]
        
        self.real_angles = np.degrees(self.real_angles)


    def _plotWithComparison(self, y_lim = None):

        fig, axs = plt.subplots(len(self.joints),2)

                
        # Plot Raw Angles
        for joint, idx in zip(self.joints,range(len(self.joints))):
            axs[idx,0].set_title(f'Raw {joint} Angle')
            axs[idx,0].plot(self.real_angles[:,idx])
            axs[idx,0].plot(self.angles[:,idx],color='purple')
            for val,x in zip(self.percent_estimated[:,idx], range(len(self.percent_estimated[:,idx]))):
                axs[idx,0].axvspan(x-.5, x+.5, color='red', alpha=val, ec=None)

        err = self.angles - self.real_angles
        zeros_err = np.zeros(err.shape[0])

        # Plot errors
        for joint, idx in zip(self.joints,range(len(self.joints))):
            axs[idx,1].set_title(f'Angle {joint} Error')
            axs[idx,1].plot(zeros_err)
            axs[idx,1].plot(err[:,idx],color='purple')
            if y_lim is not None:
                axs[idx,1].set_ylim([-y_lim,y_lim])
            for val,x in zip(self.percent_estimated[:,idx], range(len(self.percent_estimated[:,idx]))):
                axs[idx,1].axvspan(x-.5, x+.5, color='red', alpha=val, ec=None)


        avg_err = np.mean(np.abs(err),0)
        avg_err_std = np.std(np.abs(err),0)


        print("\nAvg Error (deg):")
        for joint, idx in zip(self.joints,range(len(self.joints))):
            print(f"\t{joint}: {avg_err[idx]:.2f}")

        print("Stdev (deg):")
        for joint, idx in zip(self.joints,range(len(self.joints))):
            print(f"\t{joint}: {avg_err_std[idx]:.2f}")


        # # Determine average errors without outliers
        # avg_err_no_outliers = np.mean(reject_outliers_iqr(np.abs(err)))
        # avg_L_err = np.mean(reject_outliers_iqr(np.abs(L_err)))
        # avg_U_err = np.mean(reject_outliers_iqr(np.abs(U_err)))
        # #avg_B_err = np.mean(np.abs(B_err))
        # S_err_std = np.std(reject_outliers_iqr(np.abs(S_err)))
        # L_err_std = np.std(reject_outliers_iqr(np.abs(L_err)))
        # U_err_std = np.std(reject_outliers_iqr(np.abs(U_err)))
        # #B_err_std = np.std(np.abs(B_err))

        # print("\nOutliers Removed:")
        # print("Avg Error (deg):")
        # print(f"\tS: {avg_S_err:.2f}\n\tL: {avg_L_err:.2f}\n\tU: {avg_U_err:.2f}")
        # print("Stdev (deg):")
        # print(f"\tS: {S_err_std:.2f}\n\tL: {L_err_std:.2f}\n\tU: {U_err_std:.2f}")

        plt.show()



    def _plotSingleJoint(self, joint, y_lim = None):

        fig, axs = plt.subplots(2)
                
        vals = np.array(self.joint_data[joint]["values"])
        mults = np.array(self.joint_data[joint]["multipliers"])
        vals *= 180/np.pi
        vals[vals==0] = np.nan
        mults[mults==0] = np.nan

        # # Plot Raw Angles
        # axs[0].set_title(f'Raw {joint} Angle')
        # axs[0].plot(self.real_angles[:,self.joints.index(joint)], label="Act.")
        # for idx in range(vals.shape[1]):
        #     axs[0].plot(vals[:,idx],label=list(string.ascii_uppercase)[idx])
        # axs[0].legend()

        axs[0].set_title(f'Multipliers')
        for idx in range(vals.shape[1]):
            axs[0].plot(mults[:,idx],label=list(string.ascii_uppercase)[idx])
        axs[0].legend()

        err = vals - np.vstack([self.real_angles[:,self.joints.index(joint)]]*vals.shape[1]).transpose()
        zeros_err = np.zeros(err.shape[0])


        axs[1].set_title(f'Angle {joint} Error')
        axs[1].plot(zeros_err, label="Act.")
        # Plot errors
        for idx in range(err.shape[1]):
            axs[1].plot(err[:,idx],label=list(string.ascii_uppercase)[idx])
        if y_lim is not None:
            axs[1].set_ylim([-y_lim,y_lim])
        axs[1].legend()

        # avg_err = np.mean(np.abs(err),0)
        # avg_err_std = np.std(np.abs(err),0)

        # print("\nAvg Error (deg):")
        # for joint, idx in zip(self.joints,range(len(self.joints))):
        #     print(f"\t{joint}: {avg_err[idx]:.2f}")

        # print("Stdev (deg):")
        # for joint, idx in zip(self.joints,range(len(self.joints))):
        #     print(f"\t{joint}: {avg_err_std[idx]:.2f}")

        plt.show()