# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from .skeleton import Skeleton
import numpy as np
from .projection import fill_hole
import cv2
import string

class Predictor(Skeleton):
    
    def __init__(self, skele_name, save_history = True):
        super().__init__(skele_name)
        assert self._hasJointConfig(),"Joint Configuration must be present for angle prediction."
        self.predictable_joints = [joint for joint in self.joint_data.keys() if len(self.joint_data[joint]['predictors']) > 0]
        self.save_history = save_history
        if save_history:
            self.prediction_history = []
            self.full_prediction_history = {j:{"predictors":[],"values": [],"multipliers": [],"estimated": []} for j in self.predictable_joints}

    def load(self, keypoint_detections, pointmap):
        self.detections = {}

        for name, idx in zip(self.keypoints, range(len(self.keypoints))):
            detected = True
            px = round(keypoint_detections[idx][0])
            py = round(keypoint_detections[idx][1])
            if px < 10 and py < 10:
                detected = False
                estimated = True

            coords = pointmap[py,px]
            if detected:
                estimated = False
                if not np.any(coords):
                    coords = fill_hole(pointmap, py, px, 25)
                    estimated = True

            self.detections[name] = {'coords':coords, 'px_coords':(int(px),int(py)), 'confidence':keypoint_detections[idx][2], 'estimated':estimated, 'detected':detected}


    def visualize(self, image):
        
        for key in self.detections:
            if self.detections[key]['detected']:
                if self.keypoint_data[key]['parent_keypoint'] in self.keypoints:
                    overlay = np.copy(image)
                    overlay = cv2.line(overlay, self.detections[key]['px_coords'], self.detections[self.keypoint_data[key]['parent_keypoint']]['px_coords'], color=(255, 0, 0), thickness=3)
                    a = self.detections[key]['confidence'] * self.detections[self.keypoint_data[key]['parent_keypoint']]['confidence']
                    image = cv2.addWeighted(overlay,a,image,1-a,0)
            
        for key in self.detections:
            if self.detections[key]['detected']:
                if self.detections[key]['estimated']:
                    color=(0, 0, 255)
                else:
                    color=(0, 255, 0)
                overlay = np.copy(image)
                overlay = cv2.circle(overlay, self.detections[key]['px_coords'], radius=5, color=color, thickness=-1)
                overlay = cv2.circle(overlay, self.detections[key]['px_coords'], radius=6, color=[x//3 for x in color], thickness= 1)
                a = self.detections[key]['confidence']
                image = cv2.addWeighted(overlay,a,image,1-a,0)
        
        return image
            

    def predict(self):
        self.predictions = {}
        for joint in self.predictable_joints:
            pred, est = self._predictAngle(joint,self.predictions)
            self.predictions[joint] = {"val": pred,"percent_est":est}
            
        if self.save_history:
            self.prediction_history.append(self.predictions)

        return self.predictions


    def _predictAngle(self,joint_name, predictions):
        tp = self.joint_data[joint_name]['type']

        if tp == 1:
            pred, est = self._type1predict(joint_name)
        elif tp == 2:
            pred, est = self._type2predict(joint_name)
        elif tp == 3:
            pred, est = self._type3predict(joint_name)

        if pred is None:
            return None, 1

        pred *= self.joint_data[joint_name]['self_mult']

        if self.joint_data[joint_name]['parent']:
            pred += self.joint_data[joint_name]['parent_mult'] * predictions[self.joint_data[joint_name]['parent']]['val']
            est = 1 - ((1 - predictions[self.joint_data[joint_name]['parent']]['percent_est'])*(1-est))

        pred += self.joint_data[joint_name]['offset']

        while pred < self.joint_data[joint_name]['min']:
            pred += 2 * np.pi
        
        while pred > self.joint_data[joint_name]['max']:
            pred -= 2 * np.pi

        return pred, est


    def _type1predict(self,joint_name):
        """ L, U, B Angles"""
        pairs, lengs, confidence, offsets, estimate, detected = self._getPredictors(joint_name)
        multipliers = self._type1Multipliers(pairs, lengs, confidence, estimate)
        
        vecs = pairs[:,1] - pairs[:,0]
        rot_y = vecs[:,1]
        rot_x = np.sqrt(vecs[:,0] ** 2 + vecs[:,2] ** 2) * abs(vecs[:,0]) / vecs[:,0]

        # Convert to angles between +/-pi rad
        preds = np.arctan(rot_y/rot_x)
        neg_x = rot_x < 0
        neg_y = rot_y < 0
        pos_y = rot_y > 0

        preds[np.where(neg_x & pos_y)] = preds[np.where(neg_x & pos_y)] + np.pi
        preds[np.where(neg_x & neg_y)] = -np.pi/2 - preds[np.where(neg_x & neg_y)]

        preds += offsets

        # Range discontinuity around +pi and -pi. Convert to positive angles to see if stdev is lower.
        # if len(preds) > 1:
        #     base_std = np.std(preds)
        #     preds_alt = np.copy(preds)
        #     preds_alt[np.where(preds_alt < 0)] = 2*np.pi + preds_alt[np.where(preds_alt < 0)]
        #     alt_std = np.std(preds_alt)
        #     if alt_std < base_std:
        #         preds = preds_alt

        pred = np.sum(multipliers * preds) / np.sum(multipliers)

        z = np.zeros(len(detected))
        if self.save_history:
            z[np.where(detected)] = preds * self.joint_data[joint_name]['self_mult'] + self.joint_data[joint_name]['offset']
            if self.joint_data[joint_name]['parent']:
                z[np.where(detected)] += self.joint_data[joint_name]['parent_mult'] * self.predictions[self.joint_data[joint_name]['parent']]['val']
            self.full_prediction_history[joint_name]["values"].append(z.astype(float))
            z[np.where(detected)] = multipliers
            self.full_prediction_history[joint_name]["multipliers"].append(z.astype(float))
            z[np.where(detected)] = estimate
            self.full_prediction_history[joint_name]["estimated"].append(z.astype(bool))
    
        return pred, sum(estimate) / len(estimate)


    def _type2predict(self,joint_name):
        """ S Angle """
        pairs, lengs, confidence, offsets, estimate, detected = self._getPredictors(joint_name)
    
        vecs = pairs[:,1] - pairs[:,0]
        preds = np.arctan(vecs[:,2]/vecs[:,0])

        multipliers = self._type2Multipliers(pairs, lengs, confidence,estimate, vecs)

        preds += offsets
        pred = np.sum(multipliers * preds) / np.sum(multipliers)

        z = np.zeros(len(detected))
        if self.save_history:
            z[np.where(detected)] = preds * self.joint_data[joint_name]['self_mult'] + self.joint_data[joint_name]['offset']
            if self.joint_data[joint_name]['parent']:
                z[np.where(detected)] += self.joint_data[joint_name]['parent_mult'] * self.predictions[self.joint_data[joint_name]['parent']]['val']
            self.full_prediction_history[joint_name]["values"].append(z.astype(float))
            z[np.where(detected)] = multipliers
            self.full_prediction_history[joint_name]["multipliers"].append(z.astype(float))
            z[np.where(detected)] = estimate
            self.full_prediction_history[joint_name]["estimated"].append(z.astype(bool))


        return pred, sum(estimate) / len(estimate)


    def _type3predict(self,joint_name):
        """ R Angle """
        # Find S plane
        #   Parameterize an S plane based on the value determined for the S angle
        #   https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
        parent = self.joint_data[joint_name]['parent']
        parent_plane_vec = np.array([np.sin(self.predictions[parent]['val']),np.cos(self.predictions[parent]['val']),0])

        pairs, lengs, confidence, offsets, estimate, ratio_detected = self._getPredictors(joint_name)
    
        vecs = pairs[:,1] - pairs[:,0]

        # Find angle R plane makes with the S plane
        #   Use the vectors to determine the angle the vector makes with the normal of the plane
        #   https://www.vedantu.com/maths/angle-between-a-line-and-a-plane

        preds = []
        for vec in vecs:
            preds.append(np.arcsin(np.dot(vec, parent_plane_vec)/(np.linalg.norm(vec)*np.linalg.norm(parent_plane_vec))))
        preds = np.array(preds)

        multipliers = self._type3Multipliers(pairs, lengs, confidence, estimate)

        preds += offsets
        pred = np.sum(multipliers * preds) / np.sum(multipliers)

        return pred, sum(estimate) / len(estimate)


    def _type3Multipliers(self,pairs,lengs,confidence,estimate):
        detected_lengs = self._distances(pairs)
        len_multipliers = self._lengthMultiplier(lengs, detected_lengs)
        confidence_multipliers = 2*lengs *(np.power(np.prod(confidence,-1),1.75) / np.square(np.sum(confidence,-1)))
        est_multipliers = 1 - .75 * estimate
        multipliers = len_multipliers * confidence_multipliers * est_multipliers
        return multipliers


    def _type2Multipliers(self,pairs,lengs,confidence,estimate,vecs):
        detected_lengs = self._distances(pairs)
        len_multipliers = self._lengthMultiplier(lengs, detected_lengs)
        confidence_multipliers = 2*lengs *(np.power(np.prod(confidence,-1),1.75) / np.square(np.sum(confidence,-1)))
        x_distance_multiplier = vecs[:,0] / detected_lengs
        est_multipliers = 1 - .65 * estimate
        multipliers = len_multipliers * confidence_multipliers * x_distance_multiplier * est_multipliers
        return multipliers


    def _type1Multipliers(self,pairs,lengs,confidence, estimate):
        detected_lengs = self._distances(pairs)
        len_multipliers = self._lengthMultiplier(lengs, detected_lengs)
        confidence_multipliers = 2*lengs *(np.power(np.prod(confidence,-1),1.75) / np.square(np.sum(confidence,-1)))
        est_multipliers = 1 - .5 * estimate
        multipliers = len_multipliers * confidence_multipliers * est_multipliers
        return multipliers

    def _lengthMultiplier(self, lengs, detected_lengs):
        # return np.exp(-(np.abs(lengs - detected_lengs)/lengs))
        p = .85
        return np.clip(((-1 / lengs * p**(1/p)) * (np.abs(lengs - detected_lengs))**p + 1),0,np.inf)


    def _getPredictors(self, joint_name):
        joint_info = self.joint_data[joint_name]

        pairs = []
        lengs = []
        confidence = []
        offsets = []
        contains_estimate = []
        detected = []
        for key in joint_info['predictors'].keys():
            points = [
                self.detections[joint_info['predictors'][key]['from']]['coords'],
                self.detections[joint_info['predictors'][key]['to']]['coords']
                ]
            pairs.append(points)
            lengs.append(joint_info['predictors'][key]['length'])
            conf = [
                self.detections[joint_info['predictors'][key]['to']]['confidence'],
                self.detections[joint_info['predictors'][key]['from']]['confidence']
                ]
            confidence.append(conf)
            offsets.append(joint_info['predictors'][key]['offset'])
            contains_estimate.append(
                self.detections[joint_info['predictors'][key]['to']]['estimated'] or
                self.detections[joint_info['predictors'][key]['from']]['estimated']
                )
            detected.append(
                self.detections[joint_info['predictors'][key]['to']]['detected'] and
                self.detections[joint_info['predictors'][key]['from']]['detected']
                )

        detected = np.array(detected)
        pairs = np.array(pairs)[np.where(detected)]
        lengs = np.array(lengs)[np.where(detected)]
        confidence = np.array(confidence)[np.where(detected)]
        offsets = np.array(offsets)[np.where(detected)]
        contains_estimate = np.array(contains_estimate)[np.where(detected)]
        return pairs, lengs, confidence, offsets, contains_estimate, detected


    def _distances(self, point_arr):
        diff = point_arr[:,0,:] - point_arr[:,1,:]
        dist = np.power(np.sum(np.square(diff),-1),.5)
        return dist
