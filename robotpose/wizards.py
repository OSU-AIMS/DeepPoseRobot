# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import os
import cv2
import numpy as np

from deepposekit import Annotator
import PySimpleGUI as sg

from .dataset import DatasetInfo, Dataset
from .render import Aligner
from .skeleton import Skeleton, valid_skeletons
from .simulation.rendering import SkeletonRenderer


class DatasetWizard(DatasetInfo):
    def __init__(self):
        super().__init__()
        self.get()

        self.layout = [          
            [sg.Text("Dataset:"),sg.InputCombo(self.compiled_sets(),key='-dataset-', size=(20, 1))],
            [sg.Button("View Details",key='-load-',tooltip='View dataset details'),
                sg.Button("Align",key='-align-',tooltip='Align Dataset images with renderer')],
            [sg.HorizontalSeparator()],
            [sg.Text("Keypoint Skeleton:"),
                sg.InputCombo(valid_skeletons(),key='-skeleton-', size=(20, 1)),
                sg.Button("Edit Skeleton",key='-edit_skele-',tooltip='Edit Skeleton with Skeleton Wizard')],
            [sg.Button("View Annotations",key='-manual_annotate-', disabled=True)],
            [sg.HorizontalSeparator()],
            [sg.Button("Quit",key='-quit-',tooltip='Quit Dataset Wizard')]
            ]


    def run(self):
        self.window = sg.Window('Dataset Wizard', self.layout.copy(), finalize=True)
        self.window.bring_to_front()

        event = ''
        while event not in (sg.WIN_CLOSED,'-quit-'):
            event, values = self.window.read(10)
            if event not in (sg.WIN_CLOSED,'-quit-'):
                self._updateButtons(values)
                self._runEvent(event, values)

        self.window.close()


    def _updateButtons(self,values):
        
        if values['-dataset-'] in self.unique_sets():
            for button in ['-load-','-align-']:
                self.window[button].update(disabled = False)
            if values['-skeleton-'] in valid_skeletons():
                for button in ['-manual_annotate-']:
                    self.window[button].update(disabled = False)
            else:
                for button in ['-manual_annotate-']:
                    self.window[button].update(disabled = True)
        else:
            for button in ['-load-','-align-']:
                self.window[button].update(disabled = True)

        if values['-skeleton-'] in valid_skeletons():
            for button in ['-edit_skele-']:
                self.window[button].update(disabled = False)
        else:
            for button in ['-edit_skele-']:
                self.window[button].update(disabled = True)
                


    def _runEvent(self,event,values):
        if event =='-align-':
            self._runAligner(values['-dataset-'])
        elif event == '-load-':
            pass
        elif event == '-manual_annotate-':
            self._manualAnnotate(values['-dataset-'], values['-skeleton-'])
        elif event == '-edit_skele-':
            self._runKeypointWizard(values['-skeleton-'])
           
    def _manualAnnotate(self,dataset,skeleton):
        ds = Dataset(dataset,skeleton)
        ds.makeDeepPoseDS()
        app = Annotator(datapath=os.path.abspath(ds.deepposeds_path),
                dataset='images',
                skeleton=ds.skele.csv_path,
                shuffle_colors=False,
                text_scale=1)

        app.run()


    def _runAligner(self, dataset):
        print(f'Aligning {dataset}')
        align = Aligner(dataset)
        align.run()
        print(f'Alignment Complete')

    def _runKeypointWizard(self, skeleton):
        self.window.disable()
        self.window.disappear()
        wiz = SkeletonWizard(skeleton)
        wiz.run()
        cv2.destroyAllWindows()
        self.window.enable()
        self.window.reappear()
        self.window.bring_to_front()




class SkeletonWizard(Skeleton):

    def __init__(self, name):
        super().__init__(name)

        self.rend = SkeletonRenderer(name)
        self.base_pose = [1.5,-1.5,.35, 0,np.pi/2,0]
        self._setRotation(0,0)
        self.mode = 0

        self.rend.setJointAngles([0,0,0,0,0,0])

        def jointSlider(name, lower, upper):
            return [sg.Text(f"{name}:"),
                sg.Slider(range=(lower, upper),
                    orientation='h', tick_interval=90, 
                    size=(20, 20), default_value=0, key=f'-{name}-')]

        column1 = [
            [sg.Frame('View Settings',[
                [sg.Slider(range=(-30, 30), orientation='v', size=(5, 20), default_value=0,key='-vert_slider-'),
                    sg.VerticalSeparator(),
                    sg.Button("Change Mode",key='-view_mode-'),
                    sg.VerticalSeparator(),
                    sg.Button("Reset",key='-view_reset-')],
                [sg.Slider(range=(-180, 180), orientation='h', size=(20, 20), default_value=0, key='-horiz_slider-')]
            ]
            )],
            [sg.Frame('Robot Joints',[
                jointSlider("B",-135,135),
                jointSlider("R",-190,190),
                jointSlider("U",-135,255),
                jointSlider("L",-65,150),
                jointSlider("S",-170,170),
                [sg.Button("Reset",key='-joint_reset-')]
            ]
            )]
            ]

        self.layout = [          
            [sg.Text(f"Keypoint Skeleton: {name}")],
            [sg.Column(column1)],
            [sg.Button("Quit",key='-quit-',tooltip='Quit Skeleton Wizard')]
            ]
        

    
    def run(self):
        self.window = sg.Window('Skeleton Wizard', self.layout)

        event = ''
        while event not in (sg.WIN_CLOSED,'-quit-'):
            event, values = self.window.read(1)
            if event not in (sg.WIN_CLOSED,'-quit-'):
                self._runEvent(event, values)
                self.render()
            self.window.bring_to_front()

        self.window.close()



    def _runEvent(self, event, values):
        self._setRotation(values['-horiz_slider-'],values['-vert_slider-'])
        self._setJointAngles(values)

        if event == '-view_reset-':
            self._resetRotation()
        if event == '-joint_reset-':
            self._resetJointAngles()
        if event == '-view_mode-':
            modes = ['key','seg','seg_full','real']
            self.mode += 1
            if self.mode >= len(modes):
                self.mode = 0
            self.rend.setMode(modes[self.mode])


    def _resetRotation(self):
        self._setRotation(0,0)
        for slider in ['-horiz_slider-','-vert_slider-']:
            self.window[slider].update(0)

    def _setRotation(self, rotation_h, rotation_v):
        self.rotation_h = (rotation_h/180) * np.pi
        self.rotation_v = (rotation_v/180) * np.pi

        self.c_pose = np.copy(self.base_pose)
        self.c_pose[1] *= (1 - np.sin(self.rotation_v) * np.tan(self.rotation_v)) * np.cos(self.rotation_h)
        self.c_pose[0] *= np.sin(self.rotation_h)
        self.c_pose[2] = np.sin(self.rotation_v) * 1 + .15
        self.c_pose[4] = np.pi/2 - self.rotation_v
        self.c_pose[5] = self.rotation_h
        self.rend.setCameraPose(self.c_pose)


    def _resetJointAngles(self):
        self.rend.setJointAngles([0,0,0,0,0,0])
        for joint in ['-S-','-L-','-U-','-R-','-B-']:
            self.window[joint].update(0)

    def _setJointAngles(self, values):
        joint_angles = [0,0,0,0,0,0]
        for joint, idx in zip(['-S-','-L-','-U-','-R-','-B-'], range(5)):
            joint_angles[idx] = values[joint] * np.pi/180 

        self.rend.setJointAngles(joint_angles)
    

    def render(self):
        color, depth = self.rend.render()
        cv2.imshow("Keypoint Wizard",color)
        cv2.waitKey(1)

