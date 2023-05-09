#!/usr/bin/env python

##  object_detection_and_localization_iou.py

"""
This script is for experimenting with IoU-based loss function for the
regression part of object detection and tracking.  These loss functions are 
defined in the class  

       DIoULoss

that is in the inner class DetectAndLocalize of DLStudio.  See Slides 37 
through 42 of my Week 7 presentation on Object Detection and Localization
for an explanation of these loss functions.  This script also uses the 

       PurdueShapes5 

dataset for training and testing.
"""

import random
import numpy
import torch
import os, sys


seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


##  watch -d -n 0.5 nvidia-smi

from DLStudio import *

dls = DLStudio(
                  dataroot = "./data/PurdueShapes5/",
#                  dataroot = "/mnt/cloudNAS3/Avi/ImageDatasets/PurdueShapes5/",
#                  dataroot = "/home/kak/ImageDatasets/PurdueShapes5/",
                  image_size = [32,32],
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
#                  learning_rate = 1e-4,                    ## when loss_mode is set to d2 or diou1
                  learning_rate = 5e-3,                     ## when loss_mode is set to diou2 or diou3
                  epochs = 4,
                  batch_size = 4,
                  classes = ('rectangle','triangle','disk','oval','star'),
                  use_gpu = True,
              )


detector = DLStudio.DetectAndLocalize( dl_studio = dls )

dataserver_train = DLStudio.DetectAndLocalize.PurdueShapes5Dataset(
                                   train_or_test = 'train',
                                   dl_studio = dls,
                                   dataset_file = "PurdueShapes5-10000-train.gz", 
                                                                      )
dataserver_test = DLStudio.DetectAndLocalize.PurdueShapes5Dataset(
                                   train_or_test = 'test',
                                   dl_studio = dls,
                                   dataset_file = "PurdueShapes5-1000-test.gz"
                                                                  )
detector.dataserver_train = dataserver_train
detector.dataserver_test = dataserver_test

detector.load_PurdueShapes5_dataset(dataserver_train, dataserver_test)

model = detector.LOADnet2(skip_connections=True, depth=8)

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)

num_layers = len(list(model.parameters()))
print("\nThe number of layers in the model: %d\n\n" % num_layers)


#detector.run_code_for_training_with_iou_regression(model, loss_mode='d2', show_images=True)             ## use learning_rate = 1e-4
#detector.run_code_for_training_with_iou_regression(model, loss_mode='diou1', show_images=True)          ## use learning_rate = 1e-4
#detector.run_code_for_training_with_iou_regression(model, loss_mode='diou2', show_images=True)          ## use learning_rate = 5e-3
detector.run_code_for_training_with_iou_regression(model, loss_mode='diou3', show_images=True)           ## use learning_rate = 5e-3

import pymsgbox
response = pymsgbox.confirm("Finished training.  Start testing on unseen data?")
if response == "OK": 
    detector.run_code_for_testing_detection_and_localization(model)

