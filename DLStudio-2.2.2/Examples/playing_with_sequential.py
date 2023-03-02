#!/usr/bin/env python

##  playing_with_sequential.py


"""
Shows you how you can call on a custom inner class of the 'DLStudio'
module that is meant to experiment with your own network.  The name of the
inner class in this example script is ExperimentsWithSequential
"""

import random
import numpy
import torch
import os, sys

"""
seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)
"""

##  watch -d -n 0.5 nvidia-smi

from DLStudio import *

dls = DLStudio(
#                  dataroot = "/home/kak/ImageDatasets/CIFAR-10/",
                  dataroot = "./data/CIFAR-10/",
                  image_size = [32,32],
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
                  learning_rate = 1e-3,
                  epochs = 2,
                  batch_size = 8,
                  classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck'),
                  use_gpu = True,
              )

exp_seq = DLStudio.ExperimentsWithSequential( dl_studio = dls )
exp_seq.load_cifar_10_dataset_with_augmentation()
model = exp_seq.Net()

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the model: %d\n" % number_of_learnable_params)

num_layers = len(list(model.parameters()))
print("\nThe number of layers in the model: %d\n" % num_layers)


exp_seq.run_code_for_training(model)
exp_seq.run_code_for_testing(model)

