#!/usr/bin/env python

##  wgan_CG1.py

"""
IMPORTANT NOTE:  

    You will need to install the PurdueShapes5GAN dataset before you can execute this script.

    Download the dataset archive

            datasets_for_AdversarialLearning.tar.gz

    through the link "Download the image dataset for AdversarialLearning" provided at the top
    of the HTML version of the main module doc page and store it in the
    'ExamplesAdversarialLearning' directory of the distribution.  Subsequently, execute the
    following command in that directory:

            tar zxvf datasets_for_AdversarialLearning.tar.gz

    This command will create a 'dataGAN' subdirectory and deposit the following dataset archive
    in that subdirectory:

            PurdueShapes5GAN-20000.tar.gz

    Now execute the following in the "dataGAN" directory:

            tar zxvf PurdueShapes5GAN-20000.tar.gz

    With that, you should be able to execute the adversarial learning based scripts in the
    'ExamplesAdversarialLearning' directory.

"""



"""
ABOUT THIS SCRIPT:

In the name of this file, "CG1" stands for "Critic-Generator-1".  My plan is to
include other types of Critic-Generator pairs under different CGn labels in future
versions of the DLStudio module.  

This script uses the Critic-Generator CG1 pair in the AdversarialLearning class of
the DLStudio module for data modeling.  That Critic-Generator is an implementaiton of
adversarial learning algorithm presented in the paper "Wasserstein GAN" by Arjovsky,
Chintala, and Bottou.
"""


import random
import numpy
import torch
import os, sys


seed = 10           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


##  watch -d -n 0.5 nvidia-smi

from DLStudio import *
from AdversarialLearning import *

import sys

dls = DLStudio(                                                       
           dataroot = "./dataGAN/PurdueShapes5GAN/multiobj/", 
#           dataroot = "/mnt/cloudNAS3/Avi/ImageDatasets/PurdueShapes5GAN/multiobj/",
#           dataroot = "/home/kak/ImageDatasets/PurdueShapes5GAN/multiobj/",  
           image_size = [64,64],                                                               
           path_saved_model = "./saved_model",                   
           learning_rate = 5e-5,                             
           epochs = 500,
           batch_size = 32,                     
           use_gpu = True,                          
      )           

adversarial = AdversarialLearning(
                  dlstudio = dls,
                  ngpu = 1,    
                  latent_vector_size = 100,
                  clipping_threshold = 0.005,            ## needed for WGAN parameter clipping
                  beta1 = 0.5,                           ## for the Adam optimizer
              )

wgan =   AdversarialLearning.DataModeling( 
                  dlstudio = dls, 
                  adversarial = adversarial,
                  num_workers = 0,
              )


critic =  wgan.CriticCG1()
generator =  wgan.GeneratorCG1()

num_learnable_params_disc = sum(p.numel() for p in critic.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the Critic: %d\n" % num_learnable_params_disc)
num_learnable_params_gen = sum(p.numel() for p in generator.parameters() if p.requires_grad)
print("\nThe number of learnable parameters in the Generator: %d\n" % num_learnable_params_gen)
num_layers_critic = len(list(critic.parameters()))
print("\nThe number of layers in the Critic: %d\n" % num_layers_critic)
num_layers_gen = len(list(generator.parameters()))
print("\nThe number of layers in the Generator: %d\n\n" % num_layers_gen)

wgan.set_dataloader()

wgan.show_sample_images_from_dataset(dls)

wgan.run_wgan_code(dls, adversarial, critic=critic, generator=generator, results_dir="results_WGAN")
