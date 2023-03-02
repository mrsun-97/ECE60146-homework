#!/usr/bin/env python

##  dcgan_DG1.py

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

This script is based on the Discriminator-Generator pair DG1 in the AdversarialLearning
class of the DLStudio module.

The DG1 pair is an implementation of DCGAN as presented in the paper "Unsupervised
Representation Learning with Deep Convolutional Generative Adversarial Networks" by
Radford et al.  DCGAN is short for "Deep Convolutional Generative Adversarial Network".

The DCGAN neural networks are based on a "4-2-1" scheme for the different layers in the
two networks.  For both the Generator and the Discriminator, the "4-2-1" formula stands
for a kernel size of 4x4, a stride of 2x2, and a padding of 1x1.  In the Discriminator
where the goal is to construct an abstraction pyramid that culminates in a binary decision
about accepting an image as real or fake, 4x4 convolutional kernels are used in each layer
of the Discriminator.  In the Generator, on the other hand, a kernel size of 4x4 refers to
the TransposeConvolutions in the different layers for upsampling needed for generating an
image from a random noise vector.
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
from AdversarialLearning import *

import sys

dls = DLStudio(                                                                                       
#                  dataroot = "/mnt/cloudNAS3/Avi/ImageDatasets/PurdueShapes5GAN/multiobj/",
#                  dataroot = "/home/kak/ImageDatasets/PurdueShapes5GAN/multiobj/",  
                  dataroot = "./dataGAN/PurdueShapes5GAN/multiobj/", 
                  image_size = [64,64],                                                               
                  path_saved_model = "./saved_model", 
                  learning_rate = 2e-4,      ## <==  try smaller value if mode collapse
                  epochs = 30,
                  batch_size = 32,                                                                     
                  use_gpu = True,                                                                     
              )           

adversarial = AdversarialLearning(
                  dlstudio = dls,
                  ngpu = 1,    
                  latent_vector_size = 100,
                  beta1 = 0.5,                           ## for the Adam optimizer
              )

dcgan =   AdversarialLearning.DataModeling( dlstudio = dls, adversarial = adversarial )


discriminator =  dcgan.DiscriminatorDG1()
generator =  dcgan.GeneratorDG1()

num_learnable_params_disc = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the Discriminator: %d\n" % num_learnable_params_disc)
num_learnable_params_gen = sum(p.numel() for p in generator.parameters() if p.requires_grad)
print("\nThe number of learnable parameters in the Generator: %d\n" % num_learnable_params_gen)
num_layers_disc = len(list(discriminator.parameters()))
print("\nThe number of layers in the discriminator: %d\n" % num_layers_disc)
num_layers_gen = len(list(generator.parameters()))
print("\nThe number of layers in the generator: %d\n\n" % num_layers_gen)

dcgan.set_dataloader()

dcgan.show_sample_images_from_dataset(dls)

dcgan.run_gan_code(dls, adversarial, discriminator=discriminator, generator=generator, results_dir="results_DG1")

