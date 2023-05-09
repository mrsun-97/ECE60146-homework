#!/usr/bin/env python

##  dcgan_DG2.py

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

This script is meant to show that the 4-2-1 GAN network demonstrated in the 
script 'dcgan_multiobj_DG1.py' is very sensitive to the network layout used
for the Discriminator and the Generator.  As you will notice, this script
uses the classes DiscriminatorDG2 and GeneratorDG2 for constructing
the discriminator and the generator networks.  

To appreciate what is demonstrated by this script, compare the implementations
for the following classes in the enclosing class AdversarialLearning of the
DLStudio module:

    DiscriminatorDG1

    DiscriminatorDG2

You will notice that the latter class has the following extra statements in 
the constructor for modifying the 4-2-1 pyramid of the former class:

    self.extra =   nn.Conv2d(  64,    64,      kernel_size=4,      stride=1,    padding=2)
    self.bnX  = nn.BatchNorm2d(64)

and the following additional statements in the "forward()":

    x = self.bnX(self.extra(x))   
    x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)

The result of these changes may either be a non-convergence of the GAN, or a delayed
convergence for the weight initializations used. 

To demonstrate this effect, this script achieves convergence if you set the seed to 10,
but not if you were to set the seed to the commonly used value of 0 (for obtaining 
reproducible results).

You will also notice that, unlike what was the case with the pure DCGAN based script, the 
script shown below is able to maintain its solution state only for a few epochs before it 
suffers from a mode collapse.
"""

import random
import numpy
import torch
import os, sys

"""
seed = 10                        
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
                  learning_rate = 2e-4,     ## <== try smaller value if mode collapse
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

discriminator =  dcgan.DiscriminatorDG2()
generator =  dcgan.GeneratorDG2()

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

dcgan.run_gan_code(dls, adversarial, discriminator=discriminator, generator=generator, results_dir="results_DG2")

