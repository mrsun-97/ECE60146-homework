# -*- coding: utf-8 -*-

__version__   = '2.2.2'
__author__    = "Avinash Kak (kak@purdue.edu)"
__date__      = '2022-March-5'
__url__       = 'https://engineering.purdue.edu/kak/distDLS/DLStudio-2.2.2.html'
__copyright__ = "(C) 2022 Avinash Kak. Python Software Foundation."


__doc__ = '''


  You are looking at the AdversarialLearning co-class file in the DLStudio module.
  For the overall documentation on DLStudio, visit:

           https://engineering.purdue.edu/kak/distDLS/



INTRODUCTION TO ADVERSARIAL LEARNING FOR DATA MODELING:

    The modern excitement in adversarial learning for data modeling began with
    the paper "Generative Adversarial Nets" by Goodfellow, et al. Such
    learning involves two networks, a Discriminator and a Generator. we can
    think of the Discriminator as a function D(x,θ_d) where x is the image and
    θ_d the weights in the Discriminator network. The D(x,θ_d) function
    returns the probability that the input x is from the probability
    distribution that describes the training data. Similarly, we can think of
    the Generator as a function G(z,θ_g) that maps noise vectors to images
    that we want to look like the images in our training data. The vector θ_g
    represents the learnable parameters in the Generator network.

    We assume that the training images are described by some probability
    distribution that we denote p_data.  The goal of the Generator is to
    transform a noise vector, denoted z, into an image that should look like a
    training image.  Regarding z, we also assume that the noise vectors z are
    generated with a probability distribution p_Z(z).  Obviously, z is a
    realization of a vector random variable Z.  The output of the Generator
    consists of images that corresponds to some probability distribution that
    we will denote p_G.  So you can think of the Generator as a function that
    transforms the probability distribution p_Z into the distribution p_G.

    The question now is how do we train the Discriminator and the Generator
    networks.  The Discriminator is trained to maximize the probability of
    assigning the correct label to an input image that looks like it came from
    the same distribution as the training data.  That is, we want the
    parameters θ_d to maximize the following expectation:

          max   E           [log D(x)]                                                (1)
          θ_d    x ~ p_data

    The expression "x ~ p_data" means that x was pulled from the distribution
    p_data.  In other words, x is one of the training images.  While we are
    training D to exhibit the above behavior, we train the Generator for the
    following minimization:

          min   E         [log(1 − D(G(z)))]                                          (2)
          θ_g    z ~ p_Z

    Combining the two expressions shown above, we can express the combined
    optimization as:

                 _                                                                _
                |                                                                  |
      min  max  |  E           [log D(x)]      +     E         [log(1 − D(G(z)))]  | 
      θ_g  θ_d  |_  x ~ p_data                         z ~ p_Z                    _|
                                                                                      (3)

    Let's now see how we can translate this min-max form into a "protocol"
    for training the two networks.  For each training batch of images, we
    will first update the parameters in the Discriminator network and then in
    the Generator network.  If we use nn.BCELoss as the loss criterion, that
    will automatically take care of the logarithms in the expression shown
    above. The maximization of the first term simply requires that we use the
    target "1" for the network output D(x).  The maximization of the second
    term above is a bit more involved since it requires applying the
    Discriminator network to the output of the Generator for noise input.
    The second term also requires that we now use "-1" as the target for the
    Discriminator.  After we have calculated the two losses for the
    Discriminator, we can sum the losses and call backwards() on the sum for
    calculating the gradients of the loss with respect to its weights.  A
    subsequent call to the step() of the optimizer would update the weights
    in the Discriminator network.

    For the training required for the Generator, only the second term inside
    the square brackets above matters.  Since the goal now is to minimize the
    overall form, that translates into minimizing the second term. However,
    because the logarithm is a monotonically increasing function and also
    because the output D(G(z)) will always be between 0 and 1, so the needed
    minimization translates into maximizing D(G(z)) with respect to a target
    value of 1.  With 1 as the target, we again find the nn.BCELoss
    associated with D(G(z)).  We call backwards() on this loss --- making
    sure that we have turned off 'requires_grad()' on the Discriminator
    parameters as we are updating the Generator parameters. A subsequent call
    to the step() for the optimizer would update the weights in the Generator
    network.

DCGAN:

    The explanation presented above is how the training is carried out for
    the DCGAN implementations DG1 and DG2 in this file.  As stated elsewhere,
    DCGAN is short for "Deep Convolutional Generative Adversarial
    Network". It owes its origins to the paper "Unsupervised Representation
    Learning with Deep Convolutional Generative Adversarial Networks" by
    Radford et al.  It was the first fully convolutional network for GANs

    The main thing about the Discriminator and the Generator networks in
    DCGAN is that the network layers are based on "4-2-1" parameters. For
    example, each layer (except for the final layer) of the Discriminator
    network carries out a strided convolution with a 4x4 kernel, a 2x2 stride
    and a 1x1 padding.  The final layer is also convolutional, but its
    corresponding parameters are "4-1-0".  The output of the final layer is
    pushed through a sigmoid to yield a scalar value as the final output for
    each image in a batch.  As for the Generator network, as is the case with
    the Discriminator, you again see the 4-2-1 topology here.  A Generator's
    job is to transform a random noise vector into an image that is supposed
    to look like it came from the training dataset.


WGAN:

    As for the WGAN, as mentioned earlier, WGAN stands for "Wasserstein GAN"
    that is presented in the paper "Wasserstein GAN" by Arjovsky, Chintala,
    and Bottou.  WGAN is based on estimating the Wasserstein distance between
    the distribution that corresponds to the training images and the
    distribution that has been learned so far by the Generator.  The starting
    point for understanding this distance is its following definition:

                                             _            _
                                            |              |
     W(P,Q)   =     inf         E           | || x  - y || |                          (4)
                  γ ∈  π(P,Q)    (x,y) ~ γ  |_            _|

    where π(P,Q) is the set of all possible joint distributions γ(X,Y) over
    two random variables X and Y, with both random variables drawing samples
    from the set of outcomes for an underlying random experiment.  The joint
    distributions γ(X,Y) under consideration are only those for which the
    marginal with respect to X is P and the marginal with respect to Y is Q.
    The notation "(x,y) ~ γ" means that we are drawing the pair of samples
    (x,y) from the joint distribution γ(X,Y).  The joint distribution that
    yields the smallest value for the expectation of the norm shown above is
    the Wasserstein distance between the distributions P and Q.

    The definition of W(P,Q) shown above makes it computationally intractable
    because it requires that we carry out a Monte Carlo experiment that
    involves finding the norm "||x - y||" for EVERY possible pair (x,y)
    corresponding to EACH possible joint distribution in π(P,Q).  In
    addition, FOR EACH JOINT DISTRIBUTION γ, WE MUST FIND THE MEAN VALUE FOR
    THIS NORM.  What comes to our rescue is the following result from the
    Optimal Transport Theory:
    
                                    _                                  _
                                   |                                    |
    W(P,Q)   =         sup         |   E    f(x)      -      E    f(x)  |             (5)
                   ||f||_L =< 1    |    x~P                   y~Q       |
                                   |_                                  _|
    
    for all the 1-Lipschitz f: X→ R where X is the domain from which the
    elements x and y mentioned above are drawn and R is the range --- the set
    of all reals.  The subscript "L" in "||f||_L =< 1" is supposed to express
    the fact that we are talking about Lipschitz functions.  This equation
    can also be interpreted as: there is guaranteed to exist a 1-Lipschitz
    continuous function f() that when applied to the samples drawn from the
    distributions P and Q will yield the Wasserstein distance between the two
    distributions through the following formula:

                                  _                                         _
                                 |                                           |
    W(P,Q)   =        max        |   E     [f(x)]      -      E      [f(y)]  |        (6)
                 ||f||_L =< 1    |    x~P                      y~Q           |
                                 |_                                         _|

    The challenge then becomes one of having to learn this unknown function
    f().  This is the job that is assigned to the Critic Network in a WGAN.
    But how do we enforce the condition that we only seek functions whose
    continuity properties are those of 1-Lipschitz functions?  While,
    technically speaking, this remains an unsolved problem, the WGAN authors
    have presented an ad hoc approach that appears to work in practice.
    Their suggested approach consists of CLIPPING THE VALUES of the
    parameters of the Critic Network so that they lie in a very narrow band
    of values.

    The implementation of WGAN in this file does the same.

    What's shown above is the formula for the Wasserstein distance between a
    true distribution P and a learned approximation Q to it.  Assuming that
    the function f() needed for the calculation of the distance can be
    learned in a GAN-based framework, let C symbolically represent this
    learned function. Remember, our overarching goal remains that we need to
    also learn a Generator G that is capable of converting noise into samples
    that look like those from the distribution P.  The framework must learn G
    that MINIMIZES the Wasserstein distance between the true distribution P
    and its learned approximation Q.  At the same time, the GAN-based
    framework must discover a C that seeks to maximize the same distance (in
    the sense that the Critic learns how to maximally distrust the Generator
    G).  We thus end up with the following minimax objective for the learning
    framework:

                     _                                               _
                    |                                                 |
          min  max  |  E      [C(x)]      -     E         [C(G(z))]   |                  (7)
           G    C   |_  x ~ P                    z ~ p_Z             _|
                                                                                       
           
    In comparing this minimax objective with the one shown earlier in
    Equation (3), note that the two components of the argument to the minimax
    in Eq. (3) were additive, whereas we subtract them in the objective shown
    above.  In Eq. (3), we had a Discriminator in the GAN and our goal was to
    maximize its classification performance for images that look like they
    came from the true distribution P. On the other hand, the goal of the
    Critic here is to learn to maximize the Wasserstein distance between the
    true distribution P and its learned approximation Q.  Note that the
    distribution Q is for the images that are constructed by the Generator
    from the white-noise samples z drawn from the distribution p_z shown
    above.

    As far as the Critic is concerned, the maximization needed above can be
    achieved by using the following loss function:

                                     _    _              _    _
                                    |      |            |      |
           Critic Loss     =     E  | C(y) |    -    E  | C(x) |
                                y~Q |_    _|        x~P |_    _|

                                      _       _              _    _
                                     |         |            |      |
                           =     E   | C(G(z)) |     -    E | C(x) |                   (8)
                                z~p_z|_       _|         x~P|_    _|

    In the WGAN code shown in this file, this is accomplished by using a
    "gradient target" of -1 for the mean output of the Critic when it sees
    the images produced by the Generator and the "gradient target" of +1 for
    the mean output of the Critic when it sees the training data directly.
    By gradient here, we are talking about the gradient of the Wasserstein
    distance with respect to the learnable parameters. If we assume that the
    true values of the learnable parametes are related to their current
    estimates by an affine relationship, using these gradient targets makes
    sense.

WGAN with GP:

    The name extension "GP" stands for "Gradient Penalty".  It was shown by
    Gulrajani, Ahmed, Arjovsky, Dumouli, and Courville in their paper
    "Improved Training of Wasserstein GANs" (which followed the original WGAN
    publication) that implementing a 1-Lipschitz constraint with weight
    clipping biases the Critic towards learning simple probability
    distribution functions.  This paper showed how the performance of a WGAN
    could be improved by putting to use the theoretical property that the
    optimal WGAN critic has unit gradient norm (with respect to its inputs)
    almost everywhere under P and Q. [See Proposition 1, Corollary 1 of the
    paper cited in this paragraph.]

    In a WGAN-GP, we add a Gradient Penalty term to the Critic Loss that was
    shown earlier in Eq. (8):

                         _       _              _    _             _                      _  
                        |         |            |      |           |                        |^2 
   Critic Loss  =   E   | C(G(z)) |     -    E | C(x) |    +    λ.| ||∇_x̂ C(x̂) ||^2  − 1   |
                   z~p_z|_       _|         x~P|_    _|           |_                     _ |

                 \______________________________________/      \_____________________________/
                          Original Critic Loss                    The Gradient Penalty (GP)

                                                                                              (9)        

    The gradient in the GP term is of the output of the 1-Lipschitz function
    (meaning the Critic itself) with respect to its input.  Since the Critic
    sees both the training samples and those produced by the Generator at its
    input, for the purpose of calculating the gradient, we construct samples
    by taking a weighted sum of those drawn from the training data and those
    produced by the Generator:

               x̂  ←  εx  +  (1 − ε) x̃

    We feed such samples into the Critic and, based on its input-output
    values, estimate the gradient needed in the GP term in Eq. (9).

Finally:

    If you wish to use this module to learn about data modeling with
    adversarial learning, your entry points should be the following scripts
    in the ExamplesAdversarialLearning directory of the distro:

        1.  dcgan_DG1.py            

        2.  dcgan_DG2.py   

        3.  wgan_CG1.py             

        4.  wgan_with_gp_CG2.py             

    The first script demonstrates the DCGAN logic on the PurdueShapes5GAN
    dataset.  In order to show the sensitivity of the basic DCGAN logic to
    any variations in the network or the weight initializations, the second
    script introduces a small change in the discriniator network used by the
    first script.  The third script is a demonstration of using the
    Wasserstein distance for data modeling through adversarial learning. The
    last script shows how WGAN training can be improved with gradient
    penalty.  The results produced by these scripts (for the constructor
    options shown in the scripts) are included in a subdirectory named
    RVLCloud_based_results.

@endofdocs
'''


from DLStudio import DLStudio

import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvtF
import torch.optim as optim
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import glob                                                                                                           
import imageio                                                                                                        

#______________________________  AdversarialLearning Class Definition  ________________________________

class AdversarialLearning(object):

    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''AdversarialLearning constructor can only be called with keyword arguments for the following
                      keywords: epochs, learning_rate, batch_size, momentum, image_size, dataroot, path_saved_model, 
                      use_gpu, latent_vector_size, ngpu, dlstudio, device, LAMBDA, clipping_threshold, and beta1''')
        allowed_keys = 'dataroot','image_size','path_saved_model','momentum','learning_rate','epochs','batch_size', \
                       'classes','use_gpu','latent_vector_size','ngpu','dlstudio', 'beta1', 'LAMBDA', 'clipping_threshold'
        keywords_used = kwargs.keys()                                                                 
        for keyword in keywords_used:                                                                 
            if keyword not in allowed_keys:                                                           
                raise SyntaxError(keyword + ":  Wrong keyword used --- check spelling")  
        learning_rate = epochs = batch_size = convo_layers_config = momentum = None
        image_size = fc_layers_config = dataroot =  path_saved_model = classes = use_gpu = None
        latent_vector_size = ngpu = beta1 = LAMBDA = clipping_threshold = None
        if 'latent_vector_size' in kwargs            :   latent_vector_size = kwargs.pop('latent_vector_size')
        if 'ngpu' in kwargs                          :   ngpu  = kwargs.pop('ngpu')           
        if 'dlstudio' in kwargs                      :   dlstudio  = kwargs.pop('dlstudio')
        if 'beta1' in kwargs                         :   beta1  = kwargs.pop('beta1')
        if 'LAMBDA' in kwargs                        :   LAMBDA  = kwargs.pop('LAMBDA')
        if 'clipping_threshold' in kwargs            :   clipping_threshold = kwargs.pop('clipping_threshold')
        if latent_vector_size:
            self.latent_vector_size = latent_vector_size
        if ngpu:
            self.ngpu = ngpu
        if dlstudio:
            self.dlstudio = dlstudio
        if beta1:
            self.beta1 = beta1
        if LAMBDA:
            self.LAMBDA = LAMBDA
        if clipping_threshold:
            self.clipping_threshold = clipping_threshold 




    ####################################################################################################
    ########################     Start Definition of Inner Class DataModeling     ######################
    ####################################################################################################

    class DataModeling(nn.Module):             
        """
        The purpose of this class is demonstrate adversarial learning for constructing a neural-network
        based data model.  After you have constructed such a model for a dataset, it should be possible
        to create an instance of the model starting with just a noise vector.  When this idea is 
        applied to images, what that means is that you should be able to create an image that looks 
        very much like those in the training data.  Since the inputs to the neural network for 
        generating such "fakes" is pure noise, each instance you create in this manner would be different
        and yet look very much like what was in the training dataset.

        Class Path:  AdversarialLearning  ->   DataModeling
        """
        def __init__(self, dlstudio, adversarial, num_workers=2):
            super(AdversarialLearning.DataModeling, self).__init__()
            self.dlstudio = dlstudio
            self.adversarial  = adversarial
            self.num_workers = num_workers
            self.train_dataloader = None
            self.device = torch.device("cuda:0" if (torch.cuda.is_available() and adversarial.ngpu > 0) else "cpu")

        def show_sample_images_from_dataset(self, dlstudio):
            data = next(iter(self.train_dataloader))    
            real_batch = data[0]
            first_im = real_batch[0]
            self.dlstudio.display_tensor_as_image(torchvision.utils.make_grid(real_batch, padding=2, pad_value=1, normalize=True))

        def set_dataloader(self):
            dataset = torchvision.datasets.ImageFolder(root=self.dlstudio.dataroot,       
                           transform = tvt.Compose([                 
                                                tvt.Resize(self.dlstudio.image_size),             
                                                tvt.CenterCrop(self.dlstudio.image_size),         
                                                tvt.ToTensor(),                     
                                                tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),         
                           ]))
            self.train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.dlstudio.batch_size, 
                                                                                     shuffle=True, num_workers=self.num_workers)

        def weights_init(self,m):        
            """
            Uses the DCGAN initializations for the weights
            """
            classname = m.__class__.__name__     
            if classname.find('Conv') != -1:         
                nn.init.normal_(m.weight.data, 0.0, 0.02)      
            elif classname.find('BatchNorm') != -1:         
                nn.init.normal_(m.weight.data, 1.0, 0.02)       
                nn.init.constant_(m.bias.data, 0)      


        def calc_gradient_penalty(self, netC, real_data, fake_data):
            """
            Implementation by Marvin Cao: https://github.com/caogang/wgan-gp
            Marvin Cao's code is a PyTorch version of the Tensorflow based implementation provided by
            the authors of the paper "Improved Training of Wasserstein GANs" by Gulrajani, Ahmed, 
            Arjovsky, Dumouli,  and Courville.
            """
            BATCH_SIZE = self.dlstudio.batch_size
            LAMBDA = self.adversarial.LAMBDA
            epsilon = torch.rand(1).cuda()
            interpolates = epsilon * real_data + ((1 - epsilon) * fake_data)
            interpolates = interpolates.requires_grad_(True).cuda() 
            critic_interpolates = netC(interpolates)
            gradients = torch.autograd.grad(outputs=critic_interpolates, inputs=interpolates,
                                      grad_outputs=torch.ones(critic_interpolates.size()).cuda(), 
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
            return gradient_penalty
        

        def close_event(self):   
            '''
            from stackoverflow.com
            '''
            plt.close()


        class SkipBlockDN(nn.Module):
            """
            This is a building-block class for constructing the Critic Network for adversarial learning.  In
            general, such a building-bloc class would be used for designing a network that creates a
            resolution hierarchy for the input image in which each successive layer is a downsampled version
            of the input image with or without increasing the number of input channels.

            Class Path:  AdversarialLearning  ->   DataModeling  ->  SkipBlockDN
            """
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(AdversarialLearning.DataModeling.SkipBlockDN, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convo1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
                self.convo2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.downsampler1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2)
                self.downsampler2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=2)
            def forward(self, x):
                identity = x                                     
                out = self.convo1(x)                              
                out = self.bn1(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo2(out)                              
                    out = self.bn2(out)                              
                    out = torch.nn.functional.leaky_relu(out, negative_slope=0.2)
                if self.downsample:
                    out = self.downsampler2(out)
                    identity = self.downsampler1(identity)
                if self.skip_connections:
                    out += identity                              
                return out

        class SkipBlockUP(nn.Module):
            """
            This is also a building-block class meant for a CNN that requires upsampling the images at the 
            inputs to the successive layers.  I could use it in the Generator part of an Adversarial Network,
            but have not yet done so.

            Class Path:  AdversarialLearning  ->   DataModeling  ->  SkipBlockUP
            """
            def __init__(self, in_ch, out_ch, upsample=False, skip_connections=True):
                super(AdversarialLearning.DataModeling.SkipBlockUP, self).__init__()
                self.upsample = upsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convoT1 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
                self.convoT2 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.bn2 = nn.BatchNorm2d(out_ch)
                if upsample:
                    self.upsampler = nn.ConvTranspose2d(in_ch, out_ch, 1, stride=2, dilation=2, output_padding=1, padding=0)
            def forward(self, x):
                identity = x                                     
                out = self.convoT1(x)                              
                out = self.bn1(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convoT2(out)                              
                    out = self.bn2(out)                              
                    out = torch.nn.functional.leaky_relu(out, negative_slope=0.2)
                if self.upsample:
                    out = self.upsampler(out)
                    out = torch.nn.functional.leaky_relu(out, negative_slope=0.2)           
                    identity = self.upsampler(identity)
                    identity = torch.nn.functional.leaky_relu(identity, negative_slope=0.2) 
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out += identity[:,self.out_ch:,:,:]
                    out = torch.nn.functional.leaky_relu(out, negative_slope=0.2)           
                return out



        #####################################   Discriminator-Generator DG1   ######################################
        class DiscriminatorDG1(nn.Module):
            """
            This is an implementation of the DCGAN Discriminator. I refer to the DCGAN network topology as
            the 4-2-1 network.  Each layer of the Discriminator network carries out a strided
            convolution with a 4x4 kernel, a 2x2 stride and a 1x1 padding for all but the final
            layer. The output of the final convolutional layer is pushed through a sigmoid to yield
            a scalar value as the final output for each image in a batch.

            Class Path:  AdversarialLearning  ->   DataModeling  ->  DiscriminatorDG1
            """
            def __init__(self):
                super(AdversarialLearning.DataModeling.DiscriminatorDG1, self).__init__()
                self.conv_in = nn.Conv2d(  3,    64,      kernel_size=4,      stride=2,    padding=1)
                self.conv_in2 = nn.Conv2d( 64,   128,     kernel_size=4,      stride=2,    padding=1)
                self.conv_in3 = nn.Conv2d( 128,  256,     kernel_size=4,      stride=2,    padding=1)
                self.conv_in4 = nn.Conv2d( 256,  512,     kernel_size=4,      stride=2,    padding=1)
                self.conv_in5 = nn.Conv2d( 512,  1,       kernel_size=4,      stride=1,    padding=0)
                self.bn1  = nn.BatchNorm2d(128)
                self.bn2  = nn.BatchNorm2d(256)
                self.bn3  = nn.BatchNorm2d(512)
                self.sig = nn.Sigmoid()
            def forward(self, x):                 
                x = torch.nn.functional.leaky_relu(self.conv_in(x), negative_slope=0.2, inplace=True)
                x = self.bn1(self.conv_in2(x))
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
                x = self.bn2(self.conv_in3(x))
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
                x = self.bn3(self.conv_in4(x))
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
                x = self.conv_in5(x)
                x = self.sig(x)
                return x

        class GeneratorDG1(nn.Module):
            """
            This is an implementation of the DCGAN Generator. As was the case with the Discriminator network,
            you again see the 4-2-1 topology here.  A Generator's job is to transform a random noise
            vector into an image that is supposed to look like it came from the training dataset. (We refer 
            to the images constructed from noise vectors in this manner as fakes.)  As you will see later 
            in the "run_gan_code()" method, the starting noise vector is a 1x1 image with 100 channels.  In 
            order to output 64x64 output images, the network shown below use the Transpose Convolution 
            operator nn.ConvTranspose2d with a stride of 2.  If (H_in, W_in) are the height and the width 
            of the image at the input to a nn.ConvTranspose2d layer and (H_out, W_out) the same at the 
            output, the size pairs are related by
                         H_out   =   (H_in - 1) * s   +   k   -   2 * p
                         W_out   =   (W_in - 1) * s   +   k   -   2 * p
            
            were s is the stride and k the size of the kernel.  (I am assuming square strides, kernels, and 
            padding). Therefore, each nn.ConvTranspose2d layer shown below doubles the size of the input.

            Class Path:  AdversarialLearning  ->   DataModeling  ->  GeneratorDG1
            """
            def __init__(self):
                super(AdversarialLearning.DataModeling.GeneratorDG1, self).__init__()
                self.latent_to_image = nn.ConvTranspose2d( 100,   512,  kernel_size=4, stride=1, padding=0, bias=False)
                self.upsampler2 = nn.ConvTranspose2d( 512, 256, kernel_size=4, stride=2, padding=1, bias=False)
                self.upsampler3 = nn.ConvTranspose2d (256, 128, kernel_size=4, stride=2, padding=1, bias=False)
                self.upsampler4 = nn.ConvTranspose2d (128, 64,  kernel_size=4, stride=2, padding=1, bias=False)
                self.upsampler5 = nn.ConvTranspose2d(  64,  3,  kernel_size=4, stride=2, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(512)
                self.bn2 = nn.BatchNorm2d(256)
                self.bn3 = nn.BatchNorm2d(128)
                self.bn4 = nn.BatchNorm2d(64)
                self.tanh  = nn.Tanh()
            def forward(self, x):                     
                x = self.latent_to_image(x)
                x = torch.nn.functional.relu(self.bn1(x))
                x = self.upsampler2(x)
                x = torch.nn.functional.relu(self.bn2(x))
                x = self.upsampler3(x)
                x = torch.nn.functional.relu(self.bn3(x))
                x = self.upsampler4(x)
                x = torch.nn.functional.relu(self.bn4(x))
                x = self.upsampler5(x)
                x = self.tanh(x)
                return x
        ########################################   DG1 Definition ENDS   ###########################################



        #####################################   Discriminator-Generator DG2   ######################################
        class DiscriminatorDG2(nn.Module):
            """
            This is essentially the same network as the DCGAN for DG1, except for the extra layer
            "self.extra" shown below.  We also declare a batchnorm for this extra layer in the form
            of "self.bnX".  In the implementation of "forward()", we invoke the extra layer at the
            beginning of the network.

            Class Path:  AdversarialLearning  ->   DataModeling  ->  DiscriminatorDG2
            """            
            def __init__(self, skip_connections=True, depth=16):
                super(AdversarialLearning.DataModeling.DiscriminatorDG2, self).__init__()
                self.conv_in = nn.Conv2d(  3,    64,      kernel_size=4,      stride=2,    padding=1)
                self.extra =   nn.Conv2d(  64,    64,      kernel_size=4,      stride=1,    padding=2)
                self.conv_in2 = nn.Conv2d( 64,   128,     kernel_size=4,      stride=2,    padding=1)
                self.conv_in3 = nn.Conv2d( 128,  256,     kernel_size=4,      stride=2,    padding=1)
                self.conv_in4 = nn.Conv2d( 256,  512,     kernel_size=4,      stride=2,    padding=1)
                self.conv_in5 = nn.Conv2d( 512,  1,       kernel_size=4,      stride=1,    padding=0)
                self.bn1  = nn.BatchNorm2d(128)
                self.bn2  = nn.BatchNorm2d(256)
                self.bn3  = nn.BatchNorm2d(512)
                self.bnX  = nn.BatchNorm2d(64)
                self.sig = nn.Sigmoid()
            def forward(self, x):       
                x = torch.nn.functional.leaky_relu(self.conv_in(x), negative_slope=0.2, inplace=True)
                x = self.bnX(self.extra(x))
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
                x = self.bn1(self.conv_in2(x))
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
                x = self.bn2(self.conv_in3(x))
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
                x = self.bn3(self.conv_in4(x))
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
                x = self.conv_in5(x)
                x = self.sig(x)
                return x

        class GeneratorDG2(nn.Module):
            """
            The Generator for DG2 is exactly the same as for the DG1.  So please the comment block for that
            Generator.

            Class Path:  AdversarialLearning  ->   DataModeling  ->  GeneratorDG2
            """
            def __init__(self):
                super(AdversarialLearning.DataModeling.GeneratorDG2, self).__init__()
                self.latent_to_image = nn.ConvTranspose2d( 100,   512,  kernel_size=4, stride=1, padding=0, bias=False)
                self.upsampler2 = nn.ConvTranspose2d( 512, 256, kernel_size=4, stride=2, padding=1, bias=False)
                self.upsampler3 = nn.ConvTranspose2d (256, 128, kernel_size=4, stride=2, padding=1, bias=False)
                self.upsampler4 = nn.ConvTranspose2d (128, 64,  kernel_size=4, stride=2, padding=1, bias=False)
                self.upsampler5 = nn.ConvTranspose2d(  64,  3,  kernel_size=4, stride=2, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(512)
                self.bn2 = nn.BatchNorm2d(256)
                self.bn3 = nn.BatchNorm2d(128)
                self.bn4 = nn.BatchNorm2d(64)
                self.tanh  = nn.Tanh()
            def forward(self, x):                             
                x = self.latent_to_image(x)
                x = torch.nn.functional.relu(self.bn1(x))
                x = self.upsampler2(x)
                x = torch.nn.functional.relu(self.bn2(x))
                x = self.upsampler3(x)
                x = torch.nn.functional.relu(self.bn3(x))
                x = self.upsampler4(x)
                x = torch.nn.functional.relu(self.bn4(x))
                x = self.upsampler5(x)
                x = self.tanh(x)
                return x
        ########################################   DG2 Definition ENDS   ###########################################



        ##########################################   Critic-Generator CG1   ########################################
        class CriticCG1(nn.Module):
            """
            I have used the SkipBlockDN as a building block for the Critic network.  This I did with the hope
            that when time permits I may want to study the effect of skip connections on the behavior of the
            the critic vis-a-vis the Generator.  The final layer of the network is the same as in the 
            "official" GitHub implementation of Wasserstein GAN.  And, as in WGAN, I have used the leaky ReLU
            for activation.

            Class Path:  AdversarialLearning  ->   DataModeling  ->  CriticCG1
            """
            def __init__(self):
                super(AdversarialLearning.DataModeling.CriticCG1, self).__init__()
                self.conv_in = AdversarialLearning.DataModeling.SkipBlockDN(3, 64, downsample=True, skip_connections=True)
                self.conv_in2 = AdversarialLearning.DataModeling.SkipBlockDN( 64,   128,  downsample=True, skip_connections=False)
                self.conv_in3 = AdversarialLearning.DataModeling.SkipBlockDN(128,   256,  downsample=True, skip_connections=False)
                self.conv_in4 = AdversarialLearning.DataModeling.SkipBlockDN(256,   512,  downsample=True, skip_connections=False)
                self.conv_in5 = AdversarialLearning.DataModeling.SkipBlockDN(512,   1,  downsample=False, skip_connections=False)
                self.bn1  = nn.BatchNorm2d(128)
                self.bn2  = nn.BatchNorm2d(256)
                self.bn3  = nn.BatchNorm2d(512)
                self.final = nn.Linear(512, 1)
            def forward(self, x):              
                x = torch.nn.functional.leaky_relu(self.conv_in(x), negative_slope=0.2, inplace=True)
                x = self.bn1(self.conv_in2(x))
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
                x = self.bn2(self.conv_in3(x))
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
                x = self.bn3(self.conv_in4(x))
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
                x = self.conv_in5(x)
                x = x.view(-1)
                x = self.final(x)
                # The following will cause a single value to be returned for the entire batch. This is
                # required by the Expectation operator E() in Equation (6) in the doc section of this 
                # file (See the beginning of this file).  For the P distribution in that equation, we 
                # apply the Critic directly to the training images.  And, for the Q distribution, we apply
                # the Critic to the output of the Generator. We need to use the expection operator for both.
                x = x.mean(0)       
                x = x.view(1)
                return x

        class GeneratorCG1(nn.Module):
            """
            The Generator code remains the same as for the DCGAN shown earlier.

            Class Path:  AdversarialLearning  ->   DataModeling  ->  GeneratorCG1
            """
            def __init__(self):
                super(AdversarialLearning.DataModeling.GeneratorCG1, self).__init__()
                self.latent_to_image = nn.ConvTranspose2d( 100,   512,  kernel_size=4, stride=1, padding=0, bias=False)
                self.upsampler2 = nn.ConvTranspose2d( 512, 256, kernel_size=4, stride=2, padding=1, bias=False)
                self.upsampler3 = nn.ConvTranspose2d (256, 128, kernel_size=4, stride=2, padding=1, bias=False)
                self.upsampler4 = nn.ConvTranspose2d (128, 64,  kernel_size=4, stride=2, padding=1, bias=False)
                self.upsampler5 = nn.ConvTranspose2d(  64,  3,  kernel_size=4, stride=2, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(512)
                self.bn2 = nn.BatchNorm2d(256)
                self.bn3 = nn.BatchNorm2d(128)
                self.bn4 = nn.BatchNorm2d(64)
                self.tanh  = nn.Tanh()
            def forward(self, x):                   
                x = self.latent_to_image(x)
                x = torch.nn.functional.relu(self.bn1(x))
                x = self.upsampler2(x)
                x = torch.nn.functional.relu(self.bn2(x))
                x = self.upsampler3(x)
                x = torch.nn.functional.relu(self.bn3(x))
                x = self.upsampler4(x)
                x = torch.nn.functional.relu(self.bn4(x))
                x = self.upsampler5(x)
                x = self.tanh(x)
                return x
        ########################################   CG1 Definition ENDS   ###########################################


        ##########################################   Critic-Generator CG2   ########################################
        class CriticCG2(nn.Module):
            """
            For sake of variety, the Critic implementation in CG2 as the same Marvin Cao's Discriminator:

                    https://github.com/caogang/wgan-gp

            which in turn is the PyTorch version of the Tensorflow based Discriminator presented by the 
            authors of the paper "Improved Training of Wasserstein GANs" by Gulrajani, Ahmed, Arjovsky, Dumouli,
            and Courville.

            Class Path:  AdversarialLearning  ->   DataModeling  ->  CriticCG2
            """
            def __init__(self):
                super(AdversarialLearning.DataModeling.CriticCG2, self).__init__()
                self.DIM = 64
                main = nn.Sequential(
                    nn.Conv2d(3, self.DIM, 5, stride=2, padding=2),
                    nn.ReLU(True),
                    nn.Conv2d(self.DIM, 2*self.DIM, 5, stride=2, padding=2),
                    nn.ReLU(True),
                    nn.Conv2d(2*self.DIM, 4*self.DIM, 5, stride=2, padding=2),
                    nn.ReLU(True),
                )
                self.main = main
                self.output = nn.Linear(4*4*4*self.DIM, 1)
        
            def forward(self, input):
                input = input.view(-1, 3, 64, 64)
                out = self.main(input)
                out = out.view(-1, 4*4*4*self.DIM)
                out = self.output(out)
                out = out.mean(0)       
                out = out.view(1)
                return out
        
        class GeneratorCG2(nn.Module):
            """
            The Generator code remains the same as for DG1 shown earlier.

            Class Path:  AdversarialLearning  ->   DataModeling  ->  GeneratorCG2
            """
            def __init__(self):
                super(AdversarialLearning.DataModeling.GeneratorCG2, self).__init__()
                self.latent_to_image = nn.ConvTranspose2d( 100,   512,  kernel_size=4, stride=1, padding=0, bias=False)
                self.upsampler2 = nn.ConvTranspose2d( 512, 256, kernel_size=4, stride=2, padding=1, bias=False)
                self.upsampler3 = nn.ConvTranspose2d (256, 128, kernel_size=4, stride=2, padding=1, bias=False)
                self.upsampler4 = nn.ConvTranspose2d (128, 64,  kernel_size=4, stride=2, padding=1, bias=False)
                self.upsampler5 = nn.ConvTranspose2d(  64,  3,  kernel_size=4, stride=2, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(512)
                self.bn2 = nn.BatchNorm2d(256)
                self.bn3 = nn.BatchNorm2d(128)
                self.bn4 = nn.BatchNorm2d(64)
                self.tanh  = nn.Tanh()
            def forward(self, x):                   
                x = self.latent_to_image(x)
                x = torch.nn.functional.relu(self.bn1(x))
                x = self.upsampler2(x)
                x = torch.nn.functional.relu(self.bn2(x))
                x = self.upsampler3(x)
                x = torch.nn.functional.relu(self.bn3(x))
                x = self.upsampler4(x)
                x = torch.nn.functional.relu(self.bn4(x))
                x = self.upsampler5(x)
                x = self.tanh(x)
                return x
        ########################################   CG2 Definition ENDS   ###########################################



        ############################################################################################################
        ##  The training routines follow, first for a GAN constructed using either the DG1 and or the DG2 
        ##  Discriminator-Generator Networks, and then for a WGAN constructed using either the CG1 or the CG2
        ##  Critic-Generator Networks.
        ############################################################################################################

        def run_gan_code(self, dlstudio, adversarial, discriminator, generator, results_dir):
            """
            This function is meant for training a Discriminator-Generator based Adversarial Network.  
            The implementation shown uses several programming constructs from the "official" DCGAN 
            implementations at the PyTorch website and at GitHub. 

            Regarding how to set the parameters of this method, see the following script

                         dcgan_DG1.py

            in the "ExamplesAdversarialLearning" directory of the distribution.
            """
            dir_name_for_results = results_dir
            if os.path.exists(dir_name_for_results):
                files = glob.glob(dir_name_for_results + "/*")
                for file in files:
                    if os.path.isfile(file):
                        os.remove(file)
                    else:
                        files = glob.glob(file + "/*")
                        list(map(lambda x: os.remove(x), files))
            else:
                os.mkdir(dir_name_for_results)
            #  Set the number of channels for the 1x1 input noise vectors for the Generator:
            nz = 100
            netD = discriminator.to(self.device)
            netG = generator.to(self.device)
            #  Initialize the parameters of the Discriminator and the Generator networks according to the
            #  definition of the "weights_init()" method:
            netD.apply(self.weights_init)
            netG.apply(self.weights_init)
            #  We will use a the same noise batch to periodically check on the progress made for the Generator:
            fixed_noise = torch.randn(self.dlstudio.batch_size, nz, 1, 1, device=self.device)          
            #  Establish convention for real and fake labels during training
            real_label = 1   
            fake_label = 0         
            #  Adam optimizers for the Discriminator and the Generator:
            optimizerD = optim.Adam(netD.parameters(), lr=dlstudio.learning_rate, betas=(adversarial.beta1, 0.999))    
            optimizerG = optim.Adam(netG.parameters(), lr=dlstudio.learning_rate, betas=(adversarial.beta1, 0.999))
            #  Establish the criterion for measuring the loss at the output of the Discriminator network:
            criterion = nn.BCELoss()
            #  We will use these lists to store the results accumulated during training:
            img_list = []                               
            G_losses = []                               
            D_losses = []                               
            iters = 0                                   
            print("\n\nStarting Training Loop...\n\n")      
            start_time = time.perf_counter()            
            for epoch in range(dlstudio.epochs):        
                g_losses_per_print_cycle = []           
                d_losses_per_print_cycle = []           
                # For each batch in the dataloader
                for i, data in enumerate(self.train_dataloader, 0):         
                    #  As indicated in the DCGAN part of the doc section at the beginning of this file, the GAN
                    #  training boils down to carrying out a max-min optimization. Each iterative step
                    #  of the max part results in updating the Discriminator parameters and each iterative 
                    #  step of the min part results in the updating of the Generator parameters.  For each 
                    #  batch of the training data, we first do max and then do min.  Since the max operation 
                    #  affects both terms of the criterion shown in the doc section, it has two parts: In the
                    #  first part we apply the Discriminator to the training images using 1.0 as the target; 
                    #  and, in the second part, we supply to the Discriminator the output of the Generator 
                    #  and use -1.0 as the target. In what follows, the Discriminator is being applied to 
                    #  the training images:
                    netD.zero_grad()    
                    real_images_in_batch = data[0].to(self.device)     
                    #  Need to know how many images we pulled in since at the tailend of the dataset, the 
                    #  number of images may not equal the user-specified batch size:
                    b_size = real_images_in_batch.size(0)  
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)  
                    output = netD(real_images_in_batch).view(-1)  
                    lossD_for_reals = criterion(output, label)                                                   
                    lossD_for_reals.backward()                                                                   
            
                    #  That brings us the second part of what it takes to carry out the max operation on the
                    #  DCGAN criterion shown in the doc section at the beginning of this file.  This part
                    #  calls for applying the Discriminator to the images produced by the Generator from noise:
                    noise = torch.randn(b_size, nz, 1, 1, device=self.device)    
                    fakes = netG(noise) 
                    label.fill_(fake_label) 
                    output = netD(fakes.detach()).view(-1)  
                    lossD_for_fakes = criterion(output, label)    
                    lossD_for_fakes.backward()          
                    lossD = lossD_for_reals + lossD_for_fakes    
                    d_losses_per_print_cycle.append(lossD)  
                    optimizerD.step()  
            
                    #  That brings to the min part of the max-min optimization described in the doc section
                    #  at the beginning of this file.  The min part requires that we minimize "1 - D(G(z))"
                    #  which, since D is constrained to lie in the interval (0,1), requires that we maximize
                    #  D(G(z)).  We accomplish that by applying the Discriminator to the output of the 
                    #  Generator and use 1 as the target for each image:
                    netG.zero_grad()   
                    label.fill_(real_label)  
                    output = netD(fakes).view(-1)   
                    lossG = criterion(output, label)          
                    g_losses_per_print_cycle.append(lossG) 
                    lossG.backward()    
                    optimizerG.step() 
                    
                    if i % 100 == 99:                                                                           
                        current_time = time.perf_counter()                                                      
                        elapsed_time = current_time - start_time                                                
                        mean_D_loss = torch.mean(torch.FloatTensor(d_losses_per_print_cycle))                   
                        mean_G_loss = torch.mean(torch.FloatTensor(g_losses_per_print_cycle))                   
                        print("[epoch=%d/%d   iter=%4d   elapsed_time=%5d secs]     mean_D_loss=%7.4f    mean_G_loss=%7.4f" % 
                                      ((epoch+1),dlstudio.epochs,(i+1),elapsed_time,mean_D_loss,mean_G_loss))   
                        d_losses_per_print_cycle = []                                                           
                        g_losses_per_print_cycle = []                                                           
                    G_losses.append(lossG.item())                                                                
                    D_losses.append(lossD.item())                                                                
                    if (iters % 500 == 0) or ((epoch == dlstudio.epochs-1) and (i == len(self.train_dataloader)-1)):   
                        with torch.no_grad():             
                            fake = netG(fixed_noise).detach().cpu()  
                        img_list.append(torchvision.utils.make_grid(fake, padding=1, pad_value=1, normalize=True))
                    iters += 1              

            #  At the end of training, make plots from the data in G_losses and D_losses:
            plt.figure(figsize=(10,5))    
            plt.title("Generator and Discriminator Loss During Training")    
            plt.plot(G_losses,label="G")    
            plt.plot(D_losses,label="D") 
            plt.xlabel("iterations")   
            plt.ylabel("Loss")         
            plt.legend()          
            plt.savefig(dir_name_for_results + "/gen_and_disc_loss_training.png") 
            plt.show()    
            #  Make an animated gif from the Generator output images stored in img_list:            
            images = []           
            for imgobj in img_list:  
                img = tvtF.to_pil_image(imgobj)  
                images.append(img) 
            imageio.mimsave(dir_name_for_results + "/generation_animation.gif", images, fps=5)
            #  Make a side-by-side comparison of a batch-size sampling of real images drawn from the
            #  training data and what the Generator is capable of producing at the end of training:
            real_batch = next(iter(self.train_dataloader)) 
            real_batch = real_batch[0]
            plt.figure(figsize=(15,15))  
            plt.subplot(1,2,1)   
            plt.axis("off")   
            plt.title("Real Images")    
            plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch.to(self.device), 
                                                   padding=1, pad_value=1, normalize=True).cpu(),(1,2,0)))  
            plt.subplot(1,2,2)                                                                             
            plt.axis("off")                                                                                
            plt.title("Fake Images")                                                                       
            plt.imshow(np.transpose(img_list[-1],(1,2,0)))                                                 
            plt.savefig(dir_name_for_results + "/real_vs_fake_images.png")                                 
            plt.show()                                                                                     


        def run_wgan_code(self, dlstudio, adversarial, critic, generator, results_dir):
            """
            This function is meant for training a CG1-based Critic-Generator WGAN.   The implementation
            shown uses several programming constructs from the WGAN implementation at GitHub by the
            original authors of the famous WGAN paper. I have also used several programming constructs 
            from the DCGAN code at PyTorch and GitHub.  Regarding how to set the parameters of this method, 
            see the following script in the "ExamplesAdversarialLearning" directory of the distribution:

                         wgan_CG1.py
            """
            dir_name_for_results = results_dir
            if os.path.exists(dir_name_for_results):
                files = glob.glob(dir_name_for_results + "/*")
                for file in files:
                    if os.path.isfile(file):
                        os.remove(file)
                    else:
                        files = glob.glob(file + "/*")
                        list(map(lambda x: os.remove(x), files))
            else:
                os.mkdir(dir_name_for_results)
            #  Set the number of channels for the 1x1 input noise vectors for the Generator:
            nz = 100
            netC = critic.to(self.device)
            netG = generator.to(self.device)
            #  Initialize the parameters of the Critic and the Generator networks according to the
            #  definition of the "weights_init()" method:
            netC.apply(self.weights_init)
            netG.apply(self.weights_init)
            #  We will use a the same noise batch to periodically check on the progress made for the Generator:
            fixed_noise = torch.randn(self.dlstudio.batch_size, nz, 1, 1, device=self.device)          
            #  These are for training the Critic, 'one' is for the part of the training with actual
            #  training images, and 'minus_one' is for the part based on the images produced by the 
            #  Generator:
            one = torch.FloatTensor([1]).to(self.device)
            minus_one = torch.FloatTensor([-1]).to(self.device)
            #  Adam optimizers for the Critic and the Generator:
            optimizerC = optim.Adam(netC.parameters(), lr=dlstudio.learning_rate, betas=(adversarial.beta1, 0.999))    
            optimizerG = optim.Adam(netG.parameters(), lr=dlstudio.learning_rate, betas=(adversarial.beta1, 0.999))
            img_list = []                               
            Gen_losses = []                               
            Cri_losses = []                               
            iters = 0                                   
            gen_iterations = 0
            print("\n\nStarting Training Loop.......[Be very patient at the beginning since the Critic must separately be taken through a few hundred iterations of training before you get to see anything displayed in your terminal window.  Depending on your hardware, it may take around 5 minutes. Subsequently, each 100 iterations will take just a few seconds. ]\n\n")      
            start_time = time.perf_counter()            
            dataloader = self.train_dataloader
            clipping_thresh = self.adversarial.clipping_threshold
            # For each epoch
            for epoch in range(dlstudio.epochs):        
                data_iter = iter(dataloader)
                i = 0
                ncritic   = 5
                #  We will first update the Critic.  As was stated in the doc section for the AdversarialLearning
                #  class at the beginning of this file, a minimization of the Wasserstein distance between 
                #  the distribution that describes the training data and the distribution that has been learned
                #  so far by the Generator can be translated into a maximization of the difference of the 
                #  average outputs of a 1-Lipschitz function as applied to the training images and as applied 
                #  to the output of the Generator.  Learning this 1-Lipschitz function is the job of the Critic. 
                #  Since the Critic and the Generator parameters must be updated independently, we start by
                #  turning on the "requires_grad" property of the Critic parameters:
                while i < len(dataloader):
                    for p in netC.parameters():
                        p.requires_grad = True          
                    if gen_iterations < 25 or gen_iterations % 500 == 0:    # the choices 25 and 500 are from WGAN
                        ncritic = 100
                    ic = 0
                    while ic < ncritic and i < len(dataloader):
                        ic += 1
                        for p in netC.parameters():
                            p.data.clamp_(-clipping_thresh, clipping_thresh)
                        #  The maximization needed for training the Critic, as shown in the doc section at the
                        #  beginning of this file, consists of two parts.  The first part involves applying the
                        #  Critic network to just the training images, with each image subject to a "gradient
                        #  target" of "-1".
                        netC.zero_grad()                                                                            
                        real_images_in_batch =  data_iter.next()
                        i += 1
                        real_images_in_batch =  real_images_in_batch[0].to(self.device)   
                        #  Need to know how many images we pulled in since at the tailend of the dataset, the 
                        #  number of images may not equal the user-specified batch size:
                        b_size = real_images_in_batch.size(0)   
                        #  Note that a single scalar is produced for all the data in a batch.  This is probably
                        #  the reason why what the Generator learns is somewhat fuzzy.
                        critic_for_reals_mean = netC(real_images_in_batch)
                        ## 'minus_one' is the gradient target:
                        critic_for_reals_mean.backward(minus_one)  
                        #  The second part of Critic training requires that we apply the Critic to the images
                        #  produced by the Generator for a fresh batch of input noise vectors. The output of 
                        #  the Critic for these images must be subject to the target "-1".
                        noise = torch.randn(b_size, nz, 1, 1, device=self.device)    
                        fakes = netG(noise)          
                        #  Again, a single number is produced for the whole batch:
                        critic_for_fakes_mean = netC(fakes)
                        ## 'one' is the gradient target:
                        critic_for_fakes_mean.backward(one)
                        wasser_dist = critic_for_reals_mean - critic_for_fakes_mean
                        loss_critic = critic_for_fakes_mean - critic_for_reals_mean
                        #  Update the Critic
                        optimizerC.step()   
                    #  That brings us to the training of the Generator.  First we must turn off the "requires_grad"
                    #  of the Critic parameters since the Critic and the Generator are to be updated independently:
                    for p in netC.parameters():
                        p.requires_grad = False
                    netG.zero_grad()                         
                    #  This is again a single scalar based characterization of the whole batch of the Generator images:
                    noise = torch.randn(b_size, nz, 1, 1, device=self.device)    
                    fakes = netG(noise)          
                    critic_for_fakes_mean = netC(fakes)
                    loss_gen = critic_for_fakes_mean
                    critic_for_fakes_mean.backward(minus_one)                       
                    #  Update the Generator
                    optimizerG.step()                                                                          
                    gen_iterations += 1
                    if i % (ncritic * 20) == 0:   
                        current_time = time.perf_counter()                                                            
                        elapsed_time = current_time - start_time                                                      
                        print("[epoch=%d/%d   i=%4d   el_time=%5d secs]     loss_critic=%7.4f   loss_gen=%7.4f   Wasserstein_dist=%7.4f" %  (epoch,dlstudio.epochs,i,elapsed_time,loss_critic.data[0], loss_gen.data[0], wasser_dist.data[0]))
                    Gen_losses.append(loss_gen.data[0])      
                    Cri_losses.append(loss_critic.data[0])   
                    
                    #  Get G's output on fixed_noise for the GIF animation:
                    if (iters % 500 == 0) or ((epoch == dlstudio.epochs-1) and (i == len(dataloader)-1)): 
                        with torch.no_grad():                                                                        
                            fake = netG(fixed_noise).detach().cpu()  
                        img_list.append(torchvision.utils.make_grid(fake, padding=1, pad_value=1, normalize=True))   
                    iters += 1                                                                                        
            
            #  At the end of training, make plots from the data in Gen_losses and Cri_losses:
            plt.figure(figsize=(10,5))                                                                             
            plt.title("Generator and Critic Loss During Training")                                          
            plt.plot(Gen_losses,label="G")                                                                           
            plt.plot(Cri_losses,label="C")                                                                           
            plt.xlabel("iterations")                                                                               
            plt.ylabel("Loss")                                                                                     
            plt.legend()                                                                                           
            plt.savefig(dir_name_for_results + "/gen_and_critic_loss_training.png")                                  
            plt.show()                                                                                             
            #  Make an animated gif from the Generator output images stored in img_list:            
            images = []                                                                                            
            for imgobj in img_list:                                                                                
                img = tvtF.to_pil_image(imgobj)                                                                    
                images.append(img)                                                                                 
            imageio.mimsave(dir_name_for_results + "/generation_animation.gif", images, fps=5)                     
            
            #  Make a side-by-side comparison of a batch-size sampling of real images drawn from the
            #  training data and what the Generator is capable of producing at the end of training:
            real_batch = next(iter(dataloader))                                                        
            real_batch = real_batch[0]
            plt.figure(figsize=(15,15))                                                                           
            plt.subplot(1,2,1)                                                                                    
            plt.axis("off")                                                                                       
            plt.title("Real Images")                                                                              
            plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch.to(self.device), 
                                               padding=1, pad_value=1, normalize=True).cpu(),(1,2,0)))  
            plt.subplot(1,2,2)                                                                             
            plt.axis("off")                                                                                
            plt.title("Fake Images")                                                                       
            plt.imshow(np.transpose(img_list[-1],(1,2,0)))                                                 
            plt.savefig(dir_name_for_results + "/real_vs_fake_images.png")                                 
            plt.show()                                                                                     


        def run_wgan_with_gp_code(self, dlstudio, adversarial, critic, generator, results_dir):
            """
            This function is meant for training a CG2-based Critic-Generator WGAN. Regarding how 
            to set the parameters of this method, see the following script in the 
            "ExamplesAdversarialLearning" directory of the distribution:

                         wgan_with_gp_CG2.py
            """
            dir_name_for_results = results_dir
            if os.path.exists(dir_name_for_results):
                files = glob.glob(dir_name_for_results + "/*")
                for file in files:
                    if os.path.isfile(file):
                        os.remove(file)
                    else:
                        files = glob.glob(file + "/*")
                        list(map(lambda x: os.remove(x), files))
            else:
                os.mkdir(dir_name_for_results)
            #  Set the number of channels for the 1x1 input noise vectors for the Generator:
            nz = 100
            netC = critic.to(self.device)
            netG = generator.to(self.device)
            #  Initialize the parameters of the Critic and the Generator networks according to the
            #  definition of the "weights_init()" method:
            netC.apply(self.weights_init)
            netG.apply(self.weights_init)
            #  We will use a the same noise batch to periodically check on the progress made for the Generator:
            fixed_noise = torch.randn(self.dlstudio.batch_size, nz, 1, 1, device=self.device)          
            #  These are for training the Critic, 'one' is for the part of the training with actual
            #  training images, and 'minus_one' is for the part based on the images produced by the 
            #  Generator:
            one = torch.FloatTensor([1]).to(self.device)
            minus_one = torch.FloatTensor([-1]).to(self.device)
            #  Adam optimizers for the Critic and the Generator:
            optimizerC = optim.Adam(netC.parameters(), lr=dlstudio.learning_rate, betas=(adversarial.beta1, 0.999))    
            optimizerG = optim.Adam(netG.parameters(), lr=dlstudio.learning_rate, betas=(adversarial.beta1, 0.999))
            img_list = []                               
            Gen_losses = []                               
            Cri_losses = []                               
            iters = 0                                   
            gen_iterations = 0
            start_time = time.perf_counter()            
            dataloader = self.train_dataloader
            # For each epoch
            for epoch in range(dlstudio.epochs):        
                data_iter = iter(dataloader)
                i = 0
                ncritic   = 5
                #  In this version of WGAN training, we enforce the 1-Lipschitz condition on the function
                #  being learned by the Critic by requiring that the partial derivatives of the output of
                #  the Critic with respect to its input equal one in magnitude. This is referred as imposing
                #  a Gradient Penalty on the learning by the Critic.  As in the previous training
                #  function, we start by turning on the "requires_grad" property of the Critic parameters:
                while i < len(dataloader):
                    for p in netC.parameters():
                        p.requires_grad = True          
                    ic = 0
                    while ic < ncritic and i < len(dataloader):
                        ic += 1
                        #  The first two parts of what it takes to train the Critic are the same as for
                        #  a regular WGAN.  We want to train the Critic to recognize the training images and,
                        #  at the same time, the Critic should try to not believe the output of the Generator.
                        netC.zero_grad()                                                                            
                        real_images_in_batch =  data_iter.next()
                        i += 1
                        real_images_in_batch =  real_images_in_batch[0].to(self.device)   
                        #  Need to know how many images we pulled in since at the tailend of the dataset, the 
                        #  number of images may not equal the user-specified batch size:
                        b_size = real_images_in_batch.size(0)   
                        #  Note that a single scalar is produced for all the data in a batch.  
                        critic_for_reals_mean = netC(real_images_in_batch)     ## this is a batch based mean
                        #  The gradient target is 'minus_one'.  Note that the gradient here is one of output of 
                        #  the network with respect to the learnable parameters:
                        critic_for_reals_mean.backward(minus_one)     
                        #  The second part of Critic training requires that we apply the Critic to the images
                        #  produced by the Generator from a fresh batch of input noise vectors.
                        noise = torch.randn(b_size, nz, 1, 1, device=self.device)    
                        fakes = netG(noise)          
                        #  Again, a single number is produced for the whole batch:
                        critic_for_fakes_mean = netC(fakes.detach())
                        #  The gradient target is 'one'.  Note that the gradient here is one of output of 
                        #  the network with respect to the learnable parameters:
                        critic_for_fakes_mean.backward(one)         
                        #  For the third part of Critic training, we need to first estimate the Gradient Penalty
                        #  of the function being learned by the Critics with respect to the input to the function.
                        gradient_penalty = self.calc_gradient_penalty(netC, real_images_in_batch, fakes)
                        gradient_penalty.backward()               
                        loss_critic = critic_for_fakes_mean - critic_for_reals_mean + gradient_penalty
                        wasser_dist = critic_for_reals_mean - critic_for_fakes_mean                
                        #  Update the Critic
                        optimizerC.step()   

                    #  That brings us to the training of the Generator.  First we must turn off the "requires_grad"
                    #  of the Critic parameters since the Critic and the Generator are to be updated independently:
                    for p in netC.parameters():
                        p.requires_grad = False
                    netG.zero_grad()                         
                    #  This is again a single scalar based characterization of the whole batch of the Generator images:
                    noise = torch.randn(b_size, nz, 1, 1, device=self.device)    
                    fakes = netG(noise)          
                    critic_for_fakes_mean = netC(fakes)
                    loss_gen = critic_for_fakes_mean
                    #  The gradient target is 'minus_one'.  Note that the gradient here is one of output of the network
                    #  with respect to the learnable parameters:
                    loss_gen.backward(minus_one)      
                    #  Update the Generator
                    optimizerG.step()                                                                          
                    gen_iterations += 1
                    if i % (ncritic * 20) == 0:   
                        current_time = time.perf_counter()                                                            
                        elapsed_time = current_time - start_time                                                      
                        print("[epoch=%d/%d   i=%4d   el_time=%5d secs]     loss_critic=%7.4f   loss_gen=%7.4f   Wasserstein_dist=%7.4f" %  (epoch+1,dlstudio.epochs,i,elapsed_time,loss_critic.data[0], loss_gen.data[0], wasser_dist.data[0]))
                    Gen_losses.append(loss_gen.data[0])      
                    Cri_losses.append(loss_critic.data[0])   
                    #  Get G's output on fixed_noise for the GIF animation:
                    if (iters % 500 == 0) or ((epoch == dlstudio.epochs-1) and (i == len(self.train_dataloader)-1)): 
                        with torch.no_grad():                                                                        
                            fake = netG(fixed_noise).detach().cpu()  
                        img_list.append(torchvision.utils.make_grid(fake, padding=1, pad_value=1, normalize=True))   
                    iters += 1                                                                                        
            
            #  At the end of training, make plots from the data in Gen_losses and Cri_losses:
            plt.figure(figsize=(10,5))                                                                             
            plt.title("Generator and Critic Loss During Training")                                          
            plt.plot(Gen_losses,label="G")                                                                           
            plt.plot(Cri_losses,label="C")                                                                           
            plt.xlabel("iterations")                                                                               
            plt.ylabel("Loss")                                                                                     
            plt.legend()                                                                                           
            plt.savefig(dir_name_for_results + "/gen_and_critic_loss_training.png")                                  
            plt.show()                                                                                             
            #  Make an animated gif from the Generator output images stored in img_list:            
            images = []                                                                                            
            for imgobj in img_list:                                                                                
                img = tvtF.to_pil_image(imgobj)                                                                    
                images.append(img)                                                                                 
            imageio.mimsave(dir_name_for_results + "/generation_animation.gif", images, fps=5)                     
            
            #  Make a side-by-side comparison of a batch-size sampling of real images drawn from the
            #  training data and what the Generator is capable of producing at the end of training:
            real_batch = next(iter(self.train_dataloader))                                                        
            real_batch = real_batch[0]
            plt.figure(figsize=(15,15))                                                                           
            plt.subplot(1,2,1)                                                                                    
            plt.axis("off")                                                                                       
            plt.title("Real Images")                                                                              
            plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch.to(self.device), 
                                               padding=1, pad_value=1, normalize=True).cpu(),(1,2,0)))  
            plt.subplot(1,2,2)                                                                             
            plt.axis("off")                                                                                
            plt.title("Fake Images")                                                                       
            plt.imshow(np.transpose(img_list[-1],(1,2,0)))                                                 
            plt.savefig(dir_name_for_results + "/real_vs_fake_images.png")                                 
            plt.show()                                                                                     
            

        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)


#_________________________  End of AdversarialLearning Class Definition ___________________________

#______________________________    Test code follows    _________________________________

if __name__ == '__main__': 
    pass
