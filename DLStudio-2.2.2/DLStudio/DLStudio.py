# -*- coding: utf-8 -*-

__version__   = '2.2.2'
__author__    = "Avinash Kak (kak@purdue.edu)"
__date__      = '2022-March-5'
__url__       = 'https://engineering.purdue.edu/kak/distDLS/DLStudio-2.2.2.html'
__copyright__ = "(C) 2022 Avinash Kak. Python Software Foundation."


__doc__ = '''

DLStudio.py

Version: ''' + __version__ + '''
   
Author: Avinash Kak (kak@purdue.edu)

Date: ''' + __date__ + '''


@tag_changes
CHANGE LOG:

  Version 2.2.2:

    This version of DLStudio presents my implementation of transformers in
    deep learning. You will find two transformer implementations in the
    Transformers co-class of DLStudio in the distribution directory:
    TransformerFG and TransformerPreLN.  "FG" in TransformerFG stands for
    "Transformer First Generation"; it is my implementation of the
    architecture presented originally in the seminal paper "Attention is All
    You Need" by Vaswani et el.  And the second, TransformerPreLN ("PreLN"
    stands for "Pre Layer Norm") is a small but important modification of the
    original idea that is based on the paper "On Layer Normalization in the
    Transformer Architecture" by Xiong et al.  I could have easily combined
    the two implementations with a small number of conditional statements to
    account for the differences, however I have chosen to keep them separate
    in order to make it easier for the two to evolve separately and to be used
    differently for educational purposes.

  Versions 2.1.7 through 2.2.1:

    These version numbers are for the stepping-stones in my journey into the
    world of transformers --- my experiments with how to best implement the
    different components of a transformer for educational purposes.  As things
    stand, these versions contain features that did not make into the public
    release version 2.2.2 on account of inadequate testing.  I may include
    those features in versions of DLStudio after 2.2.2.

  Version 2.1.6:

    All the changes are confined to the DataPrediction co-class of the
    DLStudio module.  After posting the previous version, I noticed that the
    quality of the code in DataPrediction was not up to par.  The new version
    presents a cleaned-up version of the DataPrediction class.

  Version 2.1.5:

    DLStudio has now been equipped with a new co-class named DataPrediction
    whose focus is solely on solving data prediction problems for time-series
    data.  A time-series consists of a sequence of observations recorded at
    regular intervals.  These could, for example, be the price of a stock
    share recorded every hour; the hourly recordings of electrical load at
    your local power utility company; the mean average temperature recorded on
    an annual basis; and so on.  We want to use the past observations to
    predict the value of the next one.  While data prediction has much in
    common with other forms of sequence based learning, it presents certain
    unique challenges of its own and those are with respect to (1) Data
    Normalization; (2) Input Data Chunking; and (3) Multi-dimensional encoding
    of the "datetime" associated with each observation in the time-series.

  Version 2.1.3:

    Some users of DLStudio have reported that when they run the WGAN code for
    adversarial learning, the dataloader sometimes hangs in the middle of a
    training run.  (A WGAN training session may involve as many as 500
    epochs.)  In trying to reproduce this issue, I discovered that the
    training loops always ran to completion if you set the number of workers
    in the dataloader to 0.  Version 2.1.3 makes it easier for you to specify
    the number of workers in your own scripts that call on the WGAN
    functionality in the AdversarialLearning class.

  Version 2.1.2:

    The adversarial learning part of DLStudio now includes a WGAN
    implementation that uses Gradient Penalty for the learning required by
    the Critic.  All the changes made are in the AdversarialLearning class at
    the top level of the module.

  Version 2.1.1:

    In order to make it easier to navigate through the large code base of the
    module, I am adopting the convention that "Network" in the name of a class
    be reserved for only those cases when a class actually implements a
    network.  This convention requires that the name of an encapsulating class
    meant for teaching/learning a certain aspect of deep learning not contain
    "Network" in it.  Therefore, in Version 2.1.1, I have changed the names of
    the top-level classes AdversarialNetworks and Seq2SeqNetworks to
    AdversarialLearning and Seq2SeqLearning, respectively.

  Version 2.1.0:

    I have reorganized the code base a bit to make it easier for DLStudio to
    grow in the future.  This I did by moving the sequence-to-sequence
    learning (seq2seq) code to a separate co-class of the main DLStudio class.
    The name of the new class is Seq2SeqLearning and it resides at the top
    level of the distribution.

  Version 2.0.9:

    With this version, DLStudio comes with educational material on
    sequence-to-sequence learning (seq2seq). To that end, I have included the
    following two new classes in DLStudio: (1) Seq2SeqWithLearnableEmbeddings
    for seq2seq with learnable embeddings; and (2)
    Seq2SeqWithPretrainedEmbeddings for doing the same with pre-trained
    embeddings. Although I have used word2vec for the case of pre-trained
    embeddings, you would be able to run the code with the Fasttext embeddings
    also.  Both seq2seq implementations include the attention mechanism based
    on my understanding of the original paper on the subject by Bahdanau, Cho,
    and Bengio. You will find this code in a class named Attention_BCB.  For
    the sake of comparison, I have also included an implementation of the the
    attention mechanism used in the very popular NLP tutorial by Sean
    Robertson.  You will find that code in a class named Attention_SR. To
    switch between these two attention mechanisms, all you have to do is to
    comment-out and uncomment a couple of lines in the DecoderRNN code.

  Version 2.0.8:

    This version pulls into DLStudio a very important idea in text processing
    and language modeling --- word embeddings.  That is, representing words by
    fixed-sized numerical vectors that are learned on the basis of their
    contextual similarities (meaning that if two words occur frequently in
    each other's context, they should have similar numerical representations).
    Use of word embeddings is demonstrated in DLStudio through an inner class
    named TextClassificationWithEmbeddings.  Using pre-trained word2vec
    embeddings, this new inner class can be used for experimenting with text
    classification, sentiment analysis, etc.

  Version 2.0.7:

    Made incremental improvements to the visualization of intermediate results
    during training.

  Version 2.0.6:

    This is a result of further clean-up of the code base in DLStudio.  The
    basic functionality provided by the module has not changed.

  Version 2.0.5:

    This version has a bug-fix for the training loop used for demonstrating
    the power of skip connections.  I have also cleaned up how the
    intermediate results produced during training are displayed in your
    terminal window.  In addition, I deleted the part of DLStudio that dealt
    with Autograd customization since that material is now in my
    ComputationalGraphPrimer module.

  Version 2.0.4:

    This version mostly changes the HTML formatting of this documentation
    page.  The code has not changed.

  Version 2.0.3:

    I have been experimenting with how to best incorporate adversarial
    learning in the DLStudio module. That's what accounts for the jump from
    the previous public release version 1.1.4 to new version 2.0.3.  The
    latest version comes with a separate class named AdversarialLearning for
    experimenting with different types of such networks for learning data
    models with adversarial learning and, subsequently, generating new
    instances of the data from the learned models. The AdversarialLearning
    class includes two Discriminator-Generator (DG) pairs and one
    Critic-Generator (CG) pair. Of the two DG pairs, the first is based on the
    logic of DCGAN, and the second a small modification of the first.  The CG
    pair is based on the logic of Wasserstein GAN.  This version of the module
    also comes with a new examples directory, ExamplesAdversarialLearning,
    that contains example scripts that show how you can call the different DG
    and CG pairs in the AdversarialLearning class.  Also included is a new
    dataset I have created, PurdueShapes5GAN-20000, that contains 20,000
    images of size 64x64 for experimenting with the GANs in this module.

  Version 1.1.4:

    This version has a new design for the text classification class
    TEXTnetOrder2.  This has entailed new scripts for training and testing
    when using the new version of that class. Also includes a fix for a bug
    discovered in Version 1.1.3

  Version 1.1.3:

    The only change made in this version is to the class GRUnet that is used
    for text classification.  In the new version, the final output of this
    network is based on the LogSoftmax activation.

  Version 1.1.2:

    This version adds code to the module for experimenting with recurrent
    neural networks (RNN) for classifying variable-length text input. With an
    RNN, a variable-length text input can be characterized with a hidden state
    vector of a fixed size.  The text processing capabilities of the module
    allow you to compare the results that you may obtain with and without
    using a GRU. For such experiments, this version also comes with a text
    dataset based on an old archive of product reviews made available by
    Amazon.

  Version 1.1.1:

    This version fixes the buggy behavior of the module when using the 'depth'
    parameter to change the size of a network.

  Version 1.1.0:

    The main reason for this version was my observation that when the training
    data is intentionally corrupted with a high level of noise, it is possible
    for the output of regression to be a NaN (Not a Number).  In my testing at
    noise levels of 20%, 50%, and 80%, while you do not see this problem when
    the noise level is 20%, it definitely becomes a problem when the noise
    level is at 50%.  To deal with this issue, this version includes the test
    'torch.isnan()' in the training and testing code for object detection.
    This version of the module also provides additional datasets with noise
    corrupted images with different levels of noise.  However, since the total
    size of the datasets now exceeds the file-size limit at
    'https://pypi.org', you'll need to download them separately from the link
    provided in the main documentation page.

  Version 1.0.9:

    With this version, you can now use DLStudio for experiments in semantic
    segmentation of images.  The code added to the module is in a new inner
    class that, as you might guess, is named SemanticSegmentation.  The
    workhorse of this inner class is a new implementation of the famous Unet
    that I have named mUnet --- the prefix "m" stands for "multi" for the
    ability of the network to segment out multiple objects simultaneously.
    This version of DLStudio also comes with a new dataset,
    PurdueShapes5MultiObject, for experimenting with mUnet.  Each image in
    this dataset contains a random number of selections from five different
    shapes --- rectangle, triangle, disk, oval, and star --- that are randomly
    scaled, oriented, and located in each image.

  Version 1.0.7:

    The main reason for creating this version of DLStudio is to be able to use
    the module for illustrating how to simultaneously carry out classification
    and regression (C&R) with the same convolutional network.  The specific
    C&R problem that is solved in this version is the problem of object
    detection and localization. You want a CNN to categorize the object in an
    image and, at the same time, estimate the bounding-box for the detected
    object. Estimating the bounding-box is referred to as regression.  All of
    the code related to object detection and localization is in the inner
    class DetectAndLocalize of the main module file.  Training a CNN to solve
    the detection and localization problem requires a dataset that, in
    addition to the class labels for the objects, also provides bounding-box
    annotations for the objects.  Towards that end, this version also comes
    with a new dataset called PurdueShapes5.  Another new inner class,
    CustomDataLoading, that is also included in Version 1.0.7 has the
    dataloader for the PurdueShapes5 dataset.

  Version 1.0.6:

    This version has the bugfix for a bug in SkipBlock that was spotted by a
    student as I was demonstrating in class the concepts related to the use of
    skip connections in deep neural networks.

  Version 1.0.5:

    This version includes an inner class, SkipConnections, for experimenting
    with skip connections to improve the performance of a deep network.  The
    Examples subdirectory of the distribution includes a script,
    playing_with_skip_connections.py, that demonstrates how you can experiment
    with SkipConnections.  The network class used by SkipConnections is named
    BMEnet with an easy-to-use interface for experimenting with networks of
    arbitrary depth.

  Version 1.0.4:

    I have added one more inner class, AutogradCustomization, to the module
    that illustrates how to extend Autograd if you want to endow it with
    additional functionality. And, most importantly, this version fixes an
    important bug that caused wrong information to be written out to the disk
    when you tried to save the learned model at the end of a training
    session. I have also cleaned up the comment blocks in the implementation
    code.

  Version 1.0.3:

    This is the first public release version of this module.


@tag_intro
INTRODUCTION:

    Every design activity involves mixing and matching things and doing so
    repeatedly until you have achieved the desired results.  The same thing
    is true of modern deep learning networks.  When you are working with a
    new data domain, it is likely that you would want to experiment with
    different network layouts that you may have dreamed of yourself or that
    you may have seen somewhere in a publication or at some web site.

    The goal of this module is to make it easier to engage in this process.
    The idea is that you would drop in the module a new network and you
    would be able to see right away the results you would get with the new
    network.

    This module also allows you to specify a network with a configuration
    string.  The module parses the string and creates the network.  In
    upcoming revisions of this module, I am planning to add additional
    features to this approach in order to make it more general and more
    useful for production work.


@tag2_skip
    SKIP CONNECTIONS:

    Starting with Version 1.0.6, you can now experiment with skip
    connections in a CNN to see how a deep network with this feature might
    yield improved classification results.  Deep networks suffer from the
    problem of vanishing gradients that degrades their performance.
    Vanishing gradients means that the gradients of the loss calculated in
    the early layers of a network become increasingly muted as the network
    becomes deeper.  An important mitigation strategy for addressing this
    problem consists of creating a CNN using blocks with skip connections.

    The code for using skip connections is in the inner class
    SkipConnections of the module.  And the network that allows you to
    construct a CNN with skip connections is named BMEnet.  As shown in the
    script playing_with_skip_connections.py in the Examples directory of
    the distribution, you can easily create a CNN with arbitrary depth just
    by using the constructor option "depth" for BMEnet. The basic block of
    the network constructed in this manner is called SkipBlock which, very
    much like the BasicBlock in ResNet-18, has a couple of convolutional
    layers whose output is combined with the input to the block.

    Note that the value given to the "depth" constructor option for the
    BMEnet class does NOT translate directly into the actual depth of the
    CNN. [Again, see the script playing_with_skip_connections.py in the
    Examples directory for how to use this option.] The value of "depth" is
    translated into how many instances of SkipBlock to use for constructing
    the CNN.

    If you want to use DLStudio for learning how to create your own
    versions of SkipBlock-like shortcuts in a CNN, your starting point
    should be the following script in the Examples directory of the distro:

                playing_with_skip_connections.py

    This script illustrates how to use the inner class BMEnet of the module
    for experimenting with skip connections in a CNN. As the script shows,
    the constructor of the BMEnet class comes with two options:
    skip_connections and depth.  By turning the first on and off, you can
    directly illustrate in a classroom setting the improvement you can get
    with skip connections.  And by giving an appropriate value to the
    "depth" option, you can show results for networks of different depths.

@tag2_detect
    OBJECT DETECTION AND LOCALIZATION:

    The code for how to solve the problem of object detection and
    localization with a CNN is in the inner classes DetectAndLocalize and
    CustomDataLoading.  This code was developed for version 1.0.7 of the
    module.  In general, object detection and localization problems are
    more challenging than pure classification problems because solving the
    localization part requires regression for the coordinates of the
    bounding box that localize the object.  If at all possible, you would
    want the same CNN to provide answers to both the classification and the
    regression questions and do so at the same time.  This calls for a CNN
    to possess two different output layers, one for classification and the
    other for regression.  A deep network that does exactly that is
    illustrated by the LOADnet classes that are defined in the inner class
    DetectAndLocalize of the DLStudio module.  [By the way, the acronym
    "LOAD" in "LOADnet" stands for "LOcalization And Detection".] Although
    you will find three versions of the LOADnet class inside
    DetectAndLocalize, for now only pay attention to the LOADnet2 class
    since that is the one I have worked with the most for creating the
    1.0.7 distribution.

    As you would expect, training a CNN for object detection and
    localization requires a dataset that, in addition to the class labels
    for the images, also provides bounding-box annotations for the objects
    in the images. Out of my great admiration for the CIFAR-10 dataset as
    an educational tool for solving classification problems, I have created
    small-image-format training and testing datasets for illustrating the
    code devoted to object detection and localization in this module.  The
    training dataset is named PurdueShapes5-10000-train.gz and it consists
    of 10,000 images, with each image of size 32x32 containing one of five
    possible shapes --- rectangle, triangle, disk, oval, and star. The
    shape objects in the images are randomized with respect to size,
    orientation, and color.  The testing dataset is named
    PurdueShapes5-1000-test.gz and it contains 1000 images generated by the
    same randomization process as used for the training dataset.  You will
    find these datasets in the "data" subdirectory of the "Examples"
    directory in the distribution.

    Providing a new dataset for experiments with detection and localization
    meant that I also needed to supply a custom dataloader for the dataset.
    Toward that end, Version 1.0.7 also includes another inner class named
    CustomDataLoading where you will my implementation of the custom
    dataloader for the PurdueShapes5 dataset.

    If you want to use DLStudio for learning how to write your own PyTorch
    code for object detection and localization, your starting point should
    be the following script in the Examples directory of the distro:

                object_detection_and_localization.py

    Execute the script and understand what functionality of the inner class
    DetectAndLocalize it invokes for object detection and localization.

@tag2_noisy
    NOISY OBJECT DETECTION AND LOCALIZATION:

    When the training data is intentionally corrupted with a high level of
    noise, it is possible for the output of regression to be a NaN (Not a
    Number).  Here is what I observed when I tested the LOADnet2 network at
    noise levels of 20%, 50%, and 80%: At 20% noise, both the labeling and
    the regression accuracies become worse compared to the noiseless case,
    but they would still be usable depending on the application.  For
    example, with two epochs of training, the overall classification
    accuracy decreases from 91% to 83% and the regression error increases
    from under a pixel (on the average) to around 3 pixels.  However, when
    the level of noise is increased to 50%, the regression output is often
    a NaN (Not a Number), as presented by 'numpy.nan' or 'torch.nan'.  To
    deal with this problem, Version 1.1.0 of the DLStudio module checks the
    output of the bounding-box regression before drawing the rectangles on
    the images.  

    If you wish to experiment with detection and localization in the
    presence of noise, your starting point should be the script

                noisy_object_detection_and_localization.py

    in the Examples directory of the distribution.  Note that you would
    need to download the datasets for such experiments directly from the
    link provided near the top of this documentation page.

@tag2_semantic
    SEMANTIC SEGMENTATION:

    The code for how to carry out semantic segmentation is in the inner
    class that is appropriately named SemanticSegmentation.  At its
    simplest, the purpose of semantic segmentation is to assign correct
    labels to the different objects in a scene, while localizing them at
    the same time.  At a more sophisticated level, a system that carries
    out semantic segmentation should also output a symbolic expression that
    reflects an understanding of the scene in the image that is based on
    the objects found in the image and their spatial relationships with one
    another.  The code in the new inner class is based on only the simplest
    possible definition of what is meant by semantic segmentation.
    
    The convolutional network that carries out semantic segmentation
    DLStudio is named mUnet, where the letter "m" is short for "multi",
    which, in turn, stands for the fact that mUnet is capable of segmenting
    out multiple object simultaneously from an image.  The mUnet network is
    based on the now famous Unet network that was first proposed by
    Ronneberger, Fischer and Brox in the paper "U-Net: Convolutional
    Networks for Biomedical Image Segmentation".  Their UNET extracts
    binary masks for the cell pixel blobs of interest in biomedical images.
    The output of UNET can therefore be treated as a pixel-wise binary
    classifier at each pixel position.  The mUnet class, on the other hand,
    is intended for segmenting out multiple objects simultaneously form an
    image. [A weaker reason for "m" in the name of the class is that it
    uses skip connections in multiple ways --- such connections are used
    not only across the two arms of the "U", but also also along the arms.
    The skip connections in the original Unet are only between the two arms
    of the U.  

    mUnet works by assigning a separate channel in the output of the
    network to each different object type.  After the network is trained,
    for a given input image, all you have to do is examine the different
    channels of the output for the presence or the absence of the objects
    corresponding to the channel index.

    This version of DLStudio also comes with a new dataset,
    PurdueShapes5MultiObject, for experimenting with mUnet.  Each image
    in this dataset contains a random number of selections from five
    different shapes, with the shapes being randomly scaled, oriented, and
    located in each image.  The five different shapes are: rectangle,
    triangle, disk, oval, and star.

    Your starting point for learning how to use the mUnet network for
    segmenting images should be the following script in the Examples
    directory of the distro:

                semantic_segmentation.py

    Execute the script and understand how it uses the functionality packed
    in the inner class SemanticSegmentation for segmenting out the objects
    in an image.

@tag2_text
    TEXT CLASSIFICATION:

    Starting with Version 1.1.2, the module includes an inner class
    TextClassification that allows you to do simple experiments with neural
    networks with feedback (that are also called Recurrent Neural
    Networks).  With an RNN, textual data of arbitrary length can be
    characterized with a hidden state vector of a fixed size.  To
    facilitate text based experiments, this module also comes with text
    datasets derived from an old Amazon archive of product reviews.
    Further information regarding the datasets is in the comment block
    associated with the class SentimentAnalysisDataset. If you want to use
    DLStudio for experimenting with text, your starting points should be
    the following three scripts in the Examples directory of the
    distribution:

                text_classification_with_TEXTnet.py
                text_classification_with_TEXTnetOrder2.py
                text_classification_with_GRU.py

    The first of these is meant to be used with the TEXTnet network that
    does not include any protection against the vanishing gradients problem
    that a poorly designed RNN can suffer from.  The second script
    mentioned above is based on the TEXTnetOrder2 network and it includes
    rudimentary protection, but not enough to suffice for any practical
    application.  The purpose of TEXTnetOrder2 is to serve as an
    educational stepping stone to a GRU (Gated Recurrent Unit) network that
    is used in the third script listed above.

    Starting with Version 2.0.8, the Examples directory of DLStudio also
    includes the following three scripts that use the same learning
    networks as the corresponding scripts mentioned above but with word
    representations based on word2vec embeddings:

                text_classification_with_TEXTnet_word2vec.py
                text_classification_with_TEXTnetOrder2_word2vec.py
                text_classification_with_GRU_word2vec.py

    The pre-trained word2vec embeddings used in these scripts are accessed
    through the popular gensim library.


@tag2_adversarial
    DATA MODELING WITH ADVERSARIAL LEARNING:

    Starting with version 2.0.3, DLStudio includes a separate class named
    AdversarialLearning for experimenting with different adversarial
    learning approaches for data modeling.  Adversarial Learning consists
    of simultaneously training a Generator and a Discriminator (or, a
    Generator and a Critic) with the goal of getting the Generator to
    produce from pure noise images that look like those in the training
    dataset.  When Generator-Discriminator pairs are used, the
    Discriminator's job is to become an expert at recognizing the training
    images so it can let us know when the generator produces an image that
    does not look like what is in the training dataset.  The output of the
    Discriminator consists of the probability that the input to the
    discriminator is like one of the training images.

    On the other hand, when a Generator-Critic pair is used, the Critic's job
    is to become adept at estimating the distance between the distribution
    that corresponds to the training dataset and the distribution that has
    been learned by the Generator so far.  If the distance between the
    distributions is differentiable with respect to the weights in the
    networks, one can backprop the distance and update the weights in an
    iterative training loop.  This is roughly the idea of the Wasserstein GAN
    that is incorporated as a Critic-Generator pair CG1 in the
    AdversarialLearning class.

    The AdversarialLearning class includes two kinds of adversarial
    networks for data modeling: DCGAN and WGAN. 

    DCGAN is short for "Deep Convolutional Generative Adversarial Network",
    owes its origins to the paper "Unsupervised Representation Learning with
    Deep Convolutional Generative Adversarial Networks" by Radford et al.
    DCGAN was the first fully convolutional network for GANs (Generative
    Adversarial Network). CNN's typically have a fully-connected layer (an
    instance of nn.Linear) at the topmost level.  For the topmost layer in
    the Generator network, DCGAN uses another convolution layer that produces
    the final output image.  And for the topmost layer of the Discriminator,
    DCGAN flattens the output and feeds that into a sigmoid function for
    producing scalar value.  Additionally, DCGAN also gets rid of max-pooling
    for downsampling and instead uses convolutions with strides.  Yet another
    feature of a DCGAN is the use of batch normalization in all layers,
    except in the output layer of the Generator and the input layer of the
    Discriminator.  As the authors of DCGAN stated, while, in general, batch
    normalization stabilizes learning by normalizing the input to each layer
    to have zero mean and unit variance, applying BN at the output results in
    sample oscillation and model instability.  I have also retained in the
    DCGAN code the leaky ReLU activation recommended by the authors for the
    Discriminator.

    The other adversarial learning framework incorporated in
    AdversarialLearning is based on WGAN, which stands for Wasserstein GAN.
    This GAN was proposed in the paper "Wasserstein GAN" by Arjovsky,
    Chintala, and Bottou.  WGANs is based on estimating the Wasserstein
    distance between the distribution that corresponds to the training images
    and the distribution that has been learned so far by the Generator.  The
    authors of WGAN have shown that minimizing this distance in an iterative
    learning framework also involves solving a minimax problem involving a
    Critic and a Generator. The Critic's job is to become an expert at
    recognizing the training data while, at the same time, distrusting the
    output of the Generator. Unlike the Discriminator of a GAN, the Critic
    merely seeks to estimate the Wasserstein distance between the true
    distribution associated with the training data and the distribution being
    learned by the Generator.  As the Generator parameters are kept fixed,
    the Critic seems to update its parameters that maximize the Wasserstein
    distance between the true and the fake distributions. Subsequently, as
    the Critic parameters are kept fixed, the Generator updates its learnable
    parameters in an attempt to minimize the same distance.

    Estimation of the Wasserstein distance in the above logic requires for
    the Critic to learn a 1-Lipschitz function. DLStudio implements the
    following two strategies for this learning:

        --  Clipping the values of the learnable parameters of the Critic
            network to a user-specified interval;

        --  Penalizing the gradient of the norm of the Critic with respect
            to its input.

    The first of these is implemented in the function "run_gan_code()" in the
    file AdversarialLearning.py and the second in the function
    "run_wgan_with_gp_code()" in the same file.

    If you wish to use the DLStudio module to learn about data modeling with
    adversarial learning, your entry points should be the following scripts
    in the ExamplesAdversarialLearning directory of the distro:

        1.  dcgan_DG1.py            

        2.  dcgan_DG2.py   

        3.  wgan_CG1.py             

        4.  wgan_with_gp_CG2.py

    The first script demonstrates the DCGAN logic on the PurdueShapes5GAN
    dataset.  In order to show the sensitivity of the basic DCGAN logic to
    any variations in the network or the weight initializations, the second
    script introduces a small change in the network.  The third script is a
    demonstration of using the Wasserstein distance for data modeling through
    adversarial learning. The fourth script includes a gradient penalty in
    the critic logic called on by the third script.  The results produced by
    these scripts (for the constructor options shown in the scripts) are
    included in a subdirectory named RVLCloud_based_results.


@tag2_seq2seq
    SEQUENCE-TO-SEQUENCE LEARNING WITH ATTENTION

    Sequence-to-sequence learning (seq2seq) is about predicting an outcome
    sequence from a causation sequence, or, said another way, a target
    sequence from a source sequence.  Automatic machine translation is
    probably one of the most popular applications of seq2seq.  DLStudio uses
    English-to-Spanish translation to illustrate the programming idioms and
    the PyTorch structures you need for seq2seq.  To that end, Version 2.1.0
    of DLStudio includes a co-class (meaning a class that resides at the top
    level in the distribution) named Seq2SeqLearning that consists of the
    following two demonstration classes:

        1.  Seq2SeqWithLearnableEmbeddings

        2.  Seq2SeqWithPretrainedEmbeddings

    As their names imply, the first is for seq2seq with learnable
    embeddings and the second for seq2seq with pre-trained embeddings like
    word2vec or fasttext. 

    As mentioned above, the specific example of seq2seq addressed in my
    implementation code is translation from English to Spanish. (I chose this
    example because learning and keeping up with Spanish is one of my
    hobbies.)  In the Seq2SeqWithLearnableEmbeddings class, the learning
    framework learns the best embedding vectors to use for the two languages
    involved. On the other hand, in the Seq2SeqWithPretrainedEmbeddings
    class, I use the word2vec embeddings provided by Google for the source
    language.  As to why I use the pre-training embeddings for just the
    source language is explained in the main comment doc associated with the
    class Seq2SeqWithPretrainedEmbeddings.

    Any modern attempt at seq2seq must include attention.  This is done by
    incorporating a separate Attention network in the Encoder-Decoder
    framework needed for seq2seq learning.  The goal of the attention network
    is to modify the current hidden state in the decoder using the attention
    units produced previously by the encoder for the source language
    sentence.  The main Attention model I have used is based on my
    understanding of the attention mechanism proposed by Bahdanau, Cho, and
    Bengio. You will see this attention code in a class named Attention_BCB
    in the seq2seq implementations named above. I have also provided another
    attention class named Attention_SR that is my implementation of the
    attention mechanism in the very popular NLP tutorial by Sean Robertson at
    the PyTorch website.  The URLs to both these attention mechanisms are in
    my Week 14 lecture material on deep learning at Purdue.

    The following two scripts in the ExamplesSeq2SeqLearning directory are 
    your main entry points for experimenting with the seq2seq code in 
    DLStudio:

        1.  seq2seq_with_learnable_embeddings.py

        2.  seq2seq_with_pretrained_embeddings.py
    
    With the first script, the overall network will learn on its own the best
    embeddings to use for representing the words in the two languages.  And,
    with the second script, the pre-trained word2vec embeddings from Google
    are used for the source language while the system learns the embeddings
    for the target language.


@tag2_predict
    DATA PREDICTION

    Let's say you have a sequence of observations recorded at regular
    intervals.  These could, for example, be the price of a stock share
    recorded every hour; the hourly recordings of electrical load at your
    local power utility company; the mean average temperature recorded on an
    annual basis; and so on.  We want to use the past observations to predict
    the value of the next one.  Solving these types of problems is the focus
    of the DataPrediction co-class of DLStudio.

    As a problem, data prediction has much in common with text analytics and
    seq2seq processing, in the sense that the prediction at the next time
    instant must be based on the previous observations in a manner similar to
    what we do in text analytics where the next word is understood taking into
    account all the previous words in a sentence.  However, there are three
    significant differences between purely numerical data prediction problems
    and text-based problems:

    1) Data Normalization: As you know by this time, neural networks require
       that your input data be normalized to the [0,1] interval, assuming it
       consists of non-negative numbers, or the [-1,1] interval otherwise.
       When solving a sequential-data problem like text analytics, after you
       have normalized the input data (which is likely to consist of the
       numeric embeddings for the input words), you can forget about it.  You
       don't have that luxury when solving a data prediction problem.  As you
       would expect, the next value predicted by an algorithm must be at the
       same scale as the original input data.  This requires that the output
       of a neural-network-based prediction algorithm must be "inverse
       normalized".  And that, in turn, requires remembering the normalization
       parameters used in each channel of the input data.

    2) Input Data Chunking: The notion of a sentence that is important in text
       analytics does not carry over to the data prediction problem.  In
       general, you would want a prediction to be made using ALL of the past
       observations. When the sequential data available for training a
       predictor is arbitrarily long, as is the case with numerical data in
       general, you would need to decide how to "chunk" the data --- that is,
       how to extract sub-sequences from the data for the purpose of training
       a neural network.

    3) Datetime Conditioning: Time-series data typically includes a "datetime"
       stamp for each observation.  Representing datetime as a one-dimensional
       ever-increasing time value does not work for data prediction if the
       observations depend on the time of the day, the day of the week, the
       season of the year, and other such temporal effects.  Incorporating
       such effects in a prediction framework requires a multi-dimensional
       encoding of the datetime values.  See the doc page associated with the
       DataPrediction class for a longer explanation of this aspect of data
       prediction.

    Now that you understand how the data prediction problem differs from, say,
    the problem of text analytics, it is time for me to state my main goal in
    defining the DataPrediction class in the DLStudio module.  I actually have
    two goals:

    (a) To deepen your understanding of a GRU.  At this point, your
        understanding of a GRU is likely to be based on calling PyTorch's GRU
        in your own code.  Using a pre-programmed implementation for a GRU
        makes life easy and you also get a piece of highly optimized code that
        you can just call in your own code.  However, with a pre-programmed
        GRU, you are unlikely to get insights into how such an RNN is actually
        implemented.

    (b) To demonstrate how you can use a Recurrent Neural Network (RNN) for
        data prediction taking into account the data normalization, chunking,
        and datetime conditioning issues mentioned earlier.

    To address the first goal above, the DataPrediction class presented in
    this file is based on my pmGRU (Poor Man's GRU).  This GRU is my
    implementation of the "Minimal Gated Unit" GRU variant that was first
    presented by Joel Heck and Fathi Salem in their paper "Simplified Minimal
    Gated Unit Variations for Recurrent Neural Networks".  Its hallmark is
    that it combines the Update and the Reset gates of a regular GRU into a
    single gate called the Forget Gate.  You could say that pmGRU is a
    lightweight version of a regular GRU and its use may therefore lead to a
    slight loss of accuracy in the predictions.  You will find it educational
    to compare the performance you get with my pmGRU-based implementation with
    an implementation that uses PyTorch's GRU for the same dataset.

    Your main entry point for experimenting with the DataPrediction co-class
    is the following script in the ExamplesDataPrediction directory of the
    DLStudio distribution:

        power_load_prediction_with_pmGRU.py

    Before you can run this script, you would need to download the training
    dataset used in this example.  See the "For Data Prediction" part of the
    "The Datasets Included" section of the doc page for that.


@tag2_transformers
    TRANSFORMERS

    The goal of Transformer based learning is the same that of Seq2SeqLearning
    described earlier in this Introduction except that now you completely
    forgo recurrence. That is, you only use the mechanism of attention to
    translate sentences from a source language into sentences in the target
    language. For such applications, you need two forms of attention:
    self-attention and cross-attention.  Self-attention refers to the
    intra-sentence relationships between the words and cross-attention refers
    to the inter-sentence relationships between the words in a pair of
    sentences, one in the source language and the other in the target
    language. I have explained these concepts in great detail in the doc
    sections of the inner classes in the Transformers class.  In particular, I
    have explained the concept of the "dot-product" attention in which each
    word puts out three things: a Query Vector Q, a Key Vector K, and a Value
    Vector V. By taking the dot-product of the Query Vector Q of a word with
    the Key Vector K for all the words in a sentence, the neural network gets
    a measure of the extent to which each word in a sentence is important to
    every other word.  These dot-product values are then used as weights on
    the Value Vectors, V, for the individual words.  Cross attention works in
    a similar manner, except that now you take the dot-products of the Q
    vectors in the target-language sentence with the K vectors in the
    corresponding source-language sentence for producing the weight vectors
    that tell us how to weight the source-language Value Vectors vis-a-vis the
    words in the target language.

    You will see two different implementations of the transformer architecture
    in the Transformers co-class of DLStudio:

          TransformerFG
    and
          TransformerPreLN

    with the "FG" suffix standing for "First Generation" and the "PreLN"
    suffix for "Pre LayerNorm". TransformerFG is my implementation of the
    transformer architecture proposed in the famous paper by Vaswani et al.
    and TransformerPreLN my implementation of the same architecture but with
    the modification suggested by Xiong et al. for more stable learning.
    Since, the modification is small from an architectural standpoint, I could
    have combined both transformer types in the same implementation with some
    conditional logic to account for the differences.  However, I have chosen
    to keep them separate mostly for educational purposes.  Further details on
    these implementations are in the documentation blocks in the Transformers
    co-class.
 
    If you want to use my code for learning the main ideas related to how to
    create purely attention based networks, your starting point for that
    should be the following scripts in the ExamplesTransformers directory of
    the DLStudio distribution:

        seq2seq_with_transformerFG.py
        seq2seq_with_transformerPreLN.py

    These scripts uses the following English-Spanish sentence-pairs dataset

           en_es_xformer_8_90000.tar.gz

    that contains 90,000 pairs of English-Spanish sentences with the maximum
    number of words in each sentence limited to 8 words.  For processing by
    the attention networks, each sentence is enclosed in <SOS> and <EOS>
    tokens, with the former standing for "Start of Sentence" and the latter
    for "End of Sentence".


@tag_install
INSTALLATION:

    The DLStudio class was packaged using setuptools.  For
    installation, execute the following command in the source directory
    (this is the directory that contains the setup.py file after you have
    downloaded and uncompressed the package):
 
            sudo python3 setup.py install

    On Linux distributions, this will install the module file at a location
    that looks like

             /usr/local/lib/python3.7/dist-packages/

    If you do not have root access, you have the option of working directly
    off the directory in which you downloaded the software by simply
    placing the following statements at the top of your scripts that use
    the DLStudio class:

            import sys
            sys.path.append( "pathname_to_DLStudio_directory" )

    To uninstall the module, simply delete the source directory, locate
    where the DLStudio module was installed with "locate
    DLStudio" and delete those files.  As mentioned above,
    the full pathname to the installed version is likely to look like
    /usr/local/lib/python3.7/dist-packages/DLStudio*

    If you want to carry out a non-standard install of the
    DLStudio module, look up the on-line information on
    Disutils by pointing your browser to

              http://docs.python.org/dist/dist.html

@tag_usage
USAGE:

    If you want to specify a network with just a configuration string,
    your usage of the module is going to look like:

        from DLStudio import *
        
        convo_layers_config = "1x[128,3,3,1]-MaxPool(2) 1x[16,5,5,1]-MaxPool(2)"
        fc_layers_config = [-1,1024,10]
        
        dls = DLStudio(   dataroot = "/home/kak/ImageDatasets/CIFAR-10/",
                          image_size = [32,32],
                          convo_layers_config = convo_layers_config,
                          fc_layers_config = fc_layers_config,
                          path_saved_model = "./saved_model",
                          momentum = 0.9,
                          learning_rate = 1e-3,
                          epochs = 2,
                          batch_size = 4,
                          classes = ('plane','car','bird','cat','deer',
                                     'dog','frog','horse','ship','truck'),
                          use_gpu = True,
                          debug_train = 0,
                          debug_test = 1,
                      )
        
        configs_for_all_convo_layers = dls.parse_config_string_for_convo_layers()
        convo_layers = dls.build_convo_layers2( configs_for_all_convo_layers )
        fc_layers = dls.build_fc_layers()
        model = dls.Net(convo_layers, fc_layers)
        dls.show_network_summary(model)
        dls.load_cifar_10_dataset()
        dls.run_code_for_training(model)
        dls.run_code_for_testing(model)
                

    or, if you would rather experiment with a drop-in network, your usage
    of the module is going to look something like:

        dls = DLStudio(   dataroot = "/home/kak/ImageDatasets/CIFAR-10/",
                          image_size = [32,32],
                          path_saved_model = "./saved_model",
                          momentum = 0.9,
                          learning_rate = 1e-3,
                          epochs = 2,
                          batch_size = 4,
                          classes = ('plane','car','bird','cat','deer',
                                     'dog','frog','horse','ship','truck'),
                          use_gpu = True,
                          debug_train = 0,
                          debug_test = 1,
                      )
        
        exp_seq = DLStudio.ExperimentsWithSequential( dl_studio = dls )   ## for your drop-in network
        exp_seq.load_cifar_10_dataset_with_augmentation()
        model = exp_seq.Net()
        dls.show_network_summary(model)
        exp_seq.run_code_for_training(model)
        exp_seq.run_code_for_testing(model)

        
    This assumes that you copy-and-pasted the network you want to
    experiment with in a class like ExperimentsWithSequential that is
    included in the module.


@tag_constructor
CONSTRUCTOR PARAMETERS: 

    batch_size:  Carries the usual meaning in the neural network context.

    classes:  A list of the symbolic names for the classes.

    convo_layers_config: This parameter allows you to specify a convolutional network
                  with a configuration string.  Must be formatted as explained in the
                  comment block associated with the method
                  "parse_config_string_for_convo_layers()"

    dataroot: This points to where your dataset is located.

    debug_test: Setting it allow you to see images being used and their predicted
                 class labels every 2000 batch-based iterations of testing.

    debug_train: Does the same thing during training that debug_test does during
                 testing.

    epochs: Specifies the number of epochs to be used for training the network.

    fc_layers_config: This parameter allows you to specify the final
                 fully-connected portion of the network with just a list of
                 the number of nodes in each layer of this portion.  The
                 first entry in this list must be the number '-1', which
                 stands for the fact that the number of nodes in the first
                 layer will be determined by the final activation volume of
                 the convolutional portion of the network.

    image_size:  The heightxwidth size of the images in your dataset.

    learning_rate:  Again carries the usual meaning.

    momentum:  Carries the usual meaning and needed by the optimizer.

    path_saved_model: The path to where you want the trained model to be
                  saved in your disk so that it can be retrieved later
                  for inference.

    use_gpu: You must set it to True if you want the GPU to be used for training.


@tag_methods
PUBLIC METHODS:

    (1)  build_convo_layers()

         This method creates the convolutional layers from the parameters
         in the configuration string that was supplied through the
         constructor option 'convo_layers_config'.  The output produced by
         the call to 'parse_config_string_for_convo_layers()' is supplied
         as the argument to build_convo_layers().

    (2)  build_fc_layers()

         From the list of ints supplied through the constructor option
         'fc_layers_config', this method constructs the fully-connected
         portion of the overall network.

    (3)  check_a_sampling_of_images()        

         Displays the first batch_size number of images in your dataset.


    (4)  display_tensor_as_image()

         This method will display any tensor of shape (3,H,W), (1,H,W), or
         just (H,W) as an image. If any further data normalizations is
         needed for constructing a displayable image, the method takes care
         of that.  It has two input parameters: one for the tensor you want
         displayed as an image and the other for a title for the image
         display.  The latter parameter is default initialized to an empty
         string.

    (5)  load_cifar_10_dataset()

         This is just a convenience method that calls on Torchvision's
         functionality for creating a data loader.

    (6)  load_cifar_10_dataset_with_augmentation()             

         This convenience method also creates a data loader but it also
         includes the syntax for data augmentation.

    (7)  parse_config_string_for_convo_layers()

         As mentioned in the Introduction, DLStudio module allows you to
         specify a convolutional network with a string provided the string
         obeys the formatting convention described in the comment block of
         this method.  This method is for parsing such a string. The string
         itself is presented to the module through the constructor option
         'convo_layers_config'.

    (8)  run_code_for_testing()

         This is the method runs the trained model on the test data. Its
         output is a confusion matrix for the classes and the overall
         accuracy for each class.  The method has one input parameter which
         is set to the network to be tested.  This learnable parameters in
         the network are initialized with the disk-stored version of the
         trained model.

    (9)  run_code_for_training()

         This is the method that does all the training work. If a GPU was
         detected at the time an instance of the module was created, this
         method takes care of making the appropriate calls in order to
         transfer the tensors involved into the GPU memory.

    (10) save_model()

         Writes the model out to the disk at the location specified by the
         constructor option 'path_saved_model'.  Has one input parameter
         for the model that needs to be written out.

    (11) show_network_summary()

         Displays a print representation of your network and calls on the
         torchsummary module to print out the shape of the tensor at the
         output of each layer in the network. The method has one input
         parameter which is set to the network whose summary you want to
         see.


@tag_inner_classes
THE MAIN INNER CLASSES OF THE MODULE:

    By "inner classes" I mean the classes that are defined within the class
    file DLStudio.py in the DLStudio directory of the distribution.  The
    module also include what I have referred to as the Co-Classes in the
    next section.  A Co-Class resides at the same level of abstraction as
    the main DLStudio class defined in the DLStudio.py file.

    The purpose of the following two inner classes is to demonstrate how
    you can create a custom class for your own network and test it within
    the framework provided by the DLStudio module.

    (1)  class ExperimentsWithSequential

         This class is my demonstration of experimenting with a network
         that I found on GitHub.  I copy-and-pasted it in this class to
         test its capabilities.  How to call on such a custom class is
         shown by the following script in the Examples directory:

                     playing_with_sequential.py

    (2)  class ExperimentsWithCIFAR

         This is very similar to the previous inner class, but uses a
         common example of a network for experimenting with the CIFAR-10
         dataset. Consisting of 32x32 images, this is a great dataset for
         creating classroom demonstrations of convolutional networks.
         As to how you should use this class is shown in the following
         script

                    playing_with_cifar10.py

         in the Examples directory of the distribution.

    (4)  class SkipConnections

         This class is for investigating the power of skip connections in
         deep networks.  Skip connections are used to mitigate a serious
         problem associated with deep networks --- the problem of vanishing
         gradients.  It has been argued theoretically and demonstrated
         empirically that as the depth of a neural network increases, the
         gradients of the loss become more and more muted for the early
         layers in the network.

    (5)  class DetectAndLocalize

         The code in this inner class is for demonstrating how the same
         convolutional network can simultaneously solve the twin problems of
         object detection and localization.  Note that, unlike the previous
         four inner classes, class DetectAndLocalize comes with its own
         implementations for the training and testing methods. The main
         reason for that is that the training for detection and localization
         must use two different loss functions simultaneously, one for
         classification of the objects and the other for regression. The
         function for testing is also a bit more involved since it must now
         compute two kinds of errors, the classification error and the
         regression error on the unseen data. Although you will find a couple
         of different choices for the training and testing functions for
         detection and localization inside DetectAndLocalize, the ones I have
         worked with the most are those that are used in the following two
         scripts in the Examples directory:

              run_code_for_training_with_CrossEntropy_and_MSE_Losses()

              run_code_for_testing_detection_and_localization()

    (6)  class CustomDataLoading

         This is a testbed for experimenting with a completely grounds-up
         attempt at designing a custom data loader.  Ordinarily, if the
         basic format of how the dataset is stored is similar to one of the
         datasets that Torchvision knows about, you can go ahead and use
         that for your own dataset.  At worst, you may need to carry out
         some light customizations depending on the number of classes
         involved, etc.  However, if the underlying dataset is stored in a
         manner that does not look like anything in Torchvision, you have
         no choice but to supply yourself all of the data loading
         infrastructure.  That is what this inner class of the DLStudio
         module is all about.

    (7)  class SemanticSegmentation

         This inner class is for working with the mUnet convolutional network
         for semantic segmentation of images.  This network allows you to
         segment out multiple objects simultaneously from an image.  Each
         object type is assigned a different channel in the output of the
         network.  So, for segmenting out the objects of a specified type in
         a given input image, all you have to do is examine the corresponding
         channel in the output.

    (8)  class TextClassification

         The purpose of this inner class is to be able to use the DLStudio
         module for simple experiments in text classification.  Consider, for
         example, the problem of automatic classification of variable-length
         user feedback: you want to create a neural network that can label an
         uploaded product review of arbitrary length as positive or negative.
         One way to solve this problem is with a Recurrent Neural Network in
         which you use a hidden state for characterizing a variable-length
         product review with a fixed-length state vector.

    (9)  class TextClassificationWithEmbeddings

         This class has the same functionality as the previous text
         processing class except that now we use embeddings for representing
         the words.  Word embeddings are fixed-sized numerical vectors that
         are learned on the basis of the contextural similarity of the
         words. The implementation of this inner class uses the pre-trained
         300-element word2vec embeddings as made available by Google for 3
         million words and phrases drawn from the Google News dataset. In
         DLStudio, we access these embeddings through the popular gensim
         library.


@tag_coclasses
CO-CLASSES IN THE DLStudio MODULE:

    As stated at the beginning of the previous section, a Co-Class resides at
    the same level of abstraction in the distribution directory as the main
    DLStudio class. Each Co-Class is defined in a separate subdirectory at the
    top level of the distribution directory.  While the main DLStudio class is
    defined in a subdirectory of the same name, the other subdirectories that
    contain definitions for the co-classes are named AdversarialLearning,
    Seq2SeqLearning, DataPrediction, and Transformers.  What follows in this
    section are additional details regarding these co-classes:


@tag_coclass1
    ===================
    AdversarialLearning:
    ===================

    As I mentioned in the Introduction, the purpose of the AdversarialLearning
    class is to demonstrate probabilistic data modeling using Generative
    Adversarial Networks (GAN).  GANs use Discriminator-Generator or
    Critic-Generator pairs to learn probabilistic data models that can
    subsequently be used to create new image instances that look surprisingly
    similar to those in the training set.  At the moment, you will find the
    following three such pairs inside the AdversarialLearning class:

        1.  Discriminator-Generator DG1      ---  implements the DCGAN logic

        2.  Discriminator-Generator DG2      ---  a slight modification of the previous

        3.  Critic-Generator CG1             ---  implements the Wasserstein GAN logic

        4.  Critic-Generator CG2             ---  adds the Gradient Penalty to the 
                                                  Wasserstein GAN logic.

    In the ExamplesAdversarialLearning directory of the distro you will see
    the following scripts that demonstrate adversarial learning as
    incorporated in the above networks:

        1.  dcgan_DG1.py                     ---  demonstrates the DCGAN DG1

        2.  dcgan_DG2.py                     ---  demonstrates the DCGAN DG2

        3.  wgan_CG1.py                      ---  demonstrates the Wasserstein GAN CG1

        4.  wgan_with_gp_CG2.py              ---  demonstrates the Wasserstein GAN CG2

    All of these scripts use the training dataset PurdueShapes5GAN that
    consists of 20,000 images containing randomly shaped, randomly colored,
    and randomly positioned objects in 64x64 arrays.  The dataset comes in
    the form of a gzipped archive named
    "datasets_for_AdversarialLearning.tar.gz" that is provided under the link
    "Download the image dataset for AdversarialLearning" at the top of the
    HTML version of this doc page.  See the README in the
    ExamplesAdversarialLearning directory for how to unpack the archive.


@tag_coclass2
    ===============
    Seq2SeqLearning:
    ===============

    As mentioned earlier in the Introduction, sequence-to-sequence learning
    (seq2seq) is about predicting an outcome sequence from a causation
    sequence, or, said another way, a target sequence from a source sequence.
    Automatic machine translation is probably one of the most popular
    applications of seq2seq.  DLStudio uses English-to-Spanish translation to
    illustrate the programming idioms and the PyTorch structures you would
    need for writing your own code for seq2seq.

    Any attempt at seq2seq for machine translation must answer the following
    question at the outset: How to represent the words of a language for
    neural-network based processing? In general, you have two options: (1)
    Have your overall network learn on its own what are known as vector
    embeddings for the words; or (2) Use pre-trained embeddings as provided
    by word2vec or Fasttext.  

    After you have resolved the issue of word representation, your next
    challenge is how to implement the attention mechanism that you're going
    to need for aligning the similar grammatical units in the two
    languages. The seq2seq code demonstrated in this co-class uses the
    attention model proposed by Bahdanau, Cho, and Bengio in the form of a
    separate Attention class.  The name of this attention class is
    Attention_BCB.  In a separate attention class named Attention_SR, I have
    also included the attention mechanism used by Sean Robertson in his very
    popular NLP tutorial at the main PyTorch website.

    Seq2SeqLearning contains the following two inner classes for illustrating
    seq2seq:

        1.  Seq2SeqWithLearnableEmbeddings

        2.  Seq2SeqWithPretrainedEmbeddings

    In the first of these, Seq2SeqWithLearnableEmbeddings, the words
    embeddings are learned automatically by using the nn.Embeddings layer. On
    the other hand, in Seq2SeqWithPretrainedEmbeddings, I have used the
    word2vec embeddings for the source language English and allowed the
    system to learn the embeddings for the target language Spanish.

    In order to become familiar with these classes, your best entry points
    would be the following scripts in the ExamplesSeq2SeqLearning directory:

                seq2seq_with_learnable_embeddings.py

                seq2seq_with_pretrained_embeddings.py


@tag_coclass3
    ==============
    DataPrediction
    ==============

    As mentioned earlier in the Introduction, time-series data prediction
    differs from the more symbolic sequence-based learning frameworks with
    regard to the following: (1) Data normalization; (2) Data chunking; and
    (3) Datetime conditioning. The reason I mention data normalization is that
    now you have to remember the scaling parameters used for data
    normalization since you are going to need to inverse-normalize the
    predicted values. You would want to your predicted values to be at the
    same scale as the time-series observations.  The second issue, data
    chunking, refers to the fact that the notion of a "sentence" does not
    exist in time-series data.  What that implies that the user has to decide
    how to extract sequences from arbitrary long time-series data for training
    a prediction framework.  Finally, the the third issue, datetime
    conditioning, refers to creating a multi-dimensional encoding for the
    datetime stamp associated with each observation to account for the
    diurnal, weekly, seasonal, and other such temporal effects.

    The data prediction framework in the DataPrediction part of DLStudio is
    based on the following inner class:

        pmGRU

    for "Poor Man's GRU".  This GRU is my implementation of the "Minimal Gated
    Unit" GRU variant that was first presented by Joel Heck and Fathi Salem in
    their paper "Simplified Minimal Gated Unit Variations for Recurrent Neural
    Networks" and it combines the Update and the Reset gates of a regular GRU
    into a single gate called the Forget Gate.  

    My reason for using pmGRU is purely educational. While you are likely to
    use PyTorch's GRU for any production work that requires a GRU, using a
    pre-programmed piece of code makes it more difficult to gain insights into
    how the logic of a GRU (especially with regard to the gating action it
    needs) is actually implemented.  The implementation code shown for pmGRU
    is supposed to help remedy that.

    As I mentioned in the Introduction, your main entry point for
    experimenting with data prediction is the following script in the
    ExamplesDataPrediction directory of the DLStudio distribution:

        power_load_prediction_with_pmGRU.py

    However, before you can run this script, you would need to download the
    training dataset used in this example.  See the "For Data Prediction" part
    of the "The Datasets Included" section of the doc page for that.


@tag_coclass4
    ============
    Transformers
    ============

    The code in this co-class of DLStudio consists two slightly different
    implementations of the transformer architecture: TransformerFG and
    TransformerPreLN.  TransformerFG is my implementation of the architecture
    as conceptualized in the famous paper "Attention is All You Need" by
    Vaswani et el.  And TransformerPreLN is my implementation of the original
    idea along with the modifications suggested by Xiong et al. in their paper
    "On Layer Normalization in the Transformer Architecture" for more stable
    learning.  The two versions of transformers differ in only one respect:
    The placement of the LayerNorm in relation to the architectural components
    related to attention and the feedforward network.  Literally, the
    difference is small, yet its consequences on the stability of learning
    significant.

    The fundamentals of how the attention works in both TransformerFG and
    TransformerPreLN are exactly the same.  For self-attention, you associate
    a Query Vector Q_i and a Key Vector K_i with each word w_i in a sentence.
    For a given w_i, the dot product of its Q_i with the K_j vectors for all
    the other words w_j is a measure of how related w_i is to each w_j with
    regard to what's needed for the translation of a source sentence into the
    target sentence.  One more vector you associate with each word w_i is the
    Value Vector V_i.  The value vectors for the words in a sentence are
    weighted by the output of the activation nn.LogSoftMax applied to the
    dot-products.

    The self-attention mechanism described above is half of what goes into
    each base encoder of a transformer, the other half is a feedforward
    network (FFN). The overall encoder consists of a cascade of these base
    encoders.  In my implementation, I have referred to the overall encoder as
    the MasterEncoder.  The MasterDecoder also consists of a cascade of base
    decoders.  A base decoder is similar to a base encoder except for there
    being a layer of cross-attention interposed between the self-attention
    layer and the feedforward network.

    Referring to the attention half of each base encoder or a decoder as one
    half-unit and the FFN as the other half unit, the problem of vanishing
    gradients that would otherwise be caused by the depth of the overall
    network is mitigated by using LayerNorm and residual connections.  In
    TransformerFG, on the encoder side, LayerNorm is applied to the output of
    the self-attention layer and the residual connection wraps around both.
    Along the same lines, LayerNorm is applied to the output of FFN and the
    residual connection wraps around both.  

    In TransformerPreLN, on the other hand, LayerNorm is applied to the input
    to the self-attention layer and residual connection wraps around both.
    Similarly, LayerNorm is applied to the input to FFN and the residual
    connection wraps around both.  Similar considerations applied to the
    decoder side, except we now also have a layer of cross-attention
    interposed between the self-attention and FFN.
   
    As I mentioned in the Introduction, your main entry point for
    experimenting with the transformer code in DLStudio are the following two
    scripts in the ExamplesTransformers directory of the distribution:

        seq2seq_with_transformerFG.py
        seq2seq_with_transformerPreLN.py

    However, before you can run these scripts, you would need to download the
    training dataset used in these examples.  See the "For Transformers" part
    of the "The Datasets Included" section of this doc page for that.


@tag_examples_dir
Examples DIRECTORY:

    The Examples subdirectory in the distribution contains the following
    scripts:

    (1)  playing_with_reconfig.py

         Shows how you can specify a convolution network with a
         configuration string.  The DLStudio module parses the string
         constructs the network.

    (2)  playing_with_sequential.py

         Shows you how you can call on a custom inner class of the
         'DLStudio' module that is meant to experiment with your own
         network.  The name of the inner class in this example script is
         ExperimentsWithSequential

    (3)  playing_with_cifar10.py

         This is very similar to the previous example script but is based
         on the inner class ExperimentsWithCIFAR which uses more common
         examples of networks for playing with the CIFAR-10 dataset.

    (5)  playing_with_skip_connections.py

         This script illustrates how to use the inner class BMEnet of the
         module for experimenting with skip connections in a CNN. As the
         script shows, the constructor of the BMEnet class comes with two
         options: skip_connections and depth.  By turning the first on and
         off, you can directly illustrate in a classroom setting the
         improvement you can get with skip connections.  And by giving an
         appropriate value to the "depth" option, you can show results for
         networks of different depths.

    (6)  custom_data_loading.py

         This script shows how to use the custom dataloader in the inner
         class CustomDataLoading of the DLStudio module.  That custom
         dataloader is meant specifically for the PurdueShapes5 dataset
         that is used in object detection and localization experiments in
         DLStudio.

    (7)  object_detection_and_localization.py

         This script shows how you can use the functionality provided by
         the inner class DetectAndLocalize of the DLStudio module for
         experimenting with object detection and localization.  Detecting
         and localizing (D&L) objects in images is a more difficult problem
         than just classifying the objects.  D&L requires that your CNN
         make two different types of inferences simultaneously, one for
         classification and the other for localization.  For the
         localization part, the CNN must carry out what is known as
         regression. What that means is that the CNN must output the
         numerical values for the bounding box that encloses the object
         that was detected.  Generating these two types of inferences
         requires two different loss functions, one for classification and
         the other for regression.

    (8)  noisy_object_detection_and_localization.py

         This script in the Examples directory is exactly the same as the
         one described above, the only difference is that it calls on the
         noise-corrupted training and testing dataset files.  I thought it
         would be best to create a separate script for studying the effects
         of noise, just to allow for the possibility that the noise-related
         studies with DLStudio may evolve differently in the future.

    (9)  semantic_segmentation.py

         This script should be your starting point if you wish to learn how
         to use the mUnet neural network for semantic segmentation of
         images.  As mentioned elsewhere in this documentation page, mUnet
         assigns an output channel to each different type of object that
         you wish to segment out from an image. So, given a test image at
         the input to the network, all you have to do is to examine each
         channel at the output for segmenting out the objects that
         correspond to that output channel.

    (10) text_classification_with_TEXTnet.py

         This script is your first introduction in DLStudio to a Recurrent
         Neural Network, meaning a neural-network with feedback.  Such
         networks are needed for solving problems related to variable
         length input data in applications such as text classification,
         sentiment analysis, machine translation, etc.  Unfortunately,
         unless care is taken, the feedback in such networks results in
         long chains of dependencies and thus exacerbates the vanishing
         gradients problem.  The specific goal of this script is neural
         learning for automatic classification of product reviews.

    (11) text_classification_with_TEXTnet_word2vec.py

         This script uses the same learning network as in the previous
         script, but there is a big difference between the two.  The
         previous network uses one-hot vectors for representing the
         words. On the other hand, this script uses pre-trained word2vec
         embeddings.  These are fixed-sized numerical vectors that are
         learned on the basis of contextual similarities.
        
    (12) text_classification_with_TEXTnetOrder2.py

         As mentioned earlier for the script in item 10 above, the
         vanishing gradients problem becomes worse in neural networks with
         feedback.  One way to get around this problem is to use what's
         known as "gated recurrence".  This script uses the TEXTnetOrder2
         network as a stepping stone to a full-blown implementation of
         gating as provided by the nn.GRU class in item 14 below.

    (13) text_classification_with_TEXTnetOrder2_word2vec.py

         This script uses the same network as the previous script, but now
         we use the word2vec embeddings for representing the words.

    (14) text_classification_with_GRU.py

         This script demonstrates how one can use a GRU (Gated Recurrent
         Unit) to remediate one of the main problems associated with
         recurrence -- vanishing gradients in the long chains of
         dependencies created by feedback.

    (15) text_classification_with_GRU_word2vec.py

         While this script uses the same learning network as the previous
         one, the words are now represented by fixed-sized word2vec
         embeddings.


@tag_examples_advers_dir
ExamplesAdversarialLearning DIRECTORY:

    The ExamplesAdversarialLearning directory of the distribution contains
    the following scripts for demonstrating adversarial learning for data
    modeling:

        1.  dcgan_DG1.py            

        2.  dcgan_DG2.py   

        3.  wgan_CG1.py             

        4.  wgan_with_gp_CG2.py

    The first script demonstrates the DCGAN logic on the PurdueShapes5GAN
    dataset.  In order to show the sensitivity of the basic DCGAN logic to
    any variations in the network or the weight initializations, the second
    script introduces a small change in the network.  The third script is a
    demonstration of using the Wasserstein distance for data modeling through
    adversarial learning.  The fourth script adds a Gradient Penalty term to
    the Wasserstein Distance based logic of the third script.  The
    PurdueShapes5GAN dataset consists of 64x64 images with randomly shaped,
    randomly positioned, and randomly colored shapes.

    The results produced by these scripts (for the constructor options
    shown in the scripts) are included in a subdirectory named
    RVLCloud_based_results.  If you are just becoming familiar with the
    AdversarialLearning class of DLStudio, I'd urge you to run the script
    with the constructor options as shown and to compare your results with
    those that are in the RVLCloud_based_results directory.


@tag_examples_seq2seq_dir
ExamplesSeq2SeqLearning DIRECTORY:

    The ExamplesSeq2SeqLearning directory of the distribution contains the
    following scripts for demonstrating sequence-to-sequence learning:

    (1) seq2seq_with_learnable_embeddings.py

         This script demonstrates the basic PyTorch structures and idioms to
         use for seq2seq learning.  The application example addressed in the
         script is English-to-Spanish translation.  And the attention
         mechanism used for seq2seq is the one proposed by Bahdanau, Cho, and
         Bengio.  This network used in this example calls on the
         nn.Embeddings layer in the encoder to learn the embeddings for the
         words in the source language and a similar layer in the decoder to
         learn the embeddings to use for the target language.

    (2) seq2seq_with_pretrained_embeddings.py

         This script, also for seq2seq learning, differs from the previous
         one in only one respect: it uses Google's word2vec embeddings for
         representing the words in the source-language sentences (English).
         As to why I have not used at this time the pre-trained embeddings
         for the target language is explained in the main comment doc
         associated with the class Seq2SeqWithPretrainedEmbeddings.


@tag_examples_predict
ExamplesDataPrediction DIRECTORY:

    The ExampsleDataPrediction directory of the distribution contains the
    following script for demonstrating data prediction for time-series data:

        power_load_prediction_with_pmGRU.py

    This script uses a subset of the dataset provided by Kaggle for one of
    their machine learning competitions.  The dataset consists of over
    10-years worth of hourly electric load recordings made available by
    several utilities in the east and the midwest of the United States.  You
    can download this dataset from a link at the top of the main DLStudio doc
    page.


@tag_examples_xform
ExamplesTransformers DIRECTORY:

    The ExamplesTransformers directory of the distribution contains the
    following two scripts for experimenting with transformers:

        seq2seq_with_transformerFG.py 
        seq2seq_with_transformerPreLN.py         

    Both these scripts deal with English-to-Spanish translation in a manner
    similar to what's demonstrated by the code in the Seq2SeqLearning co-class
    and the example scripts associated with that co-class.


@tag_datasets
THE DATASETS INCLUDED: 

    [must be downloaded separately]

@tag2_main_dlstudio
    FOR THE MAIN DLStudio MODULE:

        Download the dataset archive 'datasets_for_DLStudio.tar.gz' through
        the link "Download the image datasets for the main DLStudio Class"
        provided at the top of this documentation page and store it in the
        'Example' directory of the distribution.  Subsequently, execute the
        following command in the 'Examples' directory:
    
            cd Examples
            tar zxvf datasets_for_DLStudio.tar.gz
    
        This command will create a 'data' subdirectory in the 'Examples'
        directory and deposit the datasets mentioned below in that
        subdirectory.
    
@tag3_dataset
        FOR OBJECT DETECTION AND LOCALIZATION:
    
        Training a CNN for object detection and localization requires training
        and testing datasets that come with bounding-box annotations. This
        module comes with the PurdueShapes5 dataset for that purpose.  I
        created this small-image-format dataset out of my admiration for the
        CIFAR-10 dataset as an educational tool for demonstrating
        classification networks in a classroom setting. You will find the
        following dataset archive files in the "data" subdirectory of the
        "Examples" directory of the distro:
    
            (1)  PurdueShapes5-10000-train.gz
                 PurdueShapes5-1000-test.gz
    
            (2)  PurdueShapes5-20-train.gz
                 PurdueShapes5-20-test.gz               
    
        The number that follows the main name string "PurdueShapes5-" is for
        the number of images in the dataset.  You will find the last two
        datasets, with 20 images each, useful for debugging your logic for
        object detection and bounding-box regression.
    
        As to how the image data is stored in the archives, please see the main
        comment block for the inner class CustomLoading in this file.
    
@tag3_dataset
        FOR DETECTING OBJECTS IN NOISE-CORRUPTED IMAGES:
    
        In terms of how the image data is stored in the dataset files, this
        dataset is no different from the PurdueShapes5 dataset described above.
        The only difference is that we now add varying degrees of noise to the
        images to make it more challenging for both classification and
        regression.
    
        The archive files you will find in the 'data' subdirectory of the
        'Examples' directory for this dataset are:
    
            (3)  PurdueShapes5-10000-train-noise-20.gz
                 PurdueShapes5-1000-test-noise-20.gz
    
            (4)  PurdueShapes5-10000-train-noise-50.gz
                 PurdueShapes5-1000-test-noise-50.gz
    
            (5)  PurdueShapes5-10000-train-noise-80.gz
                 PurdueShapes5-1000-test-noise-80.gz
    
        In the names of these six archive files, the numbers 20, 50, and 80
        stand for the level of noise in the images.  For example, 20 means 20%
        noise.  The percentage level indicates the fraction of the color value
        range that is added as randomly generated noise to the images.  The
        first integer in the name of each archive carries the same meaning as
        mentioned above for the regular PurdueShapes5 dataset: It stands for
        the number of images in the dataset.
    
@tag3_dataset
        FOR SEMANTIC SEGMENTATION:
    
        Showing interesting results with semantic segmentation requires images
        that contains multiple objects of different types.  A good semantic
        segmenter would then allow for each object type to be segmented out
        separately from an image.  A network that can carry out such
        segmentation needs training and testing datasets in which the images
        come up with multiple objects of different types in them. Towards that
        end, I have created the following dataset:

            (6) PurdueShapes5MultiObject-10000-train.gz
                PurdueShapes5MultiObject-1000-test.gz
    
            (7) PurdueShapes5MultiObject-20-train.gz
                PurdueShapes5MultiObject-20-test.gz
    
        The number that follows the main name string
        "PurdueShapes5MultiObject-" is for the number of images in the dataset.
        You will find the last two datasets, with 20 images each, useful for
        debugging your logic for semantic segmentation.
    
        As to how the image data is stored in the archive files listed above,
        please see the main comment block for the class
    
            PurdueShapes5MultiObjectDataset
    
        As explained there, in addition to the RGB values at the pixels that
        are stored in the form of three separate lists called R, G, and B, the
        shapes themselves are stored in the form an array of masks, each of
        size 64x64, with each mask array representing a particular shape. For
        illustration, the rectangle shape is represented by the first such
        array. And so on.
    
@tag3_dataset
        FOR TEXT CLASSIFICATION:
    
        My experiments tell me that, when using gated RNNs, the size of the
        vocabulary can significantly impact the time it takes to train a neural
        network for text modeling and classification.  My goal was to provide
        curated datasets extract from the Amazon user-feedback archive that
        would lend themselves to experimentation on, say, your personal laptop
        with a rudimentary GPU like the Quadro.  Here are the new datasets you
        can now download from the main documentation page for this module:
        
    
                 sentiment_dataset_train_200.tar.gz        vocab_size = 43,285
                 sentiment_dataset_test_200.tar.gz  
    
                 sentiment_dataset_train_40.tar.gz         vocab_size = 17,001
                 sentiment_dataset_test_40.tar.gz    
    
                 sentiment_dataset_train_400.tar.gz        vocab_size = 64,350
                 sentiment_dataset_test_400.tar.gz  
    
        As with the other datasets, the integer in the name of each dataset is
        the number of reviews collected from the 'positive.reviews' and the
        'negative.reviews' files for each product category.  Therefore, the
        dataset with 200 in its name has a total of 400 reviews for each
        product category.  Also provided are two datasets named
        "sentiment_dataset_train_3.tar.gz" and sentiment_dataset_test_3.tar.gz"
        just for the purpose of debugging your code.
    
        The last dataset, the one with 400 in its name, was added in Version
        1.1.3 of the module.

@tag2_dataset
    FOR Seq2Seq LEARNING:

        For sequence-to-sequence learning with DLStudio, you can download an
        English-Spanish translation corpus through the following archive:

            en_es_corpus_for_seq2sq_learning_with_DLStudio.tar.gz

        This data archive is a lightly curated version of the main dataset
        posted at "http://www.manythings.org/anki/" by the folks at
        "tatoeba.org".  My alterations to the original dataset consist mainly
        of expanding the contractions like "it's", "I'm", "don't", "didn't",
        "you'll", etc., into their "it is", "i am", "do not", "did not", "you
        will", etc. The original form of the dataset contains 417 such unique
        contractions.  Another alteration I made to the original data archive
        is to surround each sentence in both English and Spanish by the "SOS"
        and "EOS" tokens, with the former standing for "Start of Sentence"
        and the latter for "End of Sentence".

        Download the above archive in the ExamplesSeq2Seq2Learning directory
        and execute the following command in that directory:

            tar zxvf en_es_corpus_for_seq2sq_learning_with_DLStudio.tar.gz
    
        This command will create a 'data' subdirectory in the directory
        ExamplesSeq2Seq2Learning and deposit the following dataset archive in that 
        subdirectory:

            en_es_8_98988.tar.gz

        Now execute the following in the 'data' directory:

            tar zxvf en_es_8_98988.tar.gz

        With that, you should be able to execute the Seq2SeqLearning based
        scripts in the 'ExamplesSeq2SeqLearning' directory.


@tag2_dataset
    FOR ADVERSARIAL LEARNING:

        Download the dataset archive

            datasets_for_AdversarialLearning.tar.gz 

        through the link "Download the image dataset for
        AdversarialLearning" provided at the top of the HTML version of
        this doc page and store it in the 'ExamplesAdversarialLearning'
        directory of the distribution.  Subsequently, execute the following
        command in the directory 'ExamplesAdversarialLearning':
    
            tar zxvf datasets_for_AdversarialLearning.tar.gz
    
        This command will create a 'dataGAN' subdirectory and deposit the
        following dataset archive in that subdirectory:

            PurdueShapes5GAN-20000.tar.gz

        Now execute the following in the "dataGAN" directory:

            tar zxvf PurdueShapes5GAN-20000.tar.gz

        With that, you should be able to execute the adversarial learning
        based scripts in the 'ExamplesAdversarialLearning' directory.

@tag2_dataset
    FOR DATA PREDICTION:

        Download the dataset archive     

            dataset_for_DataPrediction.tar.gz             

        into the ExamplesDataPrediction directory of the DLStudio distribution.
        Next, execute the following command in that directory:

            tar zxvf dataset_for_DataPrediction.tar.gz 

        That will create data directory named "dataPred" in the
        ExamplesDataPrediction directory.  With that you should be able to 
        execute the data prediction script in that directory.


@tag2_dataset
    FOR TRANSFORMERS:

        Download the dataset archive

            en_es_corpus_for_learning_with_Transformers.tar.gz

        into the ExamplesTransformers directory of the DLStudio distribution.
        Next, execute the following command in that directory:

            tar zxvf en_es_corpus_for_learning_with_Transformers.tar.gz

        That will create a 'data' subdirectory in the ExamplesTransformers 
        directory and deposit in that subdirectory the following archives

            en_es_xformer_8_10000.tar.gz
            en_es_xformer_8_90000.tar.gz

        These are both derived from the same data source as in the dataset for
        the examples associated with the Seq2SeqLearning co-class.  The first
        has only 10,000 pars of English-Spanish sentences and meant primarily
        for debugging purposes.  The second contains 90000 pairs of such
        sentences.  The number '8' in the dataset names means that no sentence
        contains more than 8 real words.  With the "SOS" and "EOS" tokens
        used as sentence delimiters, the maximum number of words in each
        sentence in either language is 10.


@tag_bugs
BUGS:

    Please notify the author if you encounter any bugs.  When sending
    email, please place the string 'DLStudio' in the subject line to get
    past the author's spam filter.


@tag_ack
ACKNOWLEDGMENTS:

    Thanks to Praneet Singh and Noureldin Hendy for their comments related
    to the buggy behavior of the module when using the 'depth' parameter to
    change the size of a network. Thanks also go to Christina Eberhardt for
    reminding me that I needed to change the value of the 'dataroot'
    parameter in my Examples scripts prior to packaging a new distribution.
    Their feedback led to Version 1.1.1 of this module.  Regarding the
    changes made in Version 1.1.4, one of them is a fix for the bug found
    by Serdar Ozguc in Version 1.1.3. Thanks Serdar.

    Version 2.0.3: I owe thanks to Ankit Manerikar for many wonderful
    conversations related to the rapidly evolving area of generative
    adversarial networks in deep learning.  It is obviously important to
    read research papers to become familiar with the goings-on in an area.
    However, if you wish to also develop deep intuitions in those concepts,
    nothing can beat having great conversations with a strong researcher
    like Ankit.  Ankit is finishing his Ph.D. in the Robot Vision Lab at
    Purdue.

    Version 2.2.2: My laboratory's (RVL) journey into the world of
    transformers began with a series of lab seminars by Constantine Roros and
    Rahul Deshmukh.  Several subsequent conversations with them were
    instrumental in helping me improve the understanding I had gained from the
    seminars.  Additional conversations with Rahul about the issue of masking
    were important to how I eventually implemented those ideas in my code.


@tag_about_the_author
ABOUT THE AUTHOR:

    The author, Avinash Kak, is a professor of Electrical and Computer
    Engineering at Purdue University.  For all issues related to this
    module, contact the author at kak@purdue.edu If you send email, please
    place the string "DLStudio" in your subject line to get past the
    author's spam filter.


@tag_copyright
COPYRIGHT:

    Python Software Foundation License

    Copyright 2022 Avinash Kak

@endofdocs
'''


import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
import numpy as np
from PIL import ImageFilter
import numbers
import re
import math
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
import pymsgbox
import time
import logging

#______________________________  DLStudio Class Definition  ________________________________

class DLStudio(object):

    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''DLStudio constructor can only be called with keyword arguments for 
                      the following keywords: epochs, learning_rate, batch_size, momentum,
                      convo_layers_config, image_size, dataroot, path_saved_model, classes, 
                      image_size, convo_layers_config, fc_layers_config, debug_train, use_gpu, and 
                      debug_test''')
        learning_rate = epochs = batch_size = convo_layers_config = momentum = None
        image_size = fc_layers_config = dataroot =  path_saved_model = classes = use_gpu = None
        debug_train  = debug_test = None
        if 'dataroot' in kwargs                      :   dataroot = kwargs.pop('dataroot')
        if 'learning_rate' in kwargs                 :   learning_rate = kwargs.pop('learning_rate')
        if 'momentum' in kwargs                      :   momentum = kwargs.pop('momentum')
        if 'epochs' in kwargs                        :   epochs = kwargs.pop('epochs')
        if 'batch_size' in kwargs                    :   batch_size = kwargs.pop('batch_size')
        if 'convo_layers_config' in kwargs           :   convo_layers_config = kwargs.pop('convo_layers_config')
        if 'image_size' in kwargs                    :   image_size = kwargs.pop('image_size')
        if 'fc_layers_config' in kwargs              :   fc_layers_config = kwargs.pop('fc_layers_config')
        if 'path_saved_model' in kwargs              :   path_saved_model = kwargs.pop('path_saved_model')
        if 'classes' in kwargs                       :   classes = kwargs.pop('classes') 
        if 'use_gpu' in kwargs                       :   use_gpu = kwargs.pop('use_gpu') 
        if 'debug_train' in kwargs                   :   debug_train = kwargs.pop('debug_train') 
        if 'debug_test' in kwargs                    :   debug_test = kwargs.pop('debug_test') 
        if len(kwargs) != 0: raise ValueError('''You have provided unrecognizable keyword args''')
        if dataroot:
            self.dataroot = dataroot
        if convo_layers_config:
            self.convo_layers_config = convo_layers_config
        if image_size:
            self.image_size = image_size
        if fc_layers_config:
            self.fc_layers_config = fc_layers_config
#            if fc_layers_config[0] is not -1:
            if fc_layers_config[0] != -1:
                raise Exception("""\n\n\nYour 'fc_layers_config' construction option is not correct. """
                                """The first element of the list of nodes in the fc layer must be -1 """
                                """because the input to fc will be set automatically to the size of """
                                """the final activation volume of the convolutional part of the network""")
        if  path_saved_model:
            self.path_saved_model = path_saved_model
        if classes:
            self.class_labels = classes
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 1e-6
        if momentum:
            self.momentum = momentum
        if epochs:
            self.epochs = epochs
        if batch_size:
            self.batch_size = batch_size
        if use_gpu is not None:
            self.use_gpu = use_gpu
            if use_gpu is True:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda:0")
                else:
                    raise Exception("You requested GPU support, but there's no GPU on this machine")
            else:
                self.device = torch.device("cpu")
        if debug_train:                             
            self.debug_train = debug_train
        else:
            self.debug_train = 0
        if debug_test:                             
            self.debug_test = debug_test
        else:
            self.debug_test = 0
        self.debug_config = 0
#        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_gpu is False else "cpu")

    def parse_config_string_for_convo_layers(self):
        '''
        Each collection of 'n' otherwise identical layers in a convolutional network is 
        specified by a string that looks like:

                                 "nx[a,b,c,d]-MaxPool(k)"
        where 
                n      =  num of this type of convo layer
                a      =  number of out_channels                      [in_channels determined by prev layer] 
                b,c    =  kernel for this layer is of size (b,c)      [b along height, c along width]
                d      =  stride for convolutions
                k      =  maxpooling over kxk patches with stride of k

        Example:
                     "n1x[a1,b1,c1,d1]-MaxPool(k1)  n2x[a2,b2,c2,d2]-MaxPool(k2)"
        '''
        configuration = self.convo_layers_config
        configs = configuration.split()
        all_convo_layers = []
        image_size_after_layer = self.image_size
        for k,config in enumerate(configs):
            two_parts = config.split('-')
            how_many_conv_layers_with_this_config = int(two_parts[0][:config.index('x')])
            if self.debug_config:
                print("\n\nhow many convo layers with this config: %d" % how_many_conv_layers_with_this_config)
            maxpooling_size = int(re.findall(r'\d+', two_parts[1])[0])
            if self.debug_config:
                print("\nmax pooling size for all convo layers with this config: %d" % maxpooling_size)
            for conv_layer in range(how_many_conv_layers_with_this_config):            
                convo_layer = {'out_channels':None, 
                               'kernel_size':None, 
                               'convo_stride':None, 
                               'maxpool_size':None,
                               'maxpool_stride': None}
                kernel_params = two_parts[0][config.index('x')+1:][1:-1].split(',')
                if self.debug_config:
                    print("\nkernel_params: %s" % str(kernel_params))
                convo_layer['out_channels'] = int(kernel_params[0])
                convo_layer['kernel_size'] = (int(kernel_params[1]), int(kernel_params[2]))
                convo_layer['convo_stride'] =  int(kernel_params[3])
                image_size_after_layer = [x // convo_layer['convo_stride'] for x in image_size_after_layer]
                convo_layer['maxpool_size'] = maxpooling_size
                convo_layer['maxpool_stride'] = maxpooling_size
                image_size_after_layer = [x // convo_layer['maxpool_size'] for x in image_size_after_layer]
                all_convo_layers.append(convo_layer)
        configs_for_all_convo_layers = {i : all_convo_layers[i] for i in range(len(all_convo_layers))}
        if self.debug_config:
            print("\n\nAll convo layers: %s" % str(configs_for_all_convo_layers))
        last_convo_layer = configs_for_all_convo_layers[len(all_convo_layers)-1]
        out_nodes_final_layer = image_size_after_layer[0] * image_size_after_layer[1] * \
                                                                      last_convo_layer['out_channels']
        self.fc_layers_config[0] = out_nodes_final_layer
        self.configs_for_all_convo_layers = configs_for_all_convo_layers
        return configs_for_all_convo_layers


    def build_convo_layers(self, configs_for_all_convo_layers):
        conv_layers = nn.ModuleList()
        in_channels_for_next_layer = None
        for layer_index in configs_for_all_convo_layers:
            if self.debug_config:
                print("\n\n\nLayer index: %d" % layer_index)
            in_channels = 3 if layer_index == 0 else in_channels_for_next_layer
            out_channels = configs_for_all_convo_layers[layer_index]['out_channels']
            kernel_size = configs_for_all_convo_layers[layer_index]['kernel_size']
            padding = tuple((k-1) // 2 for k in kernel_size)
            stride       = configs_for_all_convo_layers[layer_index]['convo_stride']
            maxpool_size = configs_for_all_convo_layers[layer_index]['maxpool_size']
            if self.debug_config:
                print("\n     in_channels=%d   out_channels=%d    kernel_size=%s     stride=%s    \
                maxpool_size=%s" % (in_channels, out_channels, str(kernel_size), str(stride), 
                str(maxpool_size)))
            conv_layers.append( nn.Conv2d( in_channels,out_channels,kernel_size,stride=stride,padding=padding) )
            conv_layers.append( nn.MaxPool2d( maxpool_size ) )
            conv_layers.append( nn.ReLU() ),
            in_channels_for_next_layer = out_channels
        return conv_layers

    def build_fc_layers(self):
        fc_layers = nn.ModuleList()
        for layer_index in range(len(self.fc_layers_config) - 1):
            fc_layers.append( nn.Linear( self.fc_layers_config[layer_index], 
                                                                self.fc_layers_config[layer_index+1] ) )
        return fc_layers            

    def load_cifar_10_dataset(self):       
        '''
        In the code shown below, the call to "ToTensor()" converts the usual int range 0-255 for pixel 
        values to 0-1.0 float vals and then the call to "Normalize()" changes the range to -1.0-1.0 float 
        vals. For additional explanation of the call to "tvt.ToTensor()", see Slide 31 of my Week 2 
        slides at the DL course website.  And see Slides 32 and 33 for the syntax 
        "tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))".  In this call, the three numbers in the
        first tuple change the means in the three color channels and the three numbers in the second 
        tuple change the standard deviations according to the formula:

                 image_channel_val = (image_channel_val - mean) / std

        The end result is that the values in the image tensor will be normalized to fall between -1.0 
        and +1.0. If needed we can do inverse normalization  by

                 image_channel_val  =   (image_channel_val * std) + mean
        '''

        ##   But then the call to Normalize() changes the range to -1.0-1.0 float vals.
        transform = tvt.Compose([tvt.ToTensor(),
                                 tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    ## accuracy: 51%
        ##  Define where the training and the test datasets are located:
        train_data_loc = torchvision.datasets.CIFAR10(root=self.dataroot, train=True, download=True, transform=transform)
        test_data_loc = torchvision.datasets.CIFAR10(root=self.dataroot, train=False, download=True, transform=transform)
        ##  Now create the data loaders:
        self.train_data_loader = torch.utils.data.DataLoader(train_data_loc,batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.test_data_loader = torch.utils.data.DataLoader(test_data_loc,batch_size=self.batch_size, shuffle=False, num_workers=2)

    def load_cifar_10_dataset_with_augmentation(self):             
        '''
        In general, we want to do data augmentation for training:
        '''
        transform_train = tvt.Compose([
                                  tvt.RandomCrop(32, padding=4),
                                  tvt.RandomHorizontalFlip(),
                                  tvt.ToTensor(),
                                  tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])        
        ##  Don't need any augmentation for the test data: 
        transform_test = tvt.Compose([
                               tvt.ToTensor(),
                               tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ##  Define where the training and the test datasets are located
        train_data_loc = torchvision.datasets.CIFAR10(
                        root=self.dataroot, train=True, download=True, transform=transform_train)
        test_data_loc = torchvision.datasets.CIFAR10(
                      root=self.dataroot, train=False, download=True, transform=transform_test)
        ##  Now create the data loaders:
        self.train_data_loader = torch.utils.data.DataLoader(train_data_loc, batch_size=self.batch_size, 
                                                                     shuffle=True, num_workers=2)
        self.test_data_loader = torch.utils.data.DataLoader(test_data_loc, batch_size=self.batch_size, 
                                                                 shuffle=False, num_workers=2)

    def imshow(self, img):
        '''
        called by display_tensor_as_image() for displaying the image
        '''
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    class Net(nn.Module):
        def __init__(self, convo_layers, fc_layers):
            super(DLStudio.Net, self).__init__()
            self.my_modules_convo = convo_layers
            self.my_modules_fc = fc_layers
        def forward(self, x):
            for m in self.my_modules_convo:
                x = m(x)
            x = x.view(x.size(0), -1)
            for m in self.my_modules_fc:
                x = m(x)
            return x


    def run_code_for_training(self, net, display_images=False):        
        filename_for_out = "performance_numbers_" + str(self.epochs) + ".txt"
        FILE = open(filename_for_out, 'w')
        net = copy.deepcopy(net)
        net = net.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.learning_rate, momentum=self.momentum)
        print("\n\nStarting training loop...")
        start_time = time.perf_counter()
        loss_tally = []
        elapsed_time = 0.0
        for epoch in range(self.epochs):  
            print("")
            running_loss = 0.0
            for i, data in enumerate(self.train_data_loader):
                inputs, labels = data
                if i % 1000 == 999:
                    current_time = time.perf_counter()
                    elapsed_time = current_time - start_time 
                    print("\n\n[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]   Ground Truth:     " % 
                          (epoch+1, self.epochs, i+1, elapsed_time) + 
                          ' '.join('%10s' % self.class_labels[labels[j]] for j in range(self.batch_size)))
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                ##  Since PyTorch likes to construct dynamic computational graphs, we need to
                ##  zero out the previously calculated gradients for the learnable parameters:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                if i % 1000 == 999:
                    _, predicted = torch.max(outputs.data, 1)
                    print("[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]   Predicted Labels: " % 
                     (epoch+1, self.epochs, i+1, elapsed_time ) +
                     ' '.join('%10s' % self.class_labels[predicted[j]] for j in range(self.batch_size)))
                    avg_loss = running_loss / float(2000)
                    loss_tally.append(avg_loss)
                    print("[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]   Loss: %.3f" % 
                                                                   (epoch+1, self.epochs, i+1, elapsed_time, avg_loss))    
                    FILE.write("%.3f\n" % avg_loss)
                    FILE.flush()
                    running_loss = 0.0
                    if display_images:
                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[6,3])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(inputs, 
                                                            normalize=False, padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()
                        logger.setLevel(old_level)
                loss.backward()
                optimizer.step()
        print("\nFinished Training\n")
        self.save_model(net)
        plt.figure(figsize=(10,5))
        plt.title("Labeling Loss vs. Iterations")
        plt.plot(loss_tally)
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("playing_with_skips_loss.png")
        plt.show()


    def display_tensor_as_image(self, tensor, title=""):
        '''
        This method converts the argument tensor into a photo image that you can display
        in your terminal screen. It can convert tensors of three different shapes
        into images: (3,H,W), (1,H,W), and (H,W), where H, for height, stands for the
        number of pixels in the vertical direction and W, for width, for the same
        along the horizontal direction.  When the first element of the shape is 3,
        that means that the tensor represents a color image in which each pixel in
        the (H,W) plane has three values for the three color channels.  On the other
        hand, when the first element is 1, that stands for a tensor that will be
        shown as a grayscale image.  And when the shape is just (H,W), that is
        automatically taken to be for a grayscale image.
        '''
        tensor_range = (torch.min(tensor).item(), torch.max(tensor).item())
        if tensor_range == (-1.0,1.0):
            ##  The tensors must be between 0.0 and 1.0 for the display:
            print("\n\n\nimage un-normalization called")
            tensor = tensor/2.0 + 0.5     # unnormalize
        plt.figure(title)
        ###  The call to plt.imshow() shown below needs a numpy array. We must also
        ###  transpose the array so that the number of channels (the same thing as the
        ###  number of color planes) is in the last element.  For a tensor, it would be in
        ###  the first element.
        if tensor.shape[0] == 3 and len(tensor.shape) == 3:
#            plt.imshow( tensor.numpy().transpose(1,2,0) )
            plt.imshow( tensor.numpy().transpose(1,2,0) )
        ###  If the grayscale image was produced by calling torchvision.transform's
        ###  ".ToPILImage()", and the result converted to a tensor, the tensor shape will
        ###  again have three elements in it, however the first element that stands for
        ###  the number of channels will now be 1
        elif tensor.shape[0] == 1 and len(tensor.shape) == 3:
            tensor = tensor[0,:,:]
            plt.imshow( tensor.numpy(), cmap = 'gray' )
        ###  For any one color channel extracted from the tensor representation of a color
        ###  image, the shape of the tensor will be (W,H):
        elif len(tensor.shape) == 2:
            plt.imshow( tensor.numpy(), cmap = 'gray' )
        else:
            sys.exit("\n\n\nfrom 'display_tensor_as_image()': tensor for image is ill formed -- aborting")
        plt.show()

    def check_a_sampling_of_images(self):
        '''
        Displays the first batch_size number of images in your dataset.
        '''
        dataiter = iter(self.train_data_loader)
        images, labels = dataiter.next()
        # Since negative pixel values make no sense for display, setting the 'normalize' 
        # option to True will change the range back from (-1.0,1.0) to (0.0,1.0):
        self.display_tensor_as_image(torchvision.utils.make_grid(images, normalize=True))
        # Print class labels for the images shown:
        print(' '.join('%5s' % self.class_labels[labels[j]] for j in range(self.batch_size)))

    def save_model(self, model):
        '''
        Save the trained model to a disk file
        '''
        torch.save(model.state_dict(), self.path_saved_model)


    def run_code_for_testing(self, net, display_images=False):
        net.load_state_dict(torch.load(self.path_saved_model))
        net = net.eval()
        net = net.to(self.device)
        ##  In what follows, in addition to determining the predicted label for each test
        ##  image, we will also compute some stats to measure the overall performance of
        ##  the trained network.  This we will do in two different ways: For each class,
        ##  we will measure how frequently the network predicts the correct labels.  In
        ##  addition, we will compute the confusion matrix for the predictions.
        filename_for_results = "classification_results_" + str(self.epochs) + ".txt"
        FILE = open(filename_for_results, 'w')
        correct = 0
        total = 0
        confusion_matrix = torch.zeros(len(self.class_labels), len(self.class_labels))
        class_correct = [0] * len(self.class_labels)
        class_total = [0] * len(self.class_labels)
        with torch.no_grad():
            for i,data in enumerate(self.test_data_loader):
                ##  data is set to the images and the labels for one batch at a time:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                if i % 1000 == 999:
                    print("\n\n[i=%d:] Ground Truth:     " % (i+1) + ' '.join('%5s' % self.class_labels[labels[j]] 
                                                                   for j in range(self.batch_size)))
                outputs = net(images)
                ##  max() returns two things: the max value and its index in the 10 element
                ##  output vector.  We are only interested in the index --- since that is 
                ##  essentially the predicted class label:
                _, predicted = torch.max(outputs.data, 1)#
#                if display_images and i % 1000 == 999:
                if i % 1000 == 999:
                    print("[i=%d:] Predicted Labels: " % (i+1) + ' '.join('%5s' % self.class_labels[predicted[j]]
                                                              for j in range(self.batch_size)))
                    logger = logging.getLogger()
                    old_level = logger.level
                    if display_images:
                        logger.setLevel(100)
                        plt.figure(figsize=[6,3])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(images,
                                                      normalize=False, padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()
                        logger.setLevel(old_level)
                for label,prediction in zip(labels,predicted):
                        confusion_matrix[label][prediction] += 1
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                ##  comp is a list of size batch_size of "True" and "False" vals
                comp = predicted == labels       
                for j in range(self.batch_size):
                    label = labels[j]
                    ##  The following works because, in a numeric context, the boolean value
                    ##  "False" is the same as number 0 and the boolean value True is the 
                    ##  same as number 1. For that reason "4 + True" will evaluate to 5 and
                    ##  "4 + False" will evaluate to 4.  Also, "1 == True" evaluates to "True"
                    ##  "1 == False" evaluates to "False".  However, note that "1 is True" 
                    ##  evaluates to "False" because the operator "is" does not provide a 
                    ##  numeric context for "True". And so on.  In the statement that follows,
                    ##  while  c[j].item() will either return "False" or "True", for the 
                    ##  addition operator, Python will use the values 0 and 1 instead.
                    class_correct[label] += comp[j].item()
                    class_total[label] += 1
        for j in range(len(self.class_labels)):
            print('Prediction accuracy for %5s : %2d %%' % (self.class_labels[j], 100 * class_correct[j] / class_total[j]))
            FILE.write('\n\nPrediction accuracy for %5s : %2d %%\n' % (self.class_labels[j], 100 * class_correct[j] / class_total[j]))
        print("\n\n\nOverall accuracy of the network on the 10000 test images: %d %%" % (100 * correct / float(total)))
        FILE.write("\n\n\nOverall accuracy of the network on the 10000 test images: %d %%\n" % (100 * correct / float(total)))
        print("\n\nDisplaying the confusion matrix:\n")
        FILE.write("\n\nDisplaying the confusion matrix:\n\n")
        out_str = "         "
        for j in range(len(self.class_labels)):  out_str +=  "%7s" % self.class_labels[j]   
        print(out_str + "\n")
        FILE.write(out_str + "\n\n")
        for i,label in enumerate(self.class_labels):
            out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                                                      for j in range(len(self.class_labels))]
            out_percents = ["%.2f" % item.item() for item in out_percents]
            out_str = "%6s:  " % self.class_labels[i]
            for j in range(len(self.class_labels)): out_str +=  "%7s" % out_percents[j]
            print(out_str)
            FILE.write(out_str + "\n")
        FILE.close()        


    ###%%%
    ########################################################################################
    ###############  Start Definition of Inner Class ExperimentsWithSequential #############

    class ExperimentsWithSequential(nn.Module):                                
        """
        Demonstrates how to use the torch.nn.Sequential container class

        Class Path:  DLStudio  ->  ExperimentsWithSequential    
        """
        def __init__(self, dl_studio ):
            super(DLStudio.ExperimentsWithSequential, self).__init__()
            self.dl_studio = dl_studio

        def load_cifar_10_dataset(self):       
            self.dl_studio.load_cifar_10_dataset()

        def load_cifar_10_dataset_with_augmentation(self):             
            self.dl_studio.load_cifar_10_dataset_with_augmentation()

        class Net(nn.Module):
            """
            To see if the DLStudio class would work with any network that a user may want
            to experiment with, I copy-and-pasted the network shown below from the following
            page by Zhenye at GitHub:
                         https://zhenye-na.github.io/2018/09/28/pytorch-cnn-cifar10.html

            Class Path:  DLStudio  ->  ExperimentsWithSequential  ->  Net
            """
            def __init__(self):
                super(DLStudio.ExperimentsWithSequential.Net, self).__init__()
                self.conv_seqn = nn.Sequential(
                    # Conv Layer block 1:
                    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    # Conv Layer block 2:
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout2d(p=0.05),
                    # Conv Layer block 3:
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                self.fc_seqn = nn.Sequential(
                    nn.Dropout(p=0.1),
                    nn.Linear(4096, 1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.1),
                    nn.Linear(512, 10)
                )
    
            def forward(self, x):
                x = self.conv_seqn(x)
                # flatten
                x = x.view(x.size(0), -1)
                x = self.fc_seqn(x)
                return x

        def run_code_for_training(self, net):        
            self.dl_studio.run_code_for_training(net)

        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)

        def run_code_for_testing(self, model):
            self.dl_studio.run_code_for_testing(model)

    ###%%%
    ########################################################################################
    ##################  Start Definition of Inner Class ExperimentsWithCIFAR ###############

    class ExperimentsWithCIFAR(nn.Module):              
        """
        Class Path:  DLStudio  ->  ExperimentsWithCIFAR
        """

        def __init__(self, dl_studio ):
            super(DLStudio.ExperimentsWithCIFAR, self).__init__()
            self.dl_studio = dl_studio

        def load_cifar_10_dataset(self):       
            self.dl_studio.load_cifar_10_dataset()

        def load_cifar_10_dataset_with_augmentation(self):             
            self.dl_studio.load_cifar_10_dataset_with_augmentation()

        ##  You can instantiate two different types of networks when experimenting with 
        ##  the inner class ExperimentsWithCIFAR.  The network shown below is from the 
        ##  PyTorch tutorial
        ##
        ##     https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        ##
        class Net(nn.Module):
            """
            Class Path:  DLStudio  ->  ExperimentsWithCIFAR  ->  Net
            """
            def __init__(self):
                super(DLStudio.ExperimentsWithCIFAR.Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)
        
            def forward(self, x):
                x = nn.MaxPool2d(2,2)(F.relu(self.conv1(x)))
                x = nn.MaxPool2d(2,2)(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        ##  Instead of using the network shown above, you can also use the network shown below.
        ##  if you are playing with the ExperimentsWithCIFAR inner class. If that's what you
        ##  want to do, in the script "playing_with_cifar10.py" in the Examples directory,
        ##  you will need to replace the statement
        ##                          model = exp_cifar.Net()
        ##  by the statement
        ##                          model = exp_cifar.Net2()        
        ##
        class Net2(nn.Module):
            """
            Class Path:  DLStudio  ->  ExperimentsWithCIFAR  ->  Net2
            """
            def __init__(self):
                """
                I created this network class just to see if it was possible to simply calculate
                the size of the first of the fully connected layers from strides in the convo
                layers up to that point and from the out_channels used in the top-most convo 
                layer.   In what you see below, I am keeping track of all the strides by pushing 
                them into the array 'strides'.  Subsequently, in the formula shown in line (A),
                I use the product of all strides and the number of out_channels for the topmost
                layer to compute the size of the first fully-connected layer.
                """
                super(DLStudio.ExperimentsWithCIFAR.Net2, self).__init__()
                self.relu = nn.ReLU()
                strides = []
                patch_size = 2
                ## conv1:
                out_ch, ker_size, conv_stride, pool_stride = 128,5,1,2
                self.conv1 = nn.Conv2d(3, out_ch, (ker_size,ker_size), padding=(ker_size-1)//2)     
                self.pool1 = nn.MaxPool2d(patch_size, pool_stride)     
                strides += (conv_stride, pool_stride)
                ## conv2:
                in_ch = out_ch
                out_ch, ker_size, conv_stride, pool_stride = 128,3,1,2
                self.conv2 = nn.Conv2d(in_ch, out_ch, ker_size, padding=(ker_size-1)//2)
                self.pool2 = nn.MaxPool2d(patch_size, pool_stride)     
                strides += (conv_stride, pool_stride)
                ## conv3:                   
                ## meant for repeated invocation, must have same in_ch, out_ch and strides of 1
                in_ch = out_ch
                out_ch, ker_size, conv_stride, pool_stride = in_ch,2,1,1
                self.conv3 = nn.Conv2d(in_ch, out_ch, ker_size, padding=1)
                self.pool3 = nn.MaxPool2d(patch_size, pool_stride)         
#                strides += (conv_stride, pool_stride)
                ## figure out the number of nodes needed for entry into fc:
                in_size_for_fc = out_ch * (32 // np.prod(strides)) ** 2                    ## (A)
                self.in_size_for_fc = in_size_for_fc
                self.fc1 = nn.Linear(in_size_for_fc, 150)
                self.fc2 = nn.Linear(150, 100)
                self.fc3 = nn.Linear(100, 10)
        
            def forward(self, x):
                ##  We know that forward() begins its with work x shaped as (4,3,32,32) where
                ##  4 is the batch size, 3 in_channels, and where the input image size is 32x32.
                x = self.relu(self.conv1(x))  
                x = self.pool1(x)             
                x = self.relu(self.conv2(x))
                x = self.pool2(x)             
                x = self.pool3(self.relu(self.conv3(x)))
                x = x.view(-1, self.in_size_for_fc)
                x = self.relu(self.fc1( x ))
                x = self.relu(self.fc2( x ))
                x = self.fc3(x)
                return x

        def run_code_for_training(self, net, display_images=False):
            self.dl_studio.run_code_for_training(net, display_images)
            
        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)

        def run_code_for_testing(self, model, display_images=False):
            self.dl_studio.run_code_for_testing(model, display_images)


    ###%%%
    ########################################################################################
    ###################  Start Definition of Inner Class SkipConnections  ##################

    class SkipConnections(nn.Module):             
        """
        This educational class is meant for illustrating the concepts related to the 
        use of skip connections in neural network.  It is now well known that deep
        networks are difficult to train because of the vanishing gradients problem.
        What that means is that as the depth of network increases, the loss gradients
        calculated for the early layers become more and more muted, which suppresses
        the learning of the parameters in those layers.  An important mitigation
        strategy for addressing this problem consists of creating a CNN using blocks
        with skip connections.

        With the code shown in this inner class of the module, you can now experiment
        with skip connections in a CNN to see how a deep network with this feature
        might improve the classification results.  As you will see in the code shown
        below, the network that allows you to construct a CNN with skip connections
        is named BMEnet.  As shown in the script playing_with_skip_connections.py in
        the Examples directory of the distribution, you can easily create a CNN with
        arbitrary depth just by using the "depth" constructor option for the BMEnet
        class.  The basic block of the network constructed by BMEnet is called
        SkipBlock which, very much like the BasicBlock in ResNet-18, has a couple of
        convolutional layers whose output is combined with the input to the block.
    
        Note that the value given to the "depth" constructor option for the
        BMEnet class does NOT translate directly into the actual depth of the
        CNN. [Again, see the script playing_with_skip_connections.py in the Examples
        directory for how to use this option.] The value of "depth" is translated
        into how many instances of SkipBlock to use for constructing the CNN.

        Class Path:  DLStudio  ->  SkipConnections
        """
        def __init__(self, dl_studio):
            super(DLStudio.SkipConnections, self).__init__()
            self.dl_studio = dl_studio

        def load_cifar_10_dataset(self):       
            self.dl_studio.load_cifar_10_dataset()

        def load_cifar_10_dataset_with_augmentation(self):             
            self.dl_studio.load_cifar_10_dataset_with_augmentation()


        class SkipBlock(nn.Module):
            """
            Class Path:   DLStudio  ->  SkipConnections  ->  SkipBlock
            """            
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(DLStudio.SkipConnections.SkipBlock, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.bn2 = nn.BatchNorm2d(out_ch)
                if downsample:
                    ##  Setting stride to 2 and kernel_size to 1 amounts to retaining every
                    ##  other pixel in the image --- which halves the size of the image:
                    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

            def forward(self, x):
                identity = x                                     
                out = self.convo1(x)                              
                out = self.bn1(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo2(out)                              
                    out = self.bn2(out)                              
                    out = torch.nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        ## To understand the following assignments, recall that the data has the
                        ## shape [B,C,H,W]. So it is the second axis that corresponds to the channels
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
                return out


        class BMEnet(nn.Module):
            """
            Class Path:   DLStudio  ->  SkipConnections  ->  BMEnet
            """
            def __init__(self, skip_connections=True, depth=32):
                super(DLStudio.SkipConnections.BMEnet, self).__init__()
                if depth not in [8, 16, 32, 64]:
                    sys.exit("BMEnet has been tested for depth for only 8, 16, 32, and 64")
                self.depth = depth // 8
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.skip64_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64_arr.append(DLStudio.SkipConnections.SkipBlock(64, 64,
                                                          skip_connections=skip_connections))
                self.skip64ds = DLStudio.SkipConnections.SkipBlock(64, 64, 
                                            downsample=True, skip_connections=skip_connections)
                self.skip64to128 = DLStudio.SkipConnections.SkipBlock(64, 128, 
                                                            skip_connections=skip_connections )
                self.skip128_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128_arr.append(DLStudio.SkipConnections.SkipBlock(128, 128,
                                                         skip_connections=skip_connections))
                self.skip128ds = DLStudio.SkipConnections.SkipBlock(128,128,
                                            downsample=True, skip_connections=skip_connections)
                ##  The following declaration is predicated on the assumption that the number of
                ##  output nodes (CxHxW) from the final convo layer exactly 2048 for each
                ##  input image.  Depending on the size of the input image, this places a constraint
                ##  on how many downsampling instances of SkipBlock you can call in a network.
                self.fc1 =  nn.Linear(2048, 1000)
                self.fc2 =  nn.Linear(1000, 10)

            def forward(self, x):
                x = self.pool(torch.nn.functional.relu(self.conv(x)))          
                for i,skip64 in enumerate(self.skip64_arr[:self.depth//4]):
                    x = skip64(x)                
                x = self.skip64ds(x)
                for i,skip64 in enumerate(self.skip64_arr[self.depth//4:]):
                    x = skip64(x)                
                x = self.skip64ds(x)
                x = self.skip64to128(x)
                for i,skip128 in enumerate(self.skip128_arr[:self.depth//4]):
                    x = skip128(x)                
                #  x = self.skip128ds(x)                                                       ## (A)
                for i,skip128 in enumerate(self.skip128_arr[self.depth//4:]):
                    x = skip128(x)                
                ## See the comment block above the "self.fc1" declaration in the constructor code.
                x = x.view(-1, 2048 )
                x = torch.nn.functional.relu(self.fc1(x))
                x = self.fc2(x)
                return x            

        def run_code_for_training(self, net, display_images=False):        
            self.dl_studio.run_code_for_training(net, display_images)
            
        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)

        def run_code_for_testing(self, model, display_images=False):
            self.dl_studio.run_code_for_testing(model, display_images=False)


    ###%%%
    ########################################################################################
    #################  Start Definition of Inner Class CustomDataLoading  ##################

    class CustomDataLoading(nn.Module):             
        """This is a testbed for experimenting with a completely grounds-up attempt at
        designing a custom data loader.  Ordinarily, if the basic format of how the
        dataset is stored is similar to one of the datasets that the Torchvision
        module knows about, you can go ahead and use that for your own dataset.  At
        worst, you may need to carry out some light customizations depending on the
        number of classes involved, etc.

        However, if the underlying dataset is stored in a manner that does not look
        like anything in Torchvision, you have no choice but to supply yourself all
        of the data loading infrastructure.  That is what this inner class of the 
        DLStudio module is all about.

        The custom data loading exercise here is related to a dataset called
        PurdueShapes5 that contains 32x32 images of binary shapes belonging to the
        following five classes:

                       1.  rectangle
                       2.  triangle
                       3.  disk
                       4.  oval
                       5.  star

        The dataset was generated by randomizing the sizes and the orientations
        of these five patterns.  Since the patterns are rotated with a very simple
        non-interpolating transform, just the act of random rotations can introduce
        boundary and even interior noise in the patterns.

        Each 32x32 image is stored in the dataset as the following list:

                           [R, G, B, Bbox, Label]
        where
                R     :   is a 1024 element list of the values for the red component
                          of the color at all the pixels
           
                B     :   the same as above but for the green component of the color

                G     :   the same as above but for the blue component of the color

                Bbox  :   a list like [x1,y1,x2,y2] that defines the bounding box 
                          for the object in the image
           
                Label :   the shape of the object

        I serialize the dataset with Python's pickle module and then compress it with 
        the gzip module.  

        You will find the following dataset directories in the "data" subdirectory
        of Examples in the DLStudio distro:

               PurdueShapes5-10000-train.gz
               PurdueShapes5-1000-test.gz
               PurdueShapes5-20-train.gz
               PurdueShapes5-20-test.gz               

        The number that follows the main name string "PurdueShapes5-" is for the 
        number of images in the dataset.  

        You will find the last two datasets, with 20 images each, useful for debugging
        your logic for object detection and bounding-box regression.

        Class Path:   DLStudio  ->  CustomDataLoading
        """     
        def __init__(self, dl_studio, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
            super(DLStudio.CustomDataLoading, self).__init__()
            self.dl_studio = dl_studio
            self.dataserver_train = dataserver_train
            self.dataserver_test = dataserver_test

        class PurdueShapes5Dataset(torch.utils.data.Dataset):
            """
            Class Path:   DLStudio  ->  CustomDataLoading  ->  PurdueShapes5Dataset
            """
            def __init__(self, dl_studio, train_or_test, dataset_file):
                super(DLStudio.CustomDataLoading.PurdueShapes5Dataset, self).__init__()
                if train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train.gz":
                    if os.path.exists("torch_saved_PurdueShapes5-10000_dataset.pt") and \
                              os.path.exists("torch_saved_PurdueShapes5_label_map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset = torch.load("torch_saved_PurdueShapes5-10000_dataset.pt")
                        self.label_map = torch.load("torch_saved_PurdueShapes5_label_map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        torch.save(self.dataset, "torch_saved_PurdueShapes5-10000_dataset.pt")
                        torch.save(self.label_map, "torch_saved_PurdueShapes5_label_map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                else:
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
             
            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                r = np.array( self.dataset[idx][0] )
                g = np.array( self.dataset[idx][1] )
                b = np.array( self.dataset[idx][2] )
                R,G,B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
                im_tensor = torch.zeros(3,32,32, dtype=torch.float)
                im_tensor[0,:,:] = torch.from_numpy(R)
                im_tensor[1,:,:] = torch.from_numpy(G)
                im_tensor[2,:,:] = torch.from_numpy(B)
                sample = {'image' : im_tensor, 
                          'bbox' : self.dataset[idx][3],                          
                          'label' : self.dataset[idx][4] }
                return sample

        def load_PurdueShapes5_dataset(self, dataserver_train, dataserver_test ):       
            transform = tvt.Compose([tvt.ToTensor(),
                                tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
            self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                               batch_size=self.dl_studio.batch_size,shuffle=True, num_workers=4)
            self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                               batch_size=self.dl_studio.batch_size,shuffle=False, num_workers=4)

        class SkipBlock(nn.Module):
            """
            Class Path:   DLStudio  ->  CustomDataLoading  ->  SkipBlock
            """
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(DLStudio.CustomDataLoading.SkipBlock, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                norm_layer1 = nn.BatchNorm2d
                norm_layer2 = nn.BatchNorm2d
                self.bn1 = norm_layer1(out_ch)
                self.bn2 = norm_layer2(out_ch)
                if downsample:
                    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

            def forward(self, x):
                identity = x                                     
                out = self.convo1(x)                              
                out = self.bn1(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo2(out)                              
                    out = self.bn2(out)                              
                    out = torch.nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
                return out

        class BMEnet(nn.Module):
            """
            Class Path:   DLStudio  ->  CustomDataLoading  ->  BMENet
            """
            def __init__(self, skip_connections=True, depth=32):
                super(DLStudio.CustomDataLoading.BMEnet, self).__init__()
                if depth not in [6, 16, 32, 64]:
                    sys.exit("BMEnet has been tested for depth for only 16, 32, and 64")
                self.depth = depth // 8
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
#                self.pool = nn.MaxPool2d(2, 2)
                self.skip64_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64_arr.append(DLStudio.SkipConnections.SkipBlock(64, 64,
                                                          skip_connections=skip_connections))
                self.skip64ds = DLStudio.SkipConnections.SkipBlock(64, 64, 
                                            downsample=True, skip_connections=skip_connections)
                self.skip64to128 = DLStudio.SkipConnections.SkipBlock(64, 128, 
                                                            skip_connections=skip_connections )
                self.skip128_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128_arr.append(DLStudio.SkipConnections.SkipBlock(128, 128,
                                                         skip_connections=skip_connections))
                self.skip128ds = DLStudio.SkipConnections.SkipBlock(128,128,
                                            downsample=True, skip_connections=skip_connections)
                self.fc1 =  nn.Linear(2048, 1000)
                self.fc2 =  nn.Linear(1000, 10)

            def forward(self, x):
#                x = self.pool(torch.nn.functional.relu(self.conv(x)))          
                x = nn.MaxPool2d(2,2)(torch.nn.functional.relu(self.conv(x)))          
                for i,skip64 in enumerate(self.skip64_arr[:self.depth//4]):
                    x = skip64(x)                
                x = self.skip64ds(x)
                for i,skip64 in enumerate(self.skip64_arr[self.depth//4:]):
                    x = skip64(x)                
                x = self.skip64ds(x)
                x = self.skip64to128(x)
                for i,skip128 in enumerate(self.skip128_arr[:self.depth//4]):
                    x = skip128(x)                
                for i,skip128 in enumerate(self.skip128_arr[self.depth//4:]):
                    x = skip128(x)                
                x = x.view(-1, 2048 )
                x = torch.nn.functional.relu(self.fc1(x))
                x = self.fc2(x)
                return x            

        def run_code_for_training_with_custom_loading(self, net):        
            filename_for_out = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE = open(filename_for_out, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            for epoch in range(self.dl_studio.epochs):  
                running_loss = 0.0
                for i, data in enumerate(self.train_dataloader):
                    inputs, bounding_box, labels = data['image'], data['bbox'], data['label']
                    if self.dl_studio.debug_train and i % 1000 == 999:
                        print("\n\n\nlabels: %s" % str(labels))
                        print("\n\n\ntype of labels: %s" % type(labels))
                        print("\n\n[iter=%d:] Ground Truth:     " % (i+1) + 
                        ' '.join('%5s' % self.dataserver_train.class_labels[labels[j].item()] for j in range(self.dl_studio.batch_size)))
                    inputs = inputs.to(self.dl_studio.device)
                    labels = labels.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    if self.dl_studio.debug_train and i % 1000 == 999:
                        _, predicted = torch.max(outputs.data, 1)
                        print("[iter=%d:] Predicted Labels: " % (i+1) + 
                         ' '.join('%5s' % self.dataserver.class_labels[predicted[j]] 
                                           for j in range(self.dl_studio.batch_size)))
                        self.dl_studio.display_tensor_as_image(torchvision.utils.make_grid(
             inputs, normalize=True), "see terminal for TRAINING results at iter=%d" % (i+1))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    if i % 1000 == 999:    
                        avg_loss = running_loss / float(1000)
                        print("[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, avg_loss))
                        FILE.write("%.3f\n" % avg_loss)
                        FILE.flush()
                        running_loss = 0.0
            print("\nFinished Training\n")
            self.save_model(net)
            
        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)

        def run_code_for_testing_with_custom_loading(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            correct = 0
            total = 0
            confusion_matrix = torch.zeros(len(self.dataserver_train.class_labels), 
                                           len(self.dataserver_train.class_labels))
            class_correct = [0] * len(self.dataserver_train.class_labels)
            class_total = [0] * len(self.dataserver_train.class_labels)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    images, bounding_box, labels = data['image'], data['bbox'], data['label']
                    labels = labels.tolist()
                    if self.dl_studio.debug_test and i % 1000 == 0:
                        print("\n\n[i=%d:] Ground Truth:     " %i + ' '.join('%10s' % 
                          self.dataserver_train.class_labels[labels[j]] for j in range(self.dl_studio.batch_size)))
                    outputs = net(images)
                    ##  max() returns two things: the max value and its index in the 10 element
                    ##  output vector.  We are only interested in the index --- since that is 
                    ##  essentially the predicted class label:
                    _, predicted = torch.max(outputs.data, 1)
                    predicted = predicted.tolist()
                    if self.dl_studio.debug_test and i % 1000 == 0:
                        print("[i=%d:] Predicted Labels: " %i + ' '.join('%10s' % 
                          self.dataserver_train.class_labels[predicted[j]] for j in range(self.dl_studio.batch_size)))
                        self.dl_studio.display_tensor_as_image(
                              torchvision.utils.make_grid(images, normalize=True), 
                              "see terminal for test results at i=%d" % i)
                    for label,prediction in zip(labels,predicted):
                        confusion_matrix[label][prediction] += 1
                    total += len(labels)
                    correct +=  [predicted[ele] == labels[ele] for ele in range(len(predicted))].count(True)
                    comp = [predicted[ele] == labels[ele] for ele in range(len(predicted))]
                    for j in range(self.dl_studio.batch_size):
                        label = labels[j]
                        class_correct[label] += comp[j]
                        class_total[label] += 1
            print("\n")
            for j in range(len(self.dataserver_train.class_labels)):
                print('Prediction accuracy for %5s : %2d %%' % (
              self.dataserver_train.class_labels[j], 100 * class_correct[j] / class_total[j]))
            print("\n\n\nOverall accuracy of the network on the 10000 test images: %d %%" % 
                                                                   (100 * correct / float(total)))
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                "
            for j in range(len(self.dataserver_train.class_labels)):  
                                 out_str +=  "%15s" % self.dataserver_train.class_labels[j]   
            print(out_str + "\n")
            for i,label in enumerate(self.dataserver_train.class_labels):
                out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                                 for j in range(len(self.dataserver_train.class_labels))]
                out_percents = ["%.2f" % item.item() for item in out_percents]
                out_str = "%12s:  " % self.dataserver_train.class_labels[i]
                for j in range(len(self.dataserver_train.class_labels)): 
                                                       out_str +=  "%15s" % out_percents[j]
                print(out_str)
    
    ###%%%
    ########################################################################################
    ###################  Start Definition of Inner Class DetectAndLocalize  ################

    class DetectAndLocalize(nn.Module):             
        """
        The purpose of this inner class is to focus on object detection in images --- as
        opposed to image classification.  Most people would say that object detection
        is a more challenging problem than image classification because, in general,
        the former also requires localization.  The simplest interpretation of what
        is meant by localization is that the code that carries out object detection
        must also output a bounding-box rectangle for the object that was detected.

        You will find in this inner class some examples of LOADnet classes meant
        for solving the object detection and localization problem.  The acronym
        "LOAD" in "LOADnet" stands for

                    "LOcalization And Detection"

        The different network examples included here are LOADnet1, LOADnet2, and
        LOADnet3.  For now, only pay attention to LOADnet2 since that's the class I
        have worked with the most for the 1.0.7 distribution.

        Class Path:   DLStudio  ->  DetectAndLocalize
        """
        def __init__(self, dl_studio, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
            super(DLStudio.DetectAndLocalize, self).__init__()
            self.dl_studio = dl_studio
            self.dataserver_train = dataserver_train
            self.dataserver_test = dataserver_test
            self.debug = False

        class PurdueShapes5Dataset(torch.utils.data.Dataset):
            """
            Class Path:   DLStudio  ->  DetectAndLocalize  ->  PurdueShapes5Dataset
            """
            def __init__(self, dl_studio, train_or_test, dataset_file):
                super(DLStudio.DetectAndLocalize.PurdueShapes5Dataset, self).__init__()
                if train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train.gz":
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset.pt")
                        self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset, self.label_map = pickle.loads(dataset)
                        torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset.pt")
                        torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-20.gz":
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                        self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset, self.label_map = pickle.loads(dataset)
                        torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                        torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-50.gz":
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                        self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset, self.label_map = pickle.loads(dataset)
                        torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                        torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-80.gz":
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                        self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset, self.label_map = pickle.loads(dataset)
                        torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                        torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                else:
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset, self.label_map = pickle.loads(dataset)
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
             
            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                r = np.array( self.dataset[idx][0] )
                g = np.array( self.dataset[idx][1] )
                b = np.array( self.dataset[idx][2] )
                R,G,B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
                im_tensor = torch.zeros(3,32,32, dtype=torch.float)
                im_tensor[0,:,:] = torch.from_numpy(R)
                im_tensor[1,:,:] = torch.from_numpy(G)
                im_tensor[2,:,:] = torch.from_numpy(B)
                bb_tensor = torch.tensor(self.dataset[idx][3], dtype=torch.float)
                sample = {'image' : im_tensor, 
                          'bbox' : bb_tensor,
                          'label' : self.dataset[idx][4] }
                return sample

        def load_PurdueShapes5_dataset(self, dataserver_train, dataserver_test ):       
            self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                               batch_size=self.dl_studio.batch_size,shuffle=True, num_workers=4)
            self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                               batch_size=self.dl_studio.batch_size,shuffle=False, num_workers=4)
    
        class SkipBlock(nn.Module):
            """
            Class Path:   DLStudio  ->  DetectAndLocalize  ->  SkipBlock
            """
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(DLStudio.DetectAndLocalize.SkipBlock, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                norm_layer1 = nn.BatchNorm2d
                norm_layer2 = nn.BatchNorm2d
                self.bn1 = norm_layer1(out_ch)
                self.bn2 = norm_layer2(out_ch)
                if downsample:
                    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
            def forward(self, x):
                identity = x                                     
                out = self.convo1(x)                              
                out = self.bn1(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo2(out)                              
                    out = self.bn2(out)                              
                    out = torch.nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
                return out


        class LOADnet1(nn.Module):
            """
            The acronym 'LOAD' stands for 'LOcalization And Detection'.
            LOADnet1 only uses fully-connected layers for the regression

            Class Path:   DLStudio  ->  DetectAndLocalize  ->  LOADnet1
            """
            def __init__(self, skip_connections=True, depth=32):
                super(DLStudio.DetectAndLocalize.LOADnet1, self).__init__()
                self.pool_count = 3
                self.depth = depth // 2
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
#                self.pool = nn.MaxPool2d(2, 2)
                self.skip64 = DLStudio.DetectAndLocalize.SkipBlock(64, 64, 
                                                           skip_connections=skip_connections)
                self.skip64ds = DLStudio.DetectAndLocalize.SkipBlock(64, 64, 
                                           downsample=True, skip_connections=skip_connections)
                self.skip64to128 = DLStudio.DetectAndLocalize.SkipBlock(64, 128, 
                                                            skip_connections=skip_connections )
                self.skip128 = DLStudio.DetectAndLocalize.SkipBlock(128, 128, 
                                                             skip_connections=skip_connections)
                self.skip128ds = DLStudio.DetectAndLocalize.SkipBlock(128,128,
                                            downsample=True, skip_connections=skip_connections)
                self.fc1 =  nn.Linear(128 * (32 // 2**self.pool_count)**2, 1000)
                self.fc2 =  nn.Linear(1000, 5)
                self.fc3 =  nn.Linear(32768, 1000)
                self.fc4 =  nn.Linear(1000, 4)

            def forward(self, x):
#                x = self.pool(torch.nn.functional.relu(self.conv(x)))          
                x = nn.MaxPool2d(2,2)(torch.nn.functional.relu(self.conv(x)))          
                ## The labeling section:
                for _ in range(self.depth // 4):
                    x1 = self.skip64(x)                                               
                x1 = self.skip64ds(x1)
                for _ in range(self.depth // 4):
                    x1 = self.skip64(x1)                                               
                x1 = self.skip64to128(x1)
                for _ in range(self.depth // 4):
                    x1 = self.skip128(x1)                                               
                x1 = self.skip128ds(x1)                                               
                for _ in range(self.depth // 4):
                    x1 = self.skip128(x1)                                               
                x1 = x1.view(-1, 128 * (32 // 2**self.pool_count)**2 )
                x1 = torch.nn.functional.relu(self.fc1(x1))
                x1 = self.fc2(x1)
                ## The Bounding Box regression:
                x2 = x.view(-1, 32768 )
                x2 = torch.nn.functional.relu(self.fc3(x2))
                x2 = self.fc4(x2)
                return x1,x2

        class LOADnet2(nn.Module):
            """
            The acronym 'LOAD' stands for 'LOcalization And Detection'.
            LOADnet2 uses both convo and linear layers for regression

            Class Path:   DLStudio  ->  DetectAndLocalize  ->  LOADnet2
            """ 
            def __init__(self, skip_connections=True, depth=8):
                super(DLStudio.DetectAndLocalize.LOADnet2, self).__init__()
                if depth not in [8,10,12,14,16]:
                    sys.exit("LOADnet2 has only been tested for 'depth' values 8, 10, 12, 14, and 16")
                self.depth = depth // 2
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
#                self.pool = nn.MaxPool2d(2, 2)
                self.bn1  = nn.BatchNorm2d(64)
                self.bn2  = nn.BatchNorm2d(128)
                self.skip64_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64_arr.append(DLStudio.DetectAndLocalize.SkipBlock(64, 64,
                                                          skip_connections=skip_connections))
                self.skip64ds = DLStudio.DetectAndLocalize.SkipBlock(64, 64, 
                                            downsample=True, skip_connections=skip_connections)
                self.skip64to128 = DLStudio.DetectAndLocalize.SkipBlock(64, 128, 
                                                            skip_connections=skip_connections )
                self.skip128_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128_arr.append(DLStudio.DetectAndLocalize.SkipBlock(128, 128,
                                                         skip_connections=skip_connections))
                self.skip128ds = DLStudio.DetectAndLocalize.SkipBlock(128,128,
                                            downsample=True, skip_connections=skip_connections)
                self.fc1 =  nn.Linear(2048, 1000)
                self.fc2 =  nn.Linear(1000, 5)

                ##  for regression
                self.conv_seqn = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
                self.fc_seqn = nn.Sequential(
                    nn.Linear(16384, 1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 4)        ## output for the 4 coords (x_min,y_min,x_max,y_max) of BBox
                )

            def forward(self, x):
                x = nn.MaxPool2d(2,2)(torch.nn.functional.relu(self.conv(x)))          
                ## The labeling section:
                x1 = x.clone()
                for i,skip64 in enumerate(self.skip64_arr[:self.depth//4]):
                    x1 = skip64(x1)                
                x1 = self.skip64ds(x1)
                for i,skip64 in enumerate(self.skip64_arr[self.depth//4:]):
                    x1 = skip64(x1)                
                x1 = self.bn1(x1)
                x1 = self.skip64to128(x1)
                for i,skip128 in enumerate(self.skip128_arr[:self.depth//4]):
                    x1 = skip128(x1)                
                x1 = self.bn2(x1)
                x1 = self.skip128ds(x1)
                for i,skip128 in enumerate(self.skip128_arr[self.depth//4:]):
                    x1 = skip128(x1)                
                x1 = x1.view(-1, 2048 )
                x1 = torch.nn.functional.relu(self.fc1(x1))
                x1 = self.fc2(x1)
                ## The Bounding Box regression:
                x2 = self.conv_seqn(x)
                # flatten
                x2 = x2.view(x.size(0), -1)
                x2 = self.fc_seqn(x2)
                return x1,x2

        class LOADnet3(nn.Module):
            """
            The acronym 'LOAD' stands for 'LOcalization And Detection'.
            LOADnet3 uses both convo and linear layers for regression

            Class Path:   DLStudio  ->  DetectAndLocalize  ->  LOADnet3
            """ 
            def __init__(self, skip_connections=True, depth=8):
                super(DLStudio.DetectAndLocalize.LOADnet3, self).__init__()
                if depth not in [4, 8, 16]:
                    sys.exit("LOADnet2 has been tested for 'depth' for only 4, 8, and 16")
                self.depth = depth // 4
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.skip64_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64_arr.append(DLStudio.DetectAndLocalize.SkipBlock(64, 64,
                                                          skip_connections=skip_connections))
                self.skip64ds = DLStudio.DetectAndLocalize.SkipBlock(64, 64, 
                                            downsample=True, skip_connections=skip_connections)
                self.skip64to128 = DLStudio.DetectAndLocalize.SkipBlock(64, 128, 
                                                            skip_connections=skip_connections )
                self.skip128_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128_arr.append(DLStudio.DetectAndLocalize.SkipBlock(128, 128,
                                                         skip_connections=skip_connections))
                self.skip128ds = DLStudio.DetectAndLocalize.SkipBlock(128,128,
                                            downsample=True, skip_connections=skip_connections)
                self.fc1 =  nn.Linear(2048, 1000)
                self.fc2 =  nn.Linear(1000, 5)

                ##  for regression
                self.conv_seqn = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
                self.fc_seqn = nn.Sequential(
                    nn.Linear(16384, 1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 4)
                )
            def forward(self, x):
#                x = self.pool(torch.nn.functional.relu(self.conv(x)))          
                x = nn.MaxPool2d(2,2)(torch.nn.functional.relu(self.conv(x)))          
                ## The labeling section:
                x1 = x.clone()
                for i,skip64 in enumerate(self.skip64_arr[:self.depth//4]):
                    x1 = skip64(x1)                
                x1 = self.skip64ds(x1)
                for i,skip64 in enumerate(self.skip64_arr[self.depth//4:]):
                    x1 = skip64(x1)                
                x1 = self.skip64ds(x1)
                x1 = self.skip64to128(x1)
                for i,skip128 in enumerate(self.skip128_arr[:self.depth//4]):
                    x1 = skip128(x1)                
                for i,skip128 in enumerate(self.skip128_arr[self.depth//4:]):
                    x1 = skip128(x1)                
                x1 = x1.view(-1, 2048 )
                x1 = torch.nn.functional.relu(self.fc1(x1))
                x1 = self.fc2(x1)
                ## The Bounding Box regression:
                for _ in range(4):
                    x2 = self.skip64(x)                                               
                x2 = self.skip64to128(x2)
                for _ in range(4):
                    x2 = self.skip128(x2)                                               
                x2 = x.view(-1, 128 * (32 // 2**self.pool_count)**2 )
                x2 = torch.nn.functional.relu(self.fc3(x2))
                x2 = self.fc4(x2)
                return x1,x2

        class IOULoss(nn.Module):
            """
            Class Path:   DLStudio  ->  DetectAndLocalize  ->  IOULoss
            """
            def __init__(self, batch_size):
                super(DLStudio.DetectAndLocalize.IOULoss, self).__init__()
                self.batch_size = batch_size
            def forward(self, input, target):
                composite_loss = []
                for idx in range(self.batch_size):
                    union = intersection = 0.0
                    for i in range(32):
                        for j in range(32):
                            inp = input[idx,i,j]
                            tap = target[idx,i,j]
                            if (inp == tap) and (inp==1):
                                intersection += 1
                                union += 1
                            elif (inp != tap) and ((inp==1) or (tap==1)):
                                union += 1
                    if union == 0.0:
                        raise Exception("something_wrong")
                    batch_sample_iou = intersection / float(union)
                    composite_loss.append(batch_sample_iou)
                total_iou_for_batch = sum(composite_loss) 
                return 1 - torch.tensor([total_iou_for_batch / self.batch_size])


        def run_code_for_training_with_CrossEntropy_and_MSE_Losses(self, net):        
            filename_for_out1 = "performance_numbers_" + str(self.dl_studio.epochs) + "label.txt"
            filename_for_out2 = "performance_numbers_" + str(self.dl_studio.epochs) + "regres.txt"
            FILE1 = open(filename_for_out1, 'w')
            FILE2 = open(filename_for_out2, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            criterion1 = nn.CrossEntropyLoss()
            criterion2 = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            print("\n\nStarting training loop...\n\n")
            start_time = time.perf_counter()
            labeling_loss_tally = []   
            regression_loss_tally = [] 
            elapsed_time = 0.0   
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss_labeling = 0.0
                running_loss_regression = 0.0       
                for i, data in enumerate(self.train_dataloader):
                    gt_too_small = False
                    inputs, bbox_gt, labels = data['image'], data['bbox'], data['label']
                    if i % 500 == 499:
                        current_time = time.perf_counter()
                        elapsed_time = current_time - start_time
                        print("\n\n\n[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]      Ground Truth:     " % 
                                 (epoch+1, self.dl_studio.epochs, i+1, elapsed_time) 
                               + ' '.join('%10s' % self.dataserver_train.class_labels[labels[j].item()] 
                                                                for j in range(self.dl_studio.batch_size)))
                    inputs = inputs.to(self.dl_studio.device)
                    labels = labels.to(self.dl_studio.device)
                    bbox_gt = bbox_gt.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    if self.debug:
                        self.dl_studio.display_tensor_as_image(
                          torchvision.utils.make_grid(inputs.cpu(), nrow=4, normalize=True, padding=2, pad_value=10))
                    outputs = net(inputs)
                    outputs_label = outputs[0]
                    bbox_pred = outputs[1]
                    if i % 500 == 499:
                        inputs_copy = inputs.detach().clone()
                        inputs_copy = inputs_copy.cpu()
                        bbox_pc = bbox_pred.detach().clone()
                        bbox_pc[bbox_pc<0] = 0
                        bbox_pc[bbox_pc>31] = 31
                        bbox_pc[torch.isnan(bbox_pc)] = 0
                        _, predicted = torch.max(outputs_label.data, 1)
                        print("[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]  Predicted Labels:     " % 
                                (epoch+1, self.dl_studio.epochs, i+1, elapsed_time)  
                              + ' '.join('%10s' % self.dataserver_train.class_labels[predicted[j].item()] 
                                                                 for j in range(self.dl_studio.batch_size)))
                        for idx in range(self.dl_studio.batch_size):
                            i1 = int(bbox_gt[idx][1])
                            i2 = int(bbox_gt[idx][3])
                            j1 = int(bbox_gt[idx][0])
                            j2 = int(bbox_gt[idx][2])
                            k1 = int(bbox_pc[idx][1])
                            k2 = int(bbox_pc[idx][3])
                            l1 = int(bbox_pc[idx][0])
                            l2 = int(bbox_pc[idx][2])
                            print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                            print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                            inputs_copy[idx,0,i1:i2,j1] = 255
                            inputs_copy[idx,0,i1:i2,j2] = 255
                            inputs_copy[idx,0,i1,j1:j2] = 255
                            inputs_copy[idx,0,i2,j1:j2] = 255
                            inputs_copy[idx,2,k1:k2,l1] = 255                      
                            inputs_copy[idx,2,k1:k2,l2] = 255
                            inputs_copy[idx,2,k1,l1:l2] = 255
                            inputs_copy[idx,2,k2,l1:l2] = 255
                    loss_labeling = criterion1(outputs_label, labels)
                    loss_labeling.backward(retain_graph=True)        
                    loss_regression = criterion2(bbox_pred, bbox_gt)
                    loss_regression.backward()
                    optimizer.step()
                    running_loss_labeling += loss_labeling.item()    
                    running_loss_regression += loss_regression.item()                
                    if i % 500 == 499:    
                        avg_loss_labeling = running_loss_labeling / float(500)
                        avg_loss_regression = running_loss_regression / float(500)
                        labeling_loss_tally.append(avg_loss_labeling)  
                        regression_loss_tally.append(avg_loss_regression)    
                        print("[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]       loss_labeling %.3f        loss_regression: %.3f " %  (epoch+1, self.dl_studio.epochs, i+1, elapsed_time, avg_loss_labeling, avg_loss_regression))
                        FILE1.write("%.3f\n" % avg_loss_labeling)
                        FILE1.flush()
                        FILE2.write("%.3f\n" % avg_loss_regression)
                        FILE2.flush()
                        running_loss_labeling = 0.0
                        running_loss_regression = 0.0
                    if i%500==499:
                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[8,3])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(inputs_copy, normalize=True,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()
                        logger.setLevel(old_level)
            print("\nFinished Training\n")
            self.save_model(net)
            plt.figure(figsize=(10,5))
            plt.title("Labeling Loss vs. Iterations")
            plt.plot(labeling_loss_tally)
            plt.xlabel("iterations")
            plt.ylabel("labeling loss")
            plt.legend()
            plt.savefig("labeling_loss.png")
            plt.show()
            plt.title("regression Loss vs. Iterations")
            plt.plot(regression_loss_tally)
            plt.xlabel("iterations")
            plt.ylabel("regression loss")
            plt.legend()
            plt.savefig("regression_loss.png")
            plt.show()


        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)


        def run_code_for_testing_detection_and_localization(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            correct = 0
            total = 0
            confusion_matrix = torch.zeros(len(self.dataserver_train.class_labels), 
                                           len(self.dataserver_train.class_labels))
            class_correct = [0] * len(self.dataserver_train.class_labels)
            class_total = [0] * len(self.dataserver_train.class_labels)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    images, bounding_box, labels = data['image'], data['bbox'], data['label']
                    labels = labels.tolist()
                    if self.dl_studio.debug_test and i % 50 == 0:
                        print("\n\n[i=%d:] Ground Truth:     " %i + ' '.join('%10s' % 
                         self.dataserver_train.class_labels[labels[j]] for j in range(self.dl_studio.batch_size)))
                    outputs = net(images)
                    outputs_label = outputs[0]
                    outputs_regression = outputs[1]
                    outputs_regression[outputs_regression < 0] = 0
                    outputs_regression[outputs_regression > 31] = 31
                    outputs_regression[torch.isnan(outputs_regression)] = 0
                    output_bb = outputs_regression.tolist()
                    _, predicted = torch.max(outputs_label.data, 1)
                    predicted = predicted.tolist()
                    if self.dl_studio.debug_test and i % 50 == 0:
                        print("[i=%d:] Predicted Labels: " %i + ' '.join('%10s' % 
                              self.dataserver_train.class_labels[predicted[j]] for j in range(self.dl_studio.batch_size)))
                        for idx in range(self.dl_studio.batch_size):
                            i1 = int(bounding_box[idx][1])
                            i2 = int(bounding_box[idx][3])
                            j1 = int(bounding_box[idx][0])
                            j2 = int(bounding_box[idx][2])
                            k1 = int(output_bb[idx][1])
                            k2 = int(output_bb[idx][3])
                            l1 = int(output_bb[idx][0])
                            l2 = int(output_bb[idx][2])
                            print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                            print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                            images[idx,0,i1:i2,j1] = 255
                            images[idx,0,i1:i2,j2] = 255
                            images[idx,0,i1,j1:j2] = 255
                            images[idx,0,i2,j1:j2] = 255
                            images[idx,2,k1:k2,l1] = 255                      
                            images[idx,2,k1:k2,l2] = 255
                            images[idx,2,k1,l1:l2] = 255
                            images[idx,2,k2,l1:l2] = 255
                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[8,3])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(images, normalize=True,
                                                                         padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()
                        logger.setLevel(old_level)
                    for label,prediction in zip(labels,predicted):
                        confusion_matrix[label][prediction] += 1
                    total += len(labels)
                    correct +=  [predicted[ele] == labels[ele] for ele in range(len(predicted))].count(True)
                    comp = [predicted[ele] == labels[ele] for ele in range(len(predicted))]
                    for j in range(self.dl_studio.batch_size):
                        label = labels[j]
                        class_correct[label] += comp[j]
                        class_total[label] += 1
            print("\n")
            for j in range(len(self.dataserver_train.class_labels)):
                print('Prediction accuracy for %5s : %2d %%' % (
              self.dataserver_train.class_labels[j], 100 * class_correct[j] / class_total[j]))
            print("\n\n\nOverall accuracy of the network on the 1000 test images: %d %%" % 
                                                                   (100 * correct / float(total)))
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                "
            for j in range(len(self.dataserver_train.class_labels)):  
                                 out_str +=  "%15s" % self.dataserver_train.class_labels[j]   
            print(out_str + "\n")
            for i,label in enumerate(self.dataserver_train.class_labels):
                out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                                 for j in range(len(self.dataserver_train.class_labels))]
                out_percents = ["%.2f" % item.item() for item in out_percents]
                out_str = "%12s:  " % self.dataserver_train.class_labels[i]
                for j in range(len(self.dataserver_train.class_labels)): 
                                                       out_str +=  "%15s" % out_percents[j]
                print(out_str)


    ###%%%
    ########################################################################################
    ##################  Start Definition of Inner Class SemanticSegmentation  ##############

    class SemanticSegmentation(nn.Module):             
        """The purpose of this inner class is to be able to use the DLStudio module for
           experiments with semantic segmentation.  At its simplest level, the
           purpose of semantic segmentation is to assign correct labels to the
           different objects in a scene, while localizing them at the same time.  At
           a more sophisticated level, a system that carries out semantic
           segmentation should also output a symbolic expression based on the objects
           found in the image and their spatial relationships with one another.

           The workhorse of this inner class is the mUnet network that is based
           on the UNET network that was first proposed by Ronneberger, Fischer and
           Brox in the paper "U-Net: Convolutional Networks for Biomedical Image
           Segmentation".  Their Unet extracts binary masks for the cell pixel blobs
           of interest in biomedical images.  The output of their Unet can
           therefore be treated as a pixel-wise binary classifier at each pixel
           position.  The mUnet class, on the other hand, is intended for
           segmenting out multiple objects simultaneously form an image. [A weaker
           reason for "Multi" in the name of the class is that it uses skip
           connections not only across the two arms of the "U", but also also along
           the arms.  The skip connections in the original Unet are only between the
           two arms of the U.  In mUnet, each object type is assigned a separate
           channel in the output of the network.

           This version of DLStudio also comes with a new dataset,
           PurdueShapes5MultiObject, for experimenting with mUnet.  Each image in
           this dataset contains a random number of selections from five different
           shapes, with the shapes being randomly scaled, oriented, and located in
           each image.  The five different shapes are: rectangle, triangle, disk,
           oval, and star.

           Class Path:   DLStudio  ->  SemanticSegmentation
        """
        def __init__(self, dl_studio, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
            super(DLStudio.SemanticSegmentation, self).__init__()
            self.dl_studio = dl_studio
            self.dataserver_train = dataserver_train
            self.dataserver_test = dataserver_test

        class PurdueShapes5MultiObjectDataset(torch.utils.data.Dataset):
            """
            The very first thing to note is that the images in the dataset
            PurdueShapes5MultiObjectDataset are of size 64x64.  Each image has a
            random number (up to five) of the objects drawn from the following five
            shapes: rectangle, triangle, disk, oval, and star.  Each shape is
            randomized with respect to all its parameters, including those for its
            scale and location in the image.

            Each image in the dataset is represented by two data objects, one a list
            and the other a dictionary. The list data objects consists of the
            following items:

                [R, G, B, mask_array, mask_val_to_bbox_map]                                   ## (A)
            
            and the other data object is a dictionary that is set to:
            
                label_map = {'rectangle':50, 
                             'triangle' :100, 
                             'disk'     :150, 
                             'oval'     :200, 
                             'star'     :250}                                                 ## (B)
            
            Note that that second data object for each image is the same, as shown
            above.

            In the rest of this comment block, I'll explain in greater detail the
            elements of the list in line (A) above.

            
            R,G,B:
            ------

            Each of these is a 4096-element array whose elements store the
            corresponding color values at each of the 4096 pixels in a 64x64 image.
            That is, R is a list of 4096 integers, each between 0 and 255, for the
            value of the red component of the color at each pixel. Similarly, for G
            and B.
            

            mask_array:
            ----------

            The fourth item in the list shown in line (A) above is for the mask which is
            a numpy array of shape:
            
                           (5, 64, 64)
            
            It is initialized by the command:
            
                 mask_array = np.zeros((5,64,64), dtype=np.uint8)
            
            In essence, the mask_array consists of five planes, each of size 64x64.
            Each plane of the mask array represents an object type according to the
            following shape_index
            
                    shape_index = (label_map[shape] - 50) // 50
            
            where the label_map is as shown in line (B) above.  In other words, the
            shape_index values for the different shapes are:
            
                     rectangle:  0
                      triangle:  1
                          disk:  2
                          oval:  3
                          star:  4
            
            Therefore, the first layer (of index 0) of the mask is where the pixel
            values of 50 are stored at all those pixels that belong to the rectangle
            shapes.  Similarly, the second mask layer (of index 1) is where the pixel
            values of 100 are stored at all those pixel coordinates that belong to
            the triangle shapes in an image; and so on.
            
            It is in the manner described above that we define five different masks
            for an image in the dataset.  Each mask is for a different shape and the
            pixel values at the nonzero pixels in each mask layer are keyed to the
            shapes also.
            
            A reader is likely to wonder as to the need for this redundancy in the
            dataset representation of the shapes in each image.  Such a reader is
            likely to ask: Why can't we just use the binary values 1s and 0s in each
            mask layer where the corresponding pixels are in the image?  Setting
            these mask values to 50, 100, etc., was done merely for convenience.  I
            went with the intuition that the learning needed for multi-object
            segmentation would become easier if each shape was represented by a
            different pixels value in the corresponding mask. So I went ahead
            incorporated that in the dataset generation program itself.

            The mask values for the shapes are not to be confused with the actual RGB
            values of the pixels that belong to the shapes. The RGB values at the
            pixels in a shape are randomly generated.  Yes, all the pixels in a shape
            instance in an image have the same RGB values (but that value has nothing
            to do with the values given to the mask pixels for that shape).
            
            
            mask_val_to_bbox_map:
            --------------------
                   
            The fifth item in the list in line (A) above is a dictionary that tells us
            what bounding-box rectangle to associate with each shape in the image.  To
            illustrate what this dictionary looks like, assume that an image contains
            only one rectangle and only one disk, the dictionary in this case will look
            like:
            
                mask values to bbox mappings:  {200: [], 
                                                250: [], 
                                                100: [], 
                                                 50: [[56, 20, 63, 25]], 
                                                150: [[37, 41, 55, 59]]}
            
            Should there happen to be two rectangles in the same image, the dictionary
            would then be like:
            
                mask values to bbox mappings:  {200: [], 
                                                250: [], 
                                                100: [], 
                                                 50: [[56, 20, 63, 25], [18, 16, 32, 36]], 
                                                150: [[37, 41, 55, 59]]}
            
            Therefore, it is not a problem even if all the objects in an image are of
            the same type.  Remember, the object that are selected for an image are
            shown randomly from the different shapes.  By the way, an entry like '[56,
            20, 63, 25]' for the bounding box means that the upper-left corner of the
            BBox for the 'rectangle' shape is at (56,20) and the lower-right corner of
            the same is at the pixel coordinates (63,25).
            
            As far as the BBox quadruples are concerned, in the definition
            
                    [min_x,min_y,max_x,max_y]
            
            note that x is the horizontal coordinate, increasing to the right on your
            screen, and y is the vertical coordinate increasing downwards.

            Class Path:   DLStudio  ->  SemanticSegmentation  ->  PurdueShapes5MultiObjectDataset
            """
            def __init__(self, dl_studio, train_or_test, dataset_file):
                super(DLStudio.SemanticSegmentation.PurdueShapes5MultiObjectDataset, self).__init__()
                if train_or_test == 'train' and dataset_file == "PurdueShapes5MultiObject-10000-train.gz":
                    if os.path.exists("torch_saved_PurdueShapes5MultiObject-10000_dataset.pt") and \
                              os.path.exists("torch_saved_PurdueShapes5MultiObject_label_map.pt"):
                        print("\nLoading training data from torch saved file")
                        self.dataset = torch.load("torch_saved_PurdueShapes5MultiObject-10000_dataset.pt")
                        self.label_map = torch.load("torch_saved_PurdueShapes5MultiObject_label_map.pt")
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """up to 3 minutes.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        torch.save(self.dataset, "torch_saved_PurdueShapes5MultiObject-10000_dataset.pt")
                        torch.save(self.label_map, "torch_saved_PurdueShapes5MultiObject_label_map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                else:
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset, self.label_map = pickle.loads(dataset)
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                r = np.array( self.dataset[idx][0] )
                g = np.array( self.dataset[idx][1] )
                b = np.array( self.dataset[idx][2] )
                R,G,B = r.reshape(64,64), g.reshape(64,64), b.reshape(64,64)
                im_tensor = torch.zeros(3,64,64, dtype=torch.float)
                im_tensor[0,:,:] = torch.from_numpy(R)
                im_tensor[1,:,:] = torch.from_numpy(G)
                im_tensor[2,:,:] = torch.from_numpy(B)
#                mask_array = self.dataset[idx][3]
                mask_array = np.array(self.dataset[idx][3])
                mask_tensor = torch.from_numpy(mask_array)
                mask_val_to_bbox_map =  self.dataset[idx][4]
                max_bboxes_per_entry_in_map = max([ len(mask_val_to_bbox_map[key]) for key in mask_val_to_bbox_map ])
                ##  The first arg 5 is for the number of bboxes we are going to need. If all the
                ##  shapes are exactly the same, you are going to need five different bbox'es.
                ##  The second arg is the index reserved for each shape in a single bbox
                bbox_tensor = torch.zeros(5,5,4, dtype=torch.float)
                for bbox_idx in range(max_bboxes_per_entry_in_map):
                    for key in mask_val_to_bbox_map:
                        if len(mask_val_to_bbox_map[key]) == 1:
                            if bbox_idx == 0:
                                bbox_tensor[bbox_idx,key,:] = torch.from_numpy(np.array(mask_val_to_bbox_map[key][bbox_idx]))
                        elif len(mask_val_to_bbox_map[key]) > 1 and bbox_idx < len(mask_val_to_bbox_map[key]):
                            bbox_tensor[bbox_idx,key,:] = torch.from_numpy(np.array(mask_val_to_bbox_map[key][bbox_idx]))
                sample = {'image'        : im_tensor, 
                          'mask_tensor'  : mask_tensor,
                          'bbox_tensor'  : bbox_tensor }
                return sample

        def load_PurdueShapes5MultiObject_dataset(self, dataserver_train, dataserver_test ):   
            self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                        batch_size=self.dl_studio.batch_size,shuffle=True, num_workers=4)
            self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                               batch_size=self.dl_studio.batch_size,shuffle=False, num_workers=4)

        class SkipBlockDN(nn.Module):
            """
            This class for the skip connections in the downward leg of the "U"

            Class Path:   DLStudio  ->  SemanticSegmentation  ->  SkipBlockDN
            """
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(DLStudio.SemanticSegmentation.SkipBlockDN, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.bn2 = nn.BatchNorm2d(out_ch)
                if downsample:
                    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
            def forward(self, x):
                identity = x                                     
                out = self.convo1(x)                              
                out = self.bn1(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo2(out)                              
                    out = self.bn2(out)                              
                    out = torch.nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
                return out

        class SkipBlockUP(nn.Module):
            """
            This class is for the skip connections in the upward leg of the "U"

            Class Path:   DLStudio  ->  SemanticSegmentation  ->  SkipBlockUP
            """
            def __init__(self, in_ch, out_ch, upsample=False, skip_connections=True):
                super(DLStudio.SemanticSegmentation.SkipBlockUP, self).__init__()
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
                    out = torch.nn.functional.relu(out)
                if self.upsample:
                    out = self.upsampler(out)
                    identity = self.upsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out += identity[:,self.out_ch:,:,:]
                return out

        class mUnet(nn.Module):
            """
            This network is called mUnet because it is intended for segmenting
            out multiple objects simultaneously form an image. [A weaker reason for
            "Multi" in the name of the class is that it uses skip connections not
            only across the two arms of the "U", but also also along the arms.]  The
            classic UNET was first proposed by Ronneberger, Fischer and Brox in the
            paper "U-Net: Convolutional Networks for Biomedical Image Segmentation".
            Their UNET extracts binary masks for the cell pixel blobs of interest
            in biomedical images.  The output of their UNET therefore can therefore
            be treated as a pixel-wise binary classifier at each pixel position.  

            The mUnet presented here, on the other hand, is meant specifically
            for simultaneously identifying and localizing multiple objects in a
            given image.  Each object type is assigned a separate channel in the
            output of the network.  

            I have created a dataset, PurdueShapes5MultiObject, for experimenting
            with mUnet.  Each image in this dataset contains a random number of
            selections from five different shapes, with the shapes being randomly
            scaled, oriented, and located in each image.  The five different shapes
            are: rectangle, triangle, disk, oval, and star.

            Class Path:   DLStudio  ->  SemanticSegmentation  ->  mUnet
            """ 
            def __init__(self, skip_connections=True, depth=16):
                super(DLStudio.SemanticSegmentation.mUnet, self).__init__()
                self.depth = depth // 2
                self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
#                self.pool = nn.MaxPool2d(2, 2)
                ##  For the DN arm of the U:
                self.bn1DN  = nn.BatchNorm2d(64)
                self.bn2DN  = nn.BatchNorm2d(128)
                self.skip64DN_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64DN_arr.append(DLStudio.SemanticSegmentation.SkipBlockDN(64, 64,
                                                          skip_connections=skip_connections))
                self.skip64dsDN = DLStudio.SemanticSegmentation.SkipBlockDN(64, 64, 
                                           downsample=True, skip_connections=skip_connections)
                self.skip64to128DN = DLStudio.SemanticSegmentation.SkipBlockDN(64, 128, 
                                                            skip_connections=skip_connections )
                self.skip128DN_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128DN_arr.append(DLStudio.SemanticSegmentation.SkipBlockDN(128, 128,
                                                         skip_connections=skip_connections))
                self.skip128dsDN = DLStudio.SemanticSegmentation.SkipBlockDN(128,128,
                                            downsample=True, skip_connections=skip_connections)

                ##  For the UP arm of the U:
                self.bn1UP  = nn.BatchNorm2d(128)
                self.bn2UP  = nn.BatchNorm2d(64)
                self.skip64UP_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64UP_arr.append(DLStudio.SemanticSegmentation.SkipBlockUP(64, 64,
                                                          skip_connections=skip_connections))
                self.skip64usUP = DLStudio.SemanticSegmentation.SkipBlockUP(64, 64, 
                                           upsample=True, skip_connections=skip_connections)
                self.skip128to64UP = DLStudio.SemanticSegmentation.SkipBlockUP(128, 64, 
                                                            skip_connections=skip_connections )
                self.skip128UP_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128UP_arr.append(DLStudio.SemanticSegmentation.SkipBlockUP(128, 128,
                                                          skip_connections=skip_connections))
                self.skip128usUP = DLStudio.SemanticSegmentation.SkipBlockUP(128,128,
                                            upsample=True, skip_connections=skip_connections)
                self.conv_out = nn.ConvTranspose2d(64, 5, 3, stride=2,dilation=2,output_padding=1,padding=2)

            def forward(self, x):
                ##  Going down to the bottom of the U:
#                x = self.pool(torch.nn.functional.relu(self.conv_in(x)))          
                x = nn.MaxPool2d(2,2)(torch.nn.functional.relu(self.conv_in(x)))          
                for i,skip64 in enumerate(self.skip64DN_arr[:self.depth//4]):
                    x = skip64(x)                
                num_channels_to_save1 = x.shape[1] // 2
                save_for_upside_1 = x[:,:num_channels_to_save1,:,:].clone()
                x = self.skip64dsDN(x)
                for i,skip64 in enumerate(self.skip64DN_arr[self.depth//4:]):
                    x = skip64(x)                
                x = self.bn1DN(x)
                num_channels_to_save2 = x.shape[1] // 2
                save_for_upside_2 = x[:,:num_channels_to_save2,:,:].clone()
                x = self.skip64to128DN(x)
                for i,skip128 in enumerate(self.skip128DN_arr[:self.depth//4]):
                    x = skip128(x)                
                x = self.bn2DN(x)
                num_channels_to_save3 = x.shape[1] // 2
                save_for_upside_3 = x[:,:num_channels_to_save3,:,:].clone()
                for i,skip128 in enumerate(self.skip128DN_arr[self.depth//4:]):
                    x = skip128(x)                
                x = self.skip128dsDN(x)

                ## Coming up from the bottom of U on the other side:
                x = self.skip128usUP(x)          
                for i,skip128 in enumerate(self.skip128UP_arr[:self.depth//4]):
                    x = skip128(x)                
                x[:,:num_channels_to_save3,:,:] =  save_for_upside_3
                x = self.bn1UP(x)
                for i,skip128 in enumerate(self.skip128UP_arr[:self.depth//4]):
                    x = skip128(x)                
                x = self.skip128to64UP(x)
                for i,skip64 in enumerate(self.skip64UP_arr[self.depth//4:]):
                    x = skip64(x)                
                x[:,:num_channels_to_save2,:,:] =  save_for_upside_2
                x = self.bn2UP(x)
                x = self.skip64usUP(x)
                for i,skip64 in enumerate(self.skip64UP_arr[:self.depth//4]):
                    x = skip64(x)                
                x[:,:num_channels_to_save1,:,:] =  save_for_upside_1
                x = self.conv_out(x)
                return x

        class SegmentationLoss(nn.Module):
            """
            I wrote this class before I switched to MSE loss.  I am leaving it here
            in case I need to get back to it in the future.  

            Class Path:   DLStudio  ->  SemanticSegmentation  ->  SegmentationLoss
            """
            def __init__(self, batch_size):
                super(DLStudio.SemanticSegmentation.SegmentationLoss, self).__init__()
                self.batch_size = batch_size
            def forward(self, output, mask_tensor):
                composite_loss = torch.zeros(1,self.batch_size)
                mask_based_loss = torch.zeros(1,5)
                for idx in range(self.batch_size):
                    outputh = output[idx,0,:,:]
                    for mask_layer_idx in range(mask_tensor.shape[0]):
                        mask = mask_tensor[idx,mask_layer_idx,:,:]
                        element_wise = (outputh - mask)**2                   
                        mask_based_loss[0,mask_layer_idx] = torch.mean(element_wise)
                    composite_loss[0,idx] = torch.sum(mask_based_loss)
                return torch.sum(composite_loss) / self.batch_size

        def run_code_for_training_for_semantic_segmentation(self, net):        
            filename_for_out1 = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE1 = open(filename_for_out1, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            criterion1 = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            start_time = time.perf_counter()
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss_segmentation = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    im_tensor,mask_tensor,bbox_tensor =data['image'],data['mask_tensor'],data['bbox_tensor']
                    im_tensor   = im_tensor.to(self.dl_studio.device)
                    mask_tensor = mask_tensor.type(torch.FloatTensor)
                    mask_tensor = mask_tensor.to(self.dl_studio.device)                 
                    bbox_tensor = bbox_tensor.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    output = net(im_tensor) 
                    segmentation_loss = criterion1(output, mask_tensor)  
                    segmentation_loss.backward()
                    optimizer.step()
                    running_loss_segmentation += segmentation_loss.item()    
                    if i%500==499:    
                        current_time = time.perf_counter()
                        elapsed_time = current_time - start_time
                        avg_loss_segmentation = running_loss_segmentation / float(500)
                        print("[epoch=%d/%d, iter=%4d  elapsed_time=%3d secs]   MSE loss: %.3f" % (epoch+1, self.dl_studio.epochs, i+1, elapsed_time, avg_loss_segmentation))
                        FILE1.write("%.3f\n" % avg_loss_segmentation)
                        FILE1.flush()
                        running_loss_segmentation = 0.0
            print("\nFinished Training\n")
            self.save_model(net)

        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)

        def run_code_for_testing_semantic_segmentation(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    im_tensor,mask_tensor,bbox_tensor =data['image'],data['mask_tensor'],data['bbox_tensor']
                    if self.dl_studio.debug_test and i % 50 == 0:
                        print("\n\n\n\nShowing output for test batch %d: " % (i+1))
                        outputs = net(im_tensor)                        
                        ## In the statement below: 1st arg for batch items, 2nd for channels, 
                        ##                         3rd and 4th for image size
                        output_bw_tensor = torch.zeros(4,1,64,64, dtype=float)
                        for image_idx in range(self.dl_studio.batch_size):
                            for layer_idx in range(5):
                                for m in range(64):
                                    for n in range(64):
                                        output_bw_tensor[image_idx,0,m,n]  =  \
                                                  torch.max( outputs[image_idx,:,m,n] )
#                        display_tensor = torch.zeros(8,3,64,64, dtype=float)
                        display_tensor = torch.zeros(28,3,64,64, dtype=float)
                        for idx in range(self.dl_studio.batch_size):
                            for bbox_idx in range(5):         ## 5 for the five different types of obj
                                bb_tensor = bbox_tensor[idx,bbox_idx]
                                for k in range(5):
                                    i1 = int(bb_tensor[k][1])
                                    i2 = int(bb_tensor[k][3])
                                    j1 = int(bb_tensor[k][0])
                                    j2 = int(bb_tensor[k][2])
                                    output_bw_tensor[idx,0,i1:i2,j1] = 255
                                    output_bw_tensor[idx,0,i1:i2,j2] = 255
                                    output_bw_tensor[idx,0,i1,j1:j2] = 255
                                    output_bw_tensor[idx,0,i2,j1:j2] = 255
                                    im_tensor[idx,0,i1:i2,j1] = 255
                                    im_tensor[idx,0,i1:i2,j2] = 255
                                    im_tensor[idx,0,i1,j1:j2] = 255
                                    im_tensor[idx,0,i2,j1:j2] = 255
                        display_tensor[:4,:,:,:] = output_bw_tensor
                        display_tensor[4:8,:,:,:] = im_tensor

                        for batch_im_idx in range(self.dl_studio.batch_size):
                            for mask_layer_idx in range(5):
                                for i in range(64):
                                    for j in range(64):
                                        if mask_layer_idx == 0:
                                            if 25 < outputs[batch_im_idx,mask_layer_idx,i,j] < 85:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 1:
                                            if 65 < outputs[batch_im_idx,mask_layer_idx,i,j] < 135:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 2:
                                            if 115 < outputs[batch_im_idx,mask_layer_idx,i,j] < 185:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 3:
                                            if 165 < outputs[batch_im_idx,mask_layer_idx,i,j] < 230:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 4:
                                            if outputs[batch_im_idx,mask_layer_idx,i,j] > 210:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50

                                display_tensor[8+4*mask_layer_idx+batch_im_idx,:,:,:]= \
                                                          outputs[batch_im_idx,mask_layer_idx,:,:]

                        self.dl_studio.display_tensor_as_image(
                           torchvision.utils.make_grid(display_tensor, nrow=4, normalize=True, padding=2, pad_value=10))



    ###%%%
    ########################################################################################
    ##################  Start Definition of Inner Class TextClassification  ################

    class TextClassification(nn.Module):             
        """
        The purpose of this inner class is to be able to use the DLStudio module for simple 
        experiments in text classification.  Consider, for example, the problem of automatic 
        classification of variable-length user feedback: you want to create a neural network
        that can label an uploaded product review of arbitrary length as positive or negative.  
        One way to solve this problem is with a recurrent neural network in which you use a 
        hidden state for characterizing a variable-length product review with a fixed-length 
        state vector.  This inner class allows you to carry out such experiments.

        Class Path:  DLStudio -> TextClassification 
        """
        def __init__(self, dl_studio, dataserver_train=None, dataserver_test=None, dataset_file_train=None, 
                                                               dataset_file_test=None, display_train_loss=False):
            super(DLStudio.TextClassification, self).__init__()
            self.dl_studio = dl_studio
            self.dataserver_train = dataserver_train
            self.dataserver_test = dataserver_test
            self.display_train_loss = display_train_loss

        class SentimentAnalysisDataset(torch.utils.data.Dataset):
            """
            The sentiment analysis datasets that I have made available were extracted from
            an archive of user feedback comments as made available by Amazon for the year
            2007.  The original archive contains user feedback on 25 product categories. 
            For each product category, there are two files named 'positive.reviews' and
            'negative.reviews', with each file containing 1000 reviews. I believe that
            characterizing the reviews as 'positive' or 'negative' was carried out by 
            human annotators. Regardless, the reviews in these two files can be used to 
            train a neural network whose purpose would be to automatically characterize
            a product as being positive or negative. 

            I have extracted the following datasets extracted from the Amazon archive:

                 sentiment_dataset_train_200.tar.gz        vocab_size = 43,285
                 sentiment_dataset_test_200.tar.gz  

                 sentiment_dataset_train_40.tar.gz         vocab_size = 17,001
                 sentiment_dataset_test_40.tar.gz    

                 sentiment_dataset_train_3.tar.gz          vocab_size = 3,402
                 sentiment_dataset_test_3.tar.gz    

            The integer in the name of each dataset is the number of reviews collected 
            from the 'positive.reviews' and the 'negative.reviews' files for each product
            category.  Therefore, the dataset with 200 in its name has a total of 400 
            reviews for each product category.

            As to why I am presenting these three different datasets, note that, as shown
            above, the size of the vocabulary depends on the number of reviews selected
            and the size of the vocabulary has a strong bearing on how long it takes to 
            train an algorithm for text classification. For one simple reason for that: 
            the size of the one-hot representation for the words equals the size of the 
            vocabulary.  Therefore, the one-hot representation for the words for the 
            dataset with 200 in its name will be a one-axis tensor of size 43,285.

            For a purely feedforward network, it is not a big deal for the input tensors
            to be size Nx43285 where N is the number of words in a review.  And even for
            RNNs with simple feedback, that does not slow things down.  However, when 
            using GRUs, it's an entirely different matter if you are tying to run your
            experiments on, say, a laptop with a Quadro GPU.  Hence the reason for providing
            the datasets with 200 and 40 reviews.  The dataset with just 3 reviews is for
            debugging your code.

            Class Path:  DLStudio -> TextClassification -> SentimentAnalysisDataset
            """
            def __init__(self, dl_studio, train_or_test, dataset_file):
                super(DLStudio.TextClassification.SentimentAnalysisDataset, self).__init__()
                self.train_or_test = train_or_test
                root_dir = dl_studio.dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if train_or_test == 'train':
                    if sys.version_info[0] == 3:
                        self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset)
                    self.categories = sorted(list(self.positive_reviews_train.keys()))
                    self.category_sizes_train_pos = {category : len(self.positive_reviews_train[category]) for category in self.categories}
                    self.category_sizes_train_neg = {category : len(self.negative_reviews_train[category]) for category in self.categories}
                    self.indexed_dataset_train = []
                    for category in self.positive_reviews_train:
                        for review in self.positive_reviews_train[category]:
                            self.indexed_dataset_train.append([review, category, 1])
                    for category in self.negative_reviews_train:
                        for review in self.negative_reviews_train[category]:
                            self.indexed_dataset_train.append([review, category, 0])
                    random.shuffle(self.indexed_dataset_train)
                elif train_or_test == 'test':
                    if sys.version_info[0] == 3:
                        self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset)
                    self.vocab = sorted(self.vocab)
                    self.categories = sorted(list(self.positive_reviews_test.keys()))
                    self.category_sizes_test_pos = {category : len(self.positive_reviews_test[category]) for category in self.categories}
                    self.category_sizes_test_neg = {category : len(self.negative_reviews_test[category]) for category in self.categories}
                    self.indexed_dataset_test = []
                    for category in self.positive_reviews_test:
                        for review in self.positive_reviews_test[category]:
                            self.indexed_dataset_test.append([review, category, 1])
                    for category in self.negative_reviews_test:
                        for review in self.negative_reviews_test[category]:
                            self.indexed_dataset_test.append([review, category, 0])
                    random.shuffle(self.indexed_dataset_test)

            def get_vocab_size(self):
                return len(self.vocab)

            def one_hotvec_for_word(self, word):
                word_index =  self.vocab.index(word)
                hotvec = torch.zeros(1, len(self.vocab))
                hotvec[0, word_index] = 1
                return hotvec

            def review_to_tensor(self, review):
                review_tensor = torch.zeros(len(review), len(self.vocab))
                for i,word in enumerate(review):
                    review_tensor[i,:] = self.one_hotvec_for_word(word)
                return review_tensor

            def sentiment_to_tensor(self, sentiment):
                """
                Sentiment is ordinarily just a binary valued thing.  It is 0 for negative
                sentiment and 1 for positive sentiment.  We need to pack this value in a
                two-element tensor.
                """        
                sentiment_tensor = torch.zeros(2)
                if sentiment == 1:
                    sentiment_tensor[1] = 1
                elif sentiment == 0: 
                    sentiment_tensor[0] = 1
                sentiment_tensor = sentiment_tensor.type(torch.long)
                return sentiment_tensor

            def __len__(self):
                if self.train_or_test == 'train':
                    return len(self.indexed_dataset_train)
                elif self.train_or_test == 'test':
                    return len(self.indexed_dataset_test)

            def __getitem__(self, idx):
                sample = self.indexed_dataset_train[idx] if self.train_or_test == 'train' else self.indexed_dataset_test[idx]
                review = sample[0]
                review_category = sample[1]
                review_sentiment = sample[2]
                review_sentiment = self.sentiment_to_tensor(review_sentiment)
                review_tensor = self.review_to_tensor(review)
                category_index = self.categories.index(review_category)
                sample = {'review'       : review_tensor, 
                          'category'     : category_index, # should be converted to tensor, but not yet used
                          'sentiment'    : review_sentiment }
                return sample

        def load_SentimentAnalysisDataset(self, dataserver_train, dataserver_test ):   
            self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                        batch_size=self.dl_studio.batch_size,shuffle=True, num_workers=1)
            self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                               batch_size=self.dl_studio.batch_size,shuffle=False, num_workers=1)

        class TEXTnet(nn.Module):
            """
            This network is meant for semantic classification of variable-length sentiment 
            data.  Based on my limited testing, the performance of this network is very
            poor because it has no protection against vanishing gradients when used in an
            RNN.

            Class Path:  DLStudio -> TextClassification -> TEXTnet
            """
            def __init__(self, input_size, hidden_size, output_size):
                super(DLStudio.TextClassification.TEXTnet, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = output_size
                self.combined_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
                self.combined_to_middle = nn.Linear(input_size + hidden_size, 100)
                self.middle_to_out = nn.Linear(100, output_size)     
                self.logsoftmax = nn.LogSoftmax(dim=1)
                self.dropout = nn.Dropout(p=0.1)

            def forward(self, input, hidden):
                combined = torch.cat((input, hidden), 1)
                hidden = self.combined_to_hidden(combined)
                hidden = torch.tanh(hidden)                   
                out = self.combined_to_middle(combined)
                out = torch.nn.functional.relu(out)
                out = self.dropout(out)
                out = self.middle_to_out(out)
                out = self.logsoftmax(out)
                return out,hidden         

            def init_hidden(self):
                hidden = torch.zeros(1, self.hidden_size)
                return hidden


        class TEXTnetOrder2(nn.Module):
            """
            In this variant of the TEXTnet network, the value of hidden as used at
            each time step also includes its value at the previous time step.  This 
            fact, not directly apparent by the definition of the class shown below, 
            is made possible by the last parameter, cell, in the header of forward().  
            All you can see here, at the end of forward(), is that the value of cell 
            goes through a linear layer and through a sigmoid nonlinearity. By the way, 
            since the sigmoid saturates at 0 and 1, it can act like a switch. Later 
            when I use this class in the training function, you will see the cell
            values being used in such a manner that the hidden state at each time
            step is mixed with the hidden state at the previous time step.

            Class Path:  DLStudio -> TextClassification -> EXTnetOrder2
            """
            def __init__(self, input_size, hidden_size, output_size):
                super(DLStudio.TextClassification.TEXTnetOrder2, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = output_size
                self.combined_to_hidden = nn.Linear(input_size + 2*hidden_size, hidden_size)
                self.combined_to_middle = nn.Linear(input_size + 2*hidden_size, 100)
                self.middle_to_out = nn.Linear(100, output_size)     
                self.logsoftmax = nn.LogSoftmax(dim=1)
                self.dropout = nn.Dropout(p=0.1)
                # for the cell
                self.linear_for_cell = nn.Linear(hidden_size, hidden_size)

            def forward(self, input, hidden, cell):
                combined = torch.cat((input, hidden, cell), 1)
                hidden = self.combined_to_hidden(combined)
                hidden = torch.tanh(hidden)                     
                out = self.combined_to_middle(combined)
                out = torch.nn.functional.relu(out)
                out = self.dropout(out)
                out = self.middle_to_out(out)
                out = self.logsoftmax(out)
                hidden_clone = hidden.clone()
                cell = torch.sigmoid(self.linear_for_cell(hidden_clone))
                return out,hidden,cell         

            def initialize_cell(self):
                weight = next(self.linear_for_cell.parameters()).data
                cell = weight.new(1, self.hidden_size).zero_()
                return cell

            def init_hidden(self):
                hidden = torch.zeros(1, self.hidden_size)
                return hidden


        class GRUnet(nn.Module):
            """
            Source: https://blog.floydhub.com/gru-with-pytorch/
            with the only modification that the final output of forward() is now
            routed through LogSoftmax activation. 

            Class Path: DLStudio  ->  TextClassification  ->  GRUnet
            """
            def __init__(self, input_size, hidden_size, output_size, num_layers, drop_prob=0.2):
                super(DLStudio.TextClassification.GRUnet, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.gru = nn.GRU(input_size, hidden_size, num_layers)
                self.fc = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
                self.logsoftmax = nn.LogSoftmax(dim=1)
                
            def forward(self, x, h):
                out, h = self.gru(x, h)
                out = self.fc(self.relu(out[:,-1]))
                out = self.logsoftmax(out)
                return out, h

            def init_hidden(self):
                weight = next(self.parameters()).data
                #                                     batch_size   
                hidden = weight.new(  self.num_layers,     1,         self.hidden_size   ).zero_()
                return hidden

        def save_model(self, model):
            "Save the trained model to a disk file"
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)


        def run_code_for_training_with_TEXTnet(self, net, display_train_loss=False):        
            filename_for_out = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE = open(filename_for_out, 'w')
            net.to(self.dl_studio.device)
            ## Note that the TEXTnet and TEXTnetOrder2 both produce LogSoftmax output:
            criterion = nn.NLLLoss()
            accum_times = []
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            start_time = time.perf_counter()
            training_loss_tally = []
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    hidden = net.init_hidden().to(self.dl_studio.device)              
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    review_tensor = review_tensor.to(self.dl_studio.device)
                    sentiment = sentiment.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    input = torch.zeros(1,review_tensor.shape[2])
                    input = input.to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        input[0,:] = review_tensor[0,k]
                        output, hidden = net(input, hidden)
                    loss = criterion(output, torch.argmax(sentiment,1))
                    running_loss += loss.item()
                    loss.backward(retain_graph=True)        
                    optimizer.step()
                    if i % 200 == 199:    
                        avg_loss = running_loss / float(200)
                        training_loss_tally.append(avg_loss)
                        current_time = time.perf_counter()
                        time_elapsed = current_time-start_time
                        print("[epoch:%d  iter:%4d  elapsed_time: %4d secs]     loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                        accum_times.append(current_time-start_time)
                        FILE.write("%.3f\n" % avg_loss)
                        FILE.flush()
                        running_loss = 0.0
            print("\nFinished Training\n")
            self.save_model(net)
            if display_train_loss:
                plt.figure(figsize=(10,5))
                plt.title("Training Loss vs. Iterations")
                plt.plot(training_loss_tally)
                plt.xlabel("iterations")
                plt.ylabel("training loss")
                plt.legend()
                plt.savefig("training_loss.png")
                plt.show()


        def run_code_for_training_with_TEXTnetOrder2(self, net, display_train_loss=False):        
            filename_for_out = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE = open(filename_for_out, 'w')
            net.to(self.dl_studio.device)
            ## Note that the TEXTnet and TEXTnetOrder2 both produce LogSoftmax output:
            criterion = nn.NLLLoss()
            accum_times = []
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            start_time = time.perf_counter()
            training_loss_tally = []
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    hidden = net.init_hidden().to(self.dl_studio.device)              
                    cell_prev = net.initialize_cell().to(self.dl_studio.device)
                    cell_prev_2_prev = net.initialize_cell().to(self.dl_studio.device)
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    review_tensor = review_tensor.to(self.dl_studio.device)
                    sentiment = sentiment.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    input = torch.zeros(1,review_tensor.shape[2])
                    input = input.to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        input[0,:] = review_tensor[0,k]
                        output, hidden, cell = net(input, hidden, cell_prev_2_prev)
                        if k == 0:
                            cell_prev = cell
                        else:
                            cell_prev_2_prev = cell_prev
                            cell_prev = cell
                    loss = criterion(output, torch.argmax(sentiment,1))
                    running_loss += loss.item()
                    loss.backward()        
                    optimizer.step()
                    if i % 200 == 199:    
                        avg_loss = running_loss / float(200)
                        training_loss_tally.append(avg_loss)
                        current_time = time.perf_counter()
                        time_elapsed = current_time-start_time
                        print("[epoch:%d  iter:%4d  elapsed_time: %4d secs]     loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                        accum_times.append(current_time-start_time)
                        FILE.write("%.3f\n" % avg_loss)
                        FILE.flush()
                        running_loss = 0.0
            print("\nFinished Training\n")
            self.save_model(net)
            if display_train_loss:
                plt.figure(figsize=(10,5))
                plt.title("Training Loss vs. Iterations")
                plt.plot(training_loss_tally)
                plt.xlabel("iterations")
                plt.ylabel("training loss")
                plt.legend()
                plt.savefig("training_loss.png")
                plt.show()


        def run_code_for_training_for_text_classification_with_GRU(self, net, display_train_loss=False): 
            filename_for_out = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE = open(filename_for_out, 'w')
            net.to(self.dl_studio.device)
            ##  Note that the GREnet now produces the LogSoftmax output:
            criterion = nn.NLLLoss()
            accum_times = []
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            start_time = time.perf_counter()
            training_loss_tally = []
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    review_tensor = review_tensor.to(self.dl_studio.device)
                    sentiment = sentiment.to(self.dl_studio.device)
                    ## The following type conversion needed for MSELoss:
                    ##sentiment = sentiment.float()
                    optimizer.zero_grad()
                    hidden = net.init_hidden().to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                    ## If using NLLLoss, CrossEntropyLoss
                    loss = criterion(output, torch.argmax(sentiment, 1))
                    ## If using MSELoss:
                    ## loss = criterion(output, sentiment)     
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    if i % 200 == 199:    
                        avg_loss = running_loss / float(200)
                        training_loss_tally.append(avg_loss)
                        current_time = time.perf_counter()
                        time_elapsed = current_time-start_time
                        print("[epoch:%d  iter:%4d  elapsed_time:%4d secs]     loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                        accum_times.append(current_time-start_time)
                        FILE.write("%.3f\n" % avg_loss)
                        FILE.flush()
                        running_loss = 0.0
            print("Total Training Time: {}".format(str(sum(accum_times))))
            print("\nFinished Training\n")
            self.save_model(net)
            if display_train_loss:
                plt.figure(figsize=(10,5))
                plt.title("Training Loss vs. Iterations")
                plt.plot(training_loss_tally)
                plt.xlabel("iterations")
                plt.ylabel("training loss")
                plt.legend()
                plt.savefig("training_loss.png")
                plt.show()


        def run_code_for_testing_with_TEXTnet(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            net.to(self.dl_studio.device)
            classification_accuracy = 0.0
            negative_total = 0
            positive_total = 0
            confusion_matrix = torch.zeros(2,2)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    input = torch.zeros(1,review_tensor.shape[2]).to(self.dl_studio.device)
                    hidden = net.init_hidden().to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        input[0,:] = review_tensor[0,k]
                        output, hidden = net(input, hidden)
                    predicted_idx = torch.argmax(output).item()
                    gt_idx = torch.argmax(sentiment).item()
                    if i % 100 == 99:
                        print("   [i=%4d]    predicted_label=%d       gt_label=%d" % (i+1, predicted_idx,gt_idx))
                    if predicted_idx == gt_idx:
                        classification_accuracy += 1
                    if gt_idx == 0: 
                        negative_total += 1
                    elif gt_idx == 1:
                        positive_total += 1
                    confusion_matrix[gt_idx,predicted_idx] += 1
            print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(i)))
            out_percent = np.zeros((2,2), dtype='float')
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            print("\n\nNumber of positive reviews tested: %d" % positive_total)
            print("\n\nNumber of negative reviews tested: %d" % negative_total)
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                      "
            out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
            print(out_str + "\n")
            for i,label in enumerate(['true negative', 'true positive']):
                out_str = "%12s:  " % label
                for j in range(2):
                    out_str +=  "%18s" % out_percent[i,j]
                print(out_str)

        def run_code_for_testing_with_TEXTnetOrder2(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            net.to(self.dl_studio.device)
            classification_accuracy = 0.0
            negative_total = 0
            positive_total = 0
            confusion_matrix = torch.zeros(2,2)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    cell_prev = net.initialize_cell()
                    cell_prev_2_prev = net.initialize_cell()
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    input = torch.zeros(1,review_tensor.shape[2]).to(self.dl_studio.device)
                    hidden = net.init_hidden().to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        input[0,:] = review_tensor[0,k]
                        output, hidden, cell = net(input, hidden, cell_prev_2_prev)
                        if k == 0:
                            cell_prev = cell
                        else:
                            cell_prev_2_prev = cell_prev
                            cell_prev = cell
                    predicted_idx = torch.argmax(output).item()
                    gt_idx = torch.argmax(sentiment).item()
                    if i % 100 == 99:
                        print("   [i=%4d]    predicted_label=%d       gt_label=%d" % (i+1, predicted_idx,gt_idx))
                    if predicted_idx == gt_idx:
                        classification_accuracy += 1
                    if gt_idx == 0: 
                        negative_total += 1
                    elif gt_idx == 1:
                        positive_total += 1
                    confusion_matrix[gt_idx,predicted_idx] += 1
            print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(i)))
            out_percent = np.zeros((2,2), dtype='float')
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            print("\n\nNumber of positive reviews tested: %d" % positive_total)
            print("\n\nNumber of negative reviews tested: %d" % negative_total)
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                      "
            out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
            print(out_str + "\n")
            for i,label in enumerate(['true negative', 'true positive']):
                out_str = "%12s:  " % label
                for j in range(2):
                    out_str +=  "%18s" % out_percent[i,j]
                print(out_str)


        def run_code_for_testing_text_classification_with_GRU(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            net.to(self.dl_studio.device)
            classification_accuracy = 0.0
            negative_total = 0
            positive_total = 0
            confusion_matrix = torch.zeros(2,2)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    hidden = net.init_hidden().to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                    predicted_idx = torch.argmax(output).item()
                    gt_idx = torch.argmax(sentiment).item()
                    if i % 100 == 99:
                        print("   [i=%d]    predicted_label=%d       gt_label=%d\n\n" % (i+1, predicted_idx,gt_idx))
                    if predicted_idx == gt_idx:
                        classification_accuracy += 1
                    if gt_idx == 0: 
                        negative_total += 1
                    elif gt_idx == 1:
                        positive_total += 1
                    confusion_matrix[gt_idx,predicted_idx] += 1
            print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(i)))
            out_percent = np.zeros((2,2), dtype='float')
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            print("\n\nNumber of positive reviews tested: %d" % positive_total)
            print("\n\nNumber of negative reviews tested: %d" % negative_total)
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                      "
            out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
            print(out_str + "\n")
            for i,label in enumerate(['true negative', 'true positive']):
                out_str = "%12s:  " % label
                for j in range(2):
                    out_str +=  "%18s" % out_percent[i,j]
                print(out_str)


    ###%%%
    ########################################################################################
    ########  Start Definition of Inner Class TextClassificationWithEmbeddings  ############

    class TextClassificationWithEmbeddings(nn.Module):             
        """
        The text processing class described previously, TextClassification, was based on
        using one-hot vectors for representing the words.  The main challenge we faced
        with one-hot vectors was that the larger the size of the training dataset, the
        larger the size of the vocabulary, and, therefore, the larger the size of the
        one-hot vectors.  The increase in the size of the one-hot vectors led to a
        model with a significantly larger number of learnable parameters --- and, that,
        in turn, created a need for a still larger training dataset.  Sounds like a classic
        example of a vicious circle.  In this section, I use the idea of word embeddings
        to break out of this vicious circle.

        Word embeddings are fixed-sized numerical representations for words that are
        learned on the basis of the similarity of word contexts.  The original and still
        the most famous of these representations are known as the word2vec
        embeddings. The embeddings that I use in this section consist of pre-trained
        300-element word vectors for 3 million words and phrases as learned from Google
        News reports.  I access these embeddings through the popular Gensim library.
 
        Class Path:  DLStudio -> TextClassificationWithEmbeddings
        """
        def __init__(self, dl_studio,dataserver_train=None,dataserver_test=None,dataset_file_train=None,dataset_file_test=None):
            super(DLStudio.TextClassificationWithEmbeddings, self).__init__()
            self.dl_studio = dl_studio
            self.dataserver_train = dataserver_train
            self.dataserver_test = dataserver_test

        class SentimentAnalysisDataset(torch.utils.data.Dataset):
            """
            In relation to the SentimentAnalysisDataset defined for the TextClassification section of 
            DLStudio, the __getitem__() method of the dataloader must now fetch the embeddings from
            the word2vec word vectors.

            Class Path:  DLStudio -> TextClassificationWithEmbeddings -> SentimentAnalysisDataset
            """
            def __init__(self, dl_studio, train_or_test, dataset_file, path_to_saved_embeddings=None):
                super(DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset, self).__init__()
                import gensim.downloader as gen_api
#                self.word_vectors = gen_api.load("word2vec-google-news-300")
                self.path_to_saved_embeddings = path_to_saved_embeddings
                self.train_or_test = train_or_test
                root_dir = dl_studio.dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if path_to_saved_embeddings is not None:
                    import gensim.downloader as genapi
                    from gensim.models import KeyedVectors 
                    if os.path.exists(path_to_saved_embeddings + 'vectors.kv'):
                        self.word_vectors = KeyedVectors.load(path_to_saved_embeddings + 'vectors.kv')
                    else:
                        print("""\n\nSince this is your first time to install the word2vec embeddings, it may take"""
                              """\na couple of minutes. The embeddings occupy around 3.6GB of your disk space.\n\n""")
                        self.word_vectors = genapi.load("word2vec-google-news-300")               
                        ##  'kv' stands for  "KeyedVectors", a special datatype used by gensim because it 
                        ##  has a smaller footprint than dict
                        self.word_vectors.save(path_to_saved_embeddings + 'vectors.kv')    
                if train_or_test == 'train':
                    if sys.version_info[0] == 3:
                        self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset)
                    self.categories = sorted(list(self.positive_reviews_train.keys()))
                    self.category_sizes_train_pos = {category : len(self.positive_reviews_train[category]) for category in self.categories}
                    self.category_sizes_train_neg = {category : len(self.negative_reviews_train[category]) for category in self.categories}
                    self.indexed_dataset_train = []
                    for category in self.positive_reviews_train:
                        for review in self.positive_reviews_train[category]:
                            self.indexed_dataset_train.append([review, category, 1])
                    for category in self.negative_reviews_train:
                        for review in self.negative_reviews_train[category]:
                            self.indexed_dataset_train.append([review, category, 0])
                    random.shuffle(self.indexed_dataset_train)
                elif train_or_test == 'test':
                    if sys.version_info[0] == 3:
                        self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset)
                    self.vocab = sorted(self.vocab)
                    self.categories = sorted(list(self.positive_reviews_test.keys()))
                    self.category_sizes_test_pos = {category : len(self.positive_reviews_test[category]) for category in self.categories}
                    self.category_sizes_test_neg = {category : len(self.negative_reviews_test[category]) for category in self.categories}
                    self.indexed_dataset_test = []
                    for category in self.positive_reviews_test:
                        for review in self.positive_reviews_test[category]:
                            self.indexed_dataset_test.append([review, category, 1])
                    for category in self.negative_reviews_test:
                        for review in self.negative_reviews_test[category]:
                            self.indexed_dataset_test.append([review, category, 0])
                    random.shuffle(self.indexed_dataset_test)

            def review_to_tensor(self, review):
                list_of_embeddings = []
                for i,word in enumerate(review):
                    if word in self.word_vectors.key_to_index:
                        embedding = self.word_vectors[word]
                        list_of_embeddings.append(np.array(embedding))
                    else:
                        next
                review_tensor = torch.FloatTensor( list_of_embeddings )
                return review_tensor

            def sentiment_to_tensor(self, sentiment):
                """
                Sentiment is ordinarily just a binary valued thing.  It is 0 for negative
                sentiment and 1 for positive sentiment.  We need to pack this value in a
                two-element tensor.
                """        
                sentiment_tensor = torch.zeros(2)
                if sentiment == 1:
                    sentiment_tensor[1] = 1
                elif sentiment == 0: 
                    sentiment_tensor[0] = 1
                sentiment_tensor = sentiment_tensor.type(torch.long)
                return sentiment_tensor

            def __len__(self):
                if self.train_or_test == 'train':
                    return len(self.indexed_dataset_train)
                elif self.train_or_test == 'test':
                    return len(self.indexed_dataset_test)

            def __getitem__(self, idx):
                sample = self.indexed_dataset_train[idx] if self.train_or_test == 'train' else self.indexed_dataset_test[idx]
                review = sample[0]
                review_category = sample[1]
                review_sentiment = sample[2]
                review_sentiment = self.sentiment_to_tensor(review_sentiment)
                review_tensor = self.review_to_tensor(review)
                category_index = self.categories.index(review_category)
                sample = {'review'       : review_tensor, 
                          'category'     : category_index, # should be converted to tensor, but not yet used
                          'sentiment'    : review_sentiment }
                return sample

        def load_SentimentAnalysisDataset(self, dataserver_train, dataserver_test ):   
            self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                        batch_size=self.dl_studio.batch_size,shuffle=True, num_workers=2)
            self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                               batch_size=self.dl_studio.batch_size,shuffle=False, num_workers=2)

        class TEXTnetWithEmbeddings(nn.Module):
            """
            This is embeddings version of the class TEXTnet class shown previously.  Since we
            are using the word2vec embeddings, we know that the input size for each word vector 
            will be a constant value of 300.  Overall, though, this network is meant for semantic 
            classification of variable-length sentiment data.  Based on my limited testing, the 
            performance of this network is very poor because it has no protection against 
            vanishing gradients when used in an RNN.  

            Class Path:  DLStudio -> TextClassificationWithEmbeddings -> TEXTnetWithEmbeddings
            """
            def __init__(self, input_size, hidden_size, output_size):
                super(DLStudio.TextClassificationWithEmbeddings.TEXTnetWithEmbeddings, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = output_size
                self.combined_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
                self.combined_to_middle = nn.Linear(input_size + hidden_size, 100)
                self.middle_to_out = nn.Linear(100, output_size)     
                self.logsoftmax = nn.LogSoftmax(dim=1)

            def forward(self, input, hidden):
                combined = torch.cat((input, hidden), 1)
                hidden = self.combined_to_hidden(combined)
                hidden = torch.tanh(hidden)                     
                out = self.combined_to_middle(combined)
                out = torch.nn.functional.relu(out)
                out = self.middle_to_out(out)
                out = self.logsoftmax(out)
                return out,hidden         

            def init_hidden(self):
                hidden = torch.zeros(1, self.hidden_size)
                return hidden


        class TEXTnetOrder2WithEmbeddings(nn.Module):
            """
            This is an embeddings version of the TEXTnetOrder2 class shown previously.
            With the embeddings, we know that the size the tensor for word will be 300.
            As to how TEXTnetOrder2 differs from TEXTnet, the value of hidden as used at
            each time step also includes its value at the previous time step.  This 
            fact, not directly apparent by the definition of the class shown below, 
            is made possible by the last parameter, cell, in the header of forward().  
            All you can see here, at the end of forward(), is that the value of cell 
            goes through a linear layer and through a sigmoid nonlinearity. By the way, 
            since the sigmoid saturates at 0 and 1, it can act like a switch. Later 
            when I use this class in the training function, you will see the cell
            values being used in such a manner that the hidden state at each time
            step is mixed with the hidden state at the previous time step.

            Class Path:  DLStudio -> TextClassificationWithEmbeddings -> TEXTnetOrder2WithEmbeddings
            """
            def __init__(self, hidden_size, output_size, input_size=300):
                super(DLStudio.TextClassificationWithEmbeddings.TEXTnetOrder2WithEmbeddings, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = output_size
                self.combined_to_hidden = nn.Linear(input_size + 2*hidden_size, hidden_size)
                self.combined_to_middle = nn.Linear(input_size + 2*hidden_size, 100)
                self.middle_to_out = nn.Linear(100, output_size)     
                self.logsoftmax = nn.LogSoftmax(dim=1)
                self.dropout = nn.Dropout(p=0.1)
                # for the cell
                self.linear_for_cell = nn.Linear(hidden_size, hidden_size)

            def forward(self, input, hidden, cell):
                combined = torch.cat((input, hidden, cell), 1)
                hidden = self.combined_to_hidden(combined)
                hidden = torch.tanh(hidden)                     
                out = self.combined_to_middle(combined)
                out = torch.nn.functional.relu(out)
                out = self.dropout(out)
                out = self.middle_to_out(out)
                out = self.logsoftmax(out)
                hidden_clone = hidden.clone()
#                cell = torch.tanh(self.linear_for_cell(hidden_clone))
                cell = torch.sigmoid(self.linear_for_cell(hidden_clone))
                return out,hidden,cell         

            def initialize_cell(self):
                weight = next(self.linear_for_cell.parameters()).data
                cell = weight.new(1, self.hidden_size).zero_()
                return cell

            def init_hidden(self):
                hidden = torch.zeros(1, self.hidden_size)
                return hidden


        class GRUnetWithEmbeddings(nn.Module):
            """
            For this embeddings adapted version of the GRUnet shown earlier, we can assume that
            the 'input_size' for a tensor representing a word is always 300.
            Source: https://blog.floydhub.com/gru-with-pytorch/
            with the only modification that the final output of forward() is now
            routed through LogSoftmax activation. 

            Class Path:  DLStudio -> TextClassificationWithEmbeddings -> GRUnetWithEmbeddings 
            """
            def __init__(self, input_size, hidden_size, output_size, num_layers=1): 
                """
                -- input_size is the size of the tensor for each word in a sequence of words.  If you word2vec
                       embedding, the value of this variable will always be equal to 300.
                -- hidden_size is the size of the hidden state in the RNN
                -- output_size is the size of output of the RNN.  For binary classification of 
                       input text, output_size is 2.
                -- num_layers creates a stack of GRUs
                """
                super(DLStudio.TextClassificationWithEmbeddings.GRUnetWithEmbeddings, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.gru = nn.GRU(input_size, hidden_size, num_layers)
                self.fc = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
                self.logsoftmax = nn.LogSoftmax(dim=1)
                
            def forward(self, x, h):
                out, h = self.gru(x, h)
                out = self.fc(self.relu(out[:,-1]))
                out = self.logsoftmax(out)
                return out, h

            def init_hidden(self):
                weight = next(self.parameters()).data
                #                  num_layers  batch_size    hidden_size
                hidden = weight.new(  2,          1,         self.hidden_size    ).zero_()
                return hidden

        def save_model(self, model):
            "Save the trained model to a disk file"
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)


        def run_code_for_training_with_TEXTnet_word2vec(self, net, display_train_loss=False):        
            filename_for_out = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE = open(filename_for_out, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            ## Note that the TEXTnet and TEXTnetOrder2 both produce LogSoftmax output. So we
            ## use nn.NLLLoss. The combined effect of LogSoftMax and NLLLoss is the same as 
            ## for the CrossEntropyLoss
            criterion = nn.NLLLoss()
            accum_times = []
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            start_time = time.perf_counter()
            training_loss_tally = []
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    hidden = net.init_hidden().to(self.dl_studio.device)              
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    review_tensor = review_tensor.to(self.dl_studio.device)
                    sentiment = sentiment.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    input = torch.zeros(1,review_tensor.shape[2]).to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        input[0,:] = review_tensor[0,k]
                        output, hidden = net(input, hidden)
                    loss = criterion(output, torch.argmax(sentiment,1))
                    running_loss += loss.item()
                    loss.backward(retain_graph=True)        
                    optimizer.step()
                    if i % 200 == 199:    
                        avg_loss = running_loss / float(200)
                        training_loss_tally.append(avg_loss)
                        running_loss = 0.0
                        current_time = time.perf_counter()
                        time_elapsed = current_time-start_time
                        print("[epoch:%d  iter:%4d  elapsed_time: %4d secs]     loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                        accum_times.append(current_time-start_time)
                        FILE.write("%.3f\n" % avg_loss)
                        FILE.flush()
            print("\nFinished Training\n\n")
            self.save_model(net)
            if display_train_loss:
                plt.figure(figsize=(10,5))
                plt.title("Training Loss vs. Iterations")
                plt.plot(training_loss_tally)
                plt.xlabel("iterations")
                plt.ylabel("training loss")
                plt.legend()
                plt.savefig("training_loss.png")
                plt.show()


        def run_code_for_training_with_TEXTnetOrder2_word2vec(self, net, display_train_loss=False):        
            filename_for_out = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE = open(filename_for_out, 'w')
            net = copy.deepcopy(net)
            net.to(self.dl_studio.device)
            ## Note that the TEXTnet and TEXTnetOrder2 both produce LogSoftmax output:
            criterion = nn.NLLLoss()
            accum_times = []
            optimizer = optim.SGD(net.parameters(), 
                                       lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            start_time = time.perf_counter()
            training_loss_tally = []
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    cell_prev = net.initialize_cell().to(self.dl_studio.device)
                    cell_prev_2_prev = net.initialize_cell().to(self.dl_studio.device)
                    hidden = net.init_hidden().to(self.dl_studio.device)              
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    review_tensor = review_tensor.to(self.dl_studio.device)
                    sentiment = sentiment.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    input = torch.zeros(1,review_tensor.shape[2])
                    input = input.to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        input[0,:] = review_tensor[0,k]
                        output, hidden, cell = net(input, hidden, cell_prev_2_prev)
                        if k == 0:
                            cell_prev = cell
                        else:
                            cell_prev_2_prev = cell_prev
                            cell_prev = cell
                    loss = criterion(output, torch.argmax(sentiment,1))
                    running_loss += loss.item()
                    loss.backward()        
                    optimizer.step()
                    if i % 200 == 199:    
                        avg_loss = running_loss / float(200)
                        training_loss_tally.append(avg_loss)
                        current_time = time.perf_counter()
                        time_elapsed = current_time-start_time
                        print("[epoch:%d  iter:%4d  elapsed_time: %4d secs]     loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                        accum_times.append(current_time-start_time)
                        FILE.write("%.3f\n" % avg_loss)
                        FILE.flush()
                        running_loss = 0.0
            print("\nFinished Training\n")
            self.save_model(net)
            if display_train_loss:
                plt.figure(figsize=(10,5))
                plt.title("Training Loss vs. Iterations")
                plt.plot(training_loss_tally)
                plt.xlabel("iterations")
                plt.ylabel("training loss")
                plt.legend()
                plt.savefig("training_loss.png")
                plt.show()


        def run_code_for_training_for_text_classification_with_GRU_word2vec(self, net, display_train_loss=False): 
            filename_for_out = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE = open(filename_for_out, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            ##  Note that the GREnet now produces the LogSoftmax output:
            criterion = nn.NLLLoss()
            accum_times = []
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            training_loss_tally = []
            start_time = time.perf_counter()
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    review_tensor = review_tensor.to(self.dl_studio.device)
                    sentiment = sentiment.to(self.dl_studio.device)
                    ## The following type conversion needed for MSELoss:
                    ##sentiment = sentiment.float()
                    optimizer.zero_grad()
                    hidden = net.init_hidden().to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                    loss = criterion(output, torch.argmax(sentiment, 1))
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    if i % 200 == 199:    
                        avg_loss = running_loss / float(200)
                        training_loss_tally.append(avg_loss)
                        current_time = time.perf_counter()
                        time_elapsed = current_time-start_time
                        print("[epoch:%d  iter:%4d  elapsed_time:%4d secs]     loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                        accum_times.append(current_time-start_time)
                        FILE.write("%.5f\n" % avg_loss)
                        FILE.flush()
                        running_loss = 0.0
            self.save_model(net)
            print("Total Training Time: {}".format(str(sum(accum_times))))
            print("\nFinished Training\n\n")
            if display_train_loss:
                plt.figure(figsize=(10,5))
                plt.title("Training Loss vs. Iterations")
                plt.plot(training_loss_tally)
                plt.xlabel("iterations")
                plt.ylabel("training loss")
                plt.legend()
                plt.savefig("training_loss.png")
                plt.show()


        def run_code_for_testing_with_TEXTnet_word2vec(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            net.to(self.dl_studio.device)
            classification_accuracy = 0.0
            negative_total = 0
            positive_total = 0
            confusion_matrix = torch.zeros(2,2)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    review_tensor = review_tensor.to(self.dl_studio.device)
                    category      = category.to(self.dl_studio.device)
                    sentiment     = sentiment.to(self.dl_studio.device)
                    input = torch.zeros(1,review_tensor.shape[2]).to(self.dl_studio.device)
                    hidden = net.init_hidden().to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        input[0,:] = review_tensor[0,k]
                        output, hidden = net(input, hidden)
                    predicted_idx = torch.argmax(output).item()
                    gt_idx = torch.argmax(sentiment).item()
                    if i % 100 == 99:
                        print("   [i=%4d]    predicted_label=%d       gt_label=%d" % (i+1, predicted_idx,gt_idx))
                    if predicted_idx == gt_idx:
                        classification_accuracy += 1
                    if gt_idx == 0: 
                        negative_total += 1
                    elif gt_idx == 1:
                        positive_total += 1
                    confusion_matrix[gt_idx,predicted_idx] += 1
            print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(i)))
            out_percent = np.zeros((2,2), dtype='float')
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            print("\n\nNumber of positive reviews tested: %d" % positive_total)
            print("\n\nNumber of negative reviews tested: %d" % negative_total)
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                      "
            out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
            print(out_str + "\n")
            for i,label in enumerate(['true negative', 'true positive']):
                out_str = "%12s%%:  " % label
                for j in range(2):
                    out_str +=  "%18s%%" % out_percent[i,j]
                print(out_str)


        def run_code_for_testing_with_TEXTnetOrder2_word2vec(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            net.to(self.dl_studio.device)
            classification_accuracy = 0.0
            negative_total = 0
            positive_total = 0
            confusion_matrix = torch.zeros(2,2)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    cell_prev = net.initialize_cell()
                    cell_prev_2_prev = net.initialize_cell()
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    input = torch.zeros(1,review_tensor.shape[2]).to(self.dl_studio.device)
                    hidden = net.init_hidden().to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        input[0,:] = review_tensor[0,k]
                        output, hidden, cell = net(input, hidden, cell_prev_2_prev)
                        if k == 0:
                            cell_prev = cell
                        else:
                            cell_prev_2_prev = cell_prev
                            cell_prev = cell
                    predicted_idx = torch.argmax(output).item()
                    gt_idx = torch.argmax(sentiment).item()
                    if i % 100 == 99:
                        print("   [i=%4d]    predicted_label=%d       gt_label=%d" % (i+1, predicted_idx,gt_idx))
                    if predicted_idx == gt_idx:
                        classification_accuracy += 1
                    if gt_idx == 0: 
                        negative_total += 1
                    elif gt_idx == 1:
                        positive_total += 1
                    confusion_matrix[gt_idx,predicted_idx] += 1
            print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(i)))
            out_percent = np.zeros((2,2), dtype='float')
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            print("\n\nNumber of positive reviews tested: %d" % positive_total)
            print("\n\nNumber of negative reviews tested: %d" % negative_total)
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                      "
            out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
            print(out_str + "\n")
            for i,label in enumerate(['true negative', 'true positive']):
                out_str = "%12s:  " % label
                for j in range(2):
                    out_str +=  "%18s" % out_percent[i,j]
                print(out_str)


        def run_code_for_testing_text_classification_with_GRU_word2vec(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            classification_accuracy = 0.0
            negative_total = 0
            positive_total = 0
            confusion_matrix = torch.zeros(2,2)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    hidden = net.init_hidden()
                    for k in range(review_tensor.shape[1]):
                        output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                    predicted_idx = torch.argmax(output).item()
                    gt_idx = torch.argmax(sentiment).item()
                    if i % 100 == 99:
                        print("   [i=%d]    predicted_label=%d       gt_label=%d" % (i+1, predicted_idx,gt_idx))
                    if predicted_idx == gt_idx:
                        classification_accuracy += 1
                    if gt_idx == 0: 
                        negative_total += 1
                    elif gt_idx == 1:
                        positive_total += 1
                    confusion_matrix[gt_idx,predicted_idx] += 1
            print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(i)))
            out_percent = np.zeros((2,2), dtype='float')
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            print("\n\nNumber of positive reviews tested: %d" % positive_total)
            print("\n\nNumber of negative reviews tested: %d" % negative_total)
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                      "
            out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
            print(out_str + "\n")
            for i,label in enumerate(['true negative', 'true positive']):
                out_str = "%12s:  " % label
                for j in range(2):
                    out_str +=  "%18s%%" % out_percent[i,j]
                print(out_str)


#_________________________  End of DLStudio Class Definition ___________________________

#______________________________    Test code follows    _________________________________

if __name__ == '__main__': 
    pass
