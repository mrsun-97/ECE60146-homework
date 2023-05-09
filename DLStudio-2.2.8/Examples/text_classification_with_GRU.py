#!/usr/bin/env python

##  text_classification_with_GRU.py

"""
This script is an attempt at solving the sentiment classification problem
with a GRU based recurrence to get around the problem of vanishing gradients
that are common to neural networks with feedback.
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

#dataroot = "/home/kak/TextDatasets/sentiment_dataset/"
dataroot = "./data/TextDatasets/sentiment_dataset/"

dataset_archive_train = "sentiment_dataset_train_40.tar.gz"
#dataset_archive_train = "sentiment_dataset_train_200.tar.gz"

dataset_archive_test =  "sentiment_dataset_test_40.tar.gz"
#dataset_archive_test = "sentiment_dataset_test_200.tar.gz"


dls = DLStudio(
                  dataroot = dataroot,
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
                  learning_rate =  1e-5,
                  epochs = 1,
                  batch_size = 1,
                  classes = ('negative','positive'),
                  use_gpu = True,
              )


text_cl = DLStudio.TextClassification( dl_studio = dls )
dataserver_train = DLStudio.TextClassification.SentimentAnalysisDataset(
                                 train_or_test = 'train',
                                 dl_studio = dls,
                                 dataset_file = dataset_archive_train,
                  )
dataserver_test = DLStudio.TextClassification.SentimentAnalysisDataset(
                                 train_or_test = 'test',
                                 dl_studio = dls,
                                 dataset_file = dataset_archive_test,
                  )
text_cl.dataserver_train = dataserver_train
text_cl.dataserver_test = dataserver_test

text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)

vocab_size = dataserver_train.get_vocab_size()

model = text_cl.GRUnet(vocab_size, hidden_size=512, output_size=2, num_layers=2)

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

num_layers = len(list(model.parameters()))

print("\n\nThe number of layers in the model: %d" % num_layers)
print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
print("\nThe size of the vocabulary (which is also the size of the one-hot vecs for words): %d\n\n" % vocab_size)

## TRAINING:
text_cl.run_code_for_training_for_text_classification_with_GRU(model, display_train_loss=True)

## TESTING:
import pymsgbox
response = pymsgbox.confirm("Finished training.  Start testing on unseen data?")
if response == "OK": 
    text_cl.run_code_for_testing_text_classification_with_GRU(model)


