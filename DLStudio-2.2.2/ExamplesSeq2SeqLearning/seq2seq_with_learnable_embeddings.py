#!/usr/bin/env python

##  seq2seq_with_learnable_embeddings.py

"""
This script is for experimenting with seq2seq using learning embeddings for the 
source language words.

The seq2seq example is based on English-Spanish translation using a dataset 
that you can download from the DLStudio webpage.  The original source of the
dataset and my alterations to it are described in the main DLStudio documentation
page.

I should also mention that the seq2seq example in this script uses the attention 
mechanism proposed originally by Bahdanau, Cho, and Bengio in their original paper 
on the subject.
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
from Seq2SeqLearning import *


#dataroot = "/home/kak/TextDatasets/en_es_corpus/"
#dataroot = "/mnt/cloudNAS3/Avi/TextDatasets/en_es_corpus/"
dataroot = "./data/"

data_archive = "en_es_8_98988.tar.gz"

max_length = 10
hidden_size = 256
embedding_size = 256


dls = DLStudio(
                  dataroot = dataroot,
                  path_saved_model = {"encoder" : "./saved_encoder", "decoder" : "./saved_decoder"},
                  momentum = 0.9,
                  learning_rate =  0.001,  
                  use_gpu = True,
              )

s2s = Seq2SeqLearning.Seq2SeqWithLearnableEmbeddings( 
                                 dl_studio = dls,
                                 dataroot = dataroot,
                                 data_archive = data_archive,
                                 max_length = max_length,
                                 embedding_size = embedding_size,
                                 num_trials = 100000,
      )

encoder = Seq2SeqLearning.Seq2SeqWithLearnableEmbeddings.EncoderRNN(
                        dls,
                        s2s,
                        embedding_size = embedding_size,
                        hidden_size =  hidden_size,
                        max_length = max_length,
         )    


decoder = Seq2SeqLearning.Seq2SeqWithLearnableEmbeddings.DecoderRNN(
                        dls,
                        s2s, 
                        embedding_size = embedding_size,
                        hidden_size = hidden_size,
                        max_length = max_length,
          )


s2s.run_code_for_training_Seq2SeqWithLearnableEmbeddings(encoder, decoder, display_train_loss=True)

import pymsgbox
response = pymsgbox.confirm("Finished training.  Start evaluation?")
if response == "OK": 
    s2s.run_code_for_evaluating_Seq2SeqWithLearnableEmbeddings(encoder, decoder)


