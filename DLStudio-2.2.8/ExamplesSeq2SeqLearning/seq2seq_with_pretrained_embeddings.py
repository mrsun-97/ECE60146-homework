#!/usr/bin/env python

##  seq2seq_with_pretrained_embeddings.py


"""This script is for experimenting with seq2seq using pre-trained word2vec 
embeddings for the source language words. Although, as supplied, the script
uses the word2vec embeddings, you should be able to get it to work with 
the Fasttext embeddings also.

At the moment, I am using the pre-trained embeddings for only the source
language sentence because of the constraints on the fast memory that come into
existence when you use pre-trained embeddings for multiple languages
simultaneously.  My original plan was to use word2vec embeddings for the source
language English and the Fasttext embeddings for the target language Spanish.
The pre-trained word2vec embeddings for English occupy nearly 4GB of RAM and the
pre-trained Fasttext embeddings another 8GB.  The two objects co-residing in the
fast memory brings down to heel a 32GB machine.

The seq2seq example is based on English-Spanish translation using a dataset that
you can download from the DLStudio webpage.  The original source of the dataset
and my alterations to it are described in the main DLStudio documentation page.

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

#path_to_saved_embeddings_en = "/home/kak/TextDatasets/word2vec/"
path_to_saved_embeddings_en = "./data/TextDatasets/word2vec/"

max_length = 10
hidden_size = 300
embedding_size = 300


dls = DLStudio(
                  dataroot = dataroot,
                  path_saved_model = {"encoder" : "./saved_encoder", "decoder" : "./saved_decoder"},
                  momentum = 0.9,
                  learning_rate =  .001,  
                  use_gpu = True,
              )

s2s = Seq2SeqLearning.Seq2SeqWithPretrainedEmbeddings(
                                 dl_studio = dls,
                                 dataroot = dataroot,
                                 data_archive = data_archive,
                                 path_to_saved_embeddings_en = path_to_saved_embeddings_en,
                                 embeddings_type = 'word2vec',
                                 max_length = max_length,
                                 embedding_size = embedding_size,
                                 num_trials = 100000,
      )


encoder = Seq2SeqLearning.Seq2SeqWithPretrainedEmbeddings.EncoderRNN(
                        dls,
                        s2s,
                        embedding_size = embedding_size,
                        hidden_size =  hidden_size,
                        max_length = max_length,
         )


decoder = Seq2SeqLearning.Seq2SeqWithPretrainedEmbeddings.DecoderRNN(
                        dls,
                        s2s,
                        embedding_size = embedding_size,
                        hidden_size = hidden_size,
                        max_length = max_length,
          )


s2s.run_code_for_training_Seq2SeqWithPretrainedEmbeddings(encoder, decoder, display_train_loss=True)

import pymsgbox
response = pymsgbox.confirm("Finished training.  Start evaluation?")
if response == "OK":
    s2s.run_code_for_evaluating_Seq2SeqWithPretrainedEmbeddings(encoder, decoder)
