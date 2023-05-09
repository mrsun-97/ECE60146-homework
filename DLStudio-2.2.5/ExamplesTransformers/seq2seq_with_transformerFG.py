#!/usr/bin/env python

##  seq2seq_with_transformerFG.py

"""
    This script is for experimenting with TransformerFG.

    For an introduction to TransformerFG, read the large comment block associated
    with the definition of this class in the Transformers co-class of DLStudio.

    Also read the doc block associated with the other transformer class, TransformerPreLN,
    for the difference between TransformerFG and TransformerPreLN.

    To run this example, you will need to have installed at least one of the following
    two English-to-Spanish translation dataset archives:

          en_es_xformer_8_10000.tar.gz

          en_es_xformer_8_90000.tar.gz

    The first consists of 10,000 pairs of English-Spanish sentences and the second
    90,0000 such pairs.

    The maximum number of words in any sentence, English or Spanish, is 8.  When you
    include the sentence delimiter tokens SOS and EOS, that makes for a max length of
    10 for the sentences.
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
from Transformers import *

dataroot = "./data/"
#dataroot = "/home/kak/TextDatasets/en_es_corpus_xformer/"
#dataroot = "/mnt/cloudNAS3/Avi/TextDatasets/en_es_corpus_xformer/"


data_archive =  "en_es_xformer_8_90000.tar.gz"

max_seq_length = 10

embedding_size = 256
#embedding_size = 128


how_many_basic_encoders = how_many_basic_decoders = num_atten_heads = 4
#how_many_basic_encoders = how_many_basic_decoders = num_atten_heads = 2

#optimizer_params = {'beta1' : 0.9,  'beta2': 0.98,  'epsilon' : 1e-9}
optimizer_params = {'beta1' : 0.9,  'beta2': 0.98,  'epsilon' : 1e-6}

num_warmup_steps = 4000



dls = DLStudio(
                  dataroot = dataroot,
                  path_saved_model = {"encoder" : "./saved_encoder", "decoder" : "./saved_decoder"},
                  batch_size = 50,
                  use_gpu = True,
                  epochs = 20,
              )

xformer = Transformers.TransformerFG( 
                                 dl_studio = dls,
                                 dataroot = dataroot,
                                 data_archive = data_archive,
                                 max_seq_length = max_seq_length,
                                 embedding_size = embedding_size,
                                 num_warmup_steps = num_warmup_steps,
                                 optimizer_params = optimizer_params,
                                 save_model_end_of_epoch = True,
         )

master_encoder = Transformers.TransformerFG.MasterEncoder(
                                  dls,
                                  xformer,
                                  how_many_basic_encoders = how_many_basic_encoders,
                                  num_atten_heads = num_atten_heads,
                 )    


master_decoder = Transformers.TransformerFG.MasterDecoderWithMasking(
                                  dls,
                                  xformer, 
                                  how_many_basic_decoders = how_many_basic_decoders,
                                  num_atten_heads = num_atten_heads,
                 )


number_of_learnable_params_in_encoder = sum(p.numel() for p in master_encoder.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the Master Encoder: %d" % number_of_learnable_params_in_encoder)
num_layers_encoder = len(list(master_encoder.parameters()))
print("\nThe number of layers in the Master Encoder: %d\n\n" % num_layers_encoder)

number_of_learnable_params_in_decoder = sum(p.numel() for p in master_decoder.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the Master Decoder: %d" % number_of_learnable_params_in_decoder)
num_layers_decoder = len(list(master_decoder.parameters()))
print("\nThe number of layers in the Master Decoder: %d\n\n" % num_layers_decoder)

xformer.run_code_for_training_TransformerFG(dls, master_encoder, master_decoder, display_train_loss=True)

import pymsgbox
response = pymsgbox.confirm("Finished training.  Start evaluation?")
if response == "OK": 
    xformer.run_code_for_evaluating_TransformerFG(master_encoder, master_decoder)


