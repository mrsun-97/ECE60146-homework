#!/usr/bin/env python

##  upgrade_model.py

"""
    The literature says that training purely attention based networks 
    is difficult.  For example, the Transformer implementation as 
    described in the original paper "Attention is All You Need" 
    paper by Vaswani et el. requires a warm-up phase during training 
    in which, starting with a very small value, you gradually 
    increase the learning rate and you then bring it down again.

    This script just makes it a bit easy to play with the different 
    learning rates on a ad-hoc basis.  What that means is that you
    first run the main example in this directory

          seq2seq_with_transformer.py

    at one learning rate and you then call 

          upgrade_model.py

    with a different learning rate to see what it would do the
    final results.

    Obviously, after having executed 'seq2seq_with_transformer.py'
    once, you can invoke 'upgrade_model.py' any number of times,
    each time with a different learning rate.
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


dataroot = "/home/kak/TextDatasets/en_es_corpus_xformer/"
#dataroot = "/mnt/cloudNAS3/Avi/TextDatasets/en_es_corpus_xformer/"
#dataroot = "./data/"

data_archive =  "en_es_xformer_8_10000.tar.gz"

num_trials = 1000

max_seq_length = 10
embedding_size = 256
qkv_size = 64

#how_many_basic_encoders = how_many_basic_decoders = num_atten_heads = 4
how_many_basic_encoders = how_many_basic_decoders = num_atten_heads = 2

optimizer_params = {'beta1' : 0.9,  'beta2': 0.98,  'epsilon' : 1e-9}

num_warmup_steps = 4000


dls = DLStudio(
                  dataroot = dataroot,
                  path_saved_model = {"encoder" : "./saved_encoder", "decoder" : "./saved_decoder"},
#                  momentum = 0.9,
#                  learning_rate =  1e-4,  
                  batch_size = 4,
                  use_gpu = True,
              )

xformer = Transformers.TransformerFG( 
                                 dl_studio = dls,
                                 dataroot = dataroot,
                                 data_archive = data_archive,
                                 max_seq_length = max_seq_length,
                                 embedding_size = embedding_size,
                                 num_trials = num_trials,
                                 num_warmup_steps = num_warmup_steps,
                                 optimizer_params = optimizer_params,
         )

master_encoder = Transformers.TransformerFG.MasterEncoder(
                                  dls,
                                  xformer,
                                  how_many_basic_encoders = how_many_basic_encoders,
                                  num_atten_heads = num_atten_heads,
                                  qkv_size = qkv_size,
                 )    


master_decoder = Transformers.TransformerFG.MasterDecoderWithMasking(
                                  dls,
                                  xformer, 
                                  how_many_basic_decoders = how_many_basic_decoders,
                                  num_atten_heads = num_atten_heads,
                                  qkv_size = qkv_size,
                 )

xformer.run_code_for_upgrading_model(dls, master_encoder, master_decoder, display_train_loss=True)

import pymsgbox
response = pymsgbox.confirm("Finished training.  Start evaluation?")
if response == "OK": 
    xformer.run_code_for_evaluating_TransformerFG(master_encoder, master_decoder)



