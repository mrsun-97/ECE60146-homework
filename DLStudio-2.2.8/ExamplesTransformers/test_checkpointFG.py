#!/usr/bin/env python

##  test_checkpointFG.py

"""
    Call syntax example:

            python3  test_checkpointFG.py  checkpoints_with_masking_FG  19

    where 19 is the ending suffix on the encoder and the decoder filenames in the "checkpoints_with_masking_FG"  directory.
   
    Before you invoke this script, make sure that the following paramters are set correctly and correspond to what 
    are being used for training:

       embedding_size   =    ??????

       how_many_basic_encoders   =    how_many_basic_decoders   =    num_atten_heads    =  ????????    [choose 2 or 4]       

"""

import random
import numpy
import torch
import os, sys

if len(sys.argv) != 3:
    sys.stderr.write("\n\nUSAGE: python  %s  checkpoints_directory  N    (where 'N' is the int suffix on models in relevant checkpoints directory)\n\nEXAMPLE:  python  %s  checkpoints_with_masking   19\n\n\n" %  (sys.argv[0], sys.argv[0]) )
    sys.exit(1)   


checkpoint_dir =  sys.argv[1]
checkpoint_index = int(sys.argv[2])


seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


from DLStudio import *
from Transformers import *

dataroot = "./data/"
#dataroot = "/home/kak/TextDatasets/en_es_corpus_xformer/"
#dataroot = "/mnt/cloudNAS3/Avi/TextDatasets/en_es_corpus_xformer/"

data_archive =  "en_es_xformer_8_90000.tar.gz"

max_seq_length = 10

#embedding_size = 256                               ##  for running in RVL Cloud
#embedding_size = 128
embedding_size = 64                                 ##  for running in laptop

#how_many_basic_encoders = how_many_basic_decoders = num_atten_heads = 4        ## for running in RVL Cloud
how_many_basic_encoders = how_many_basic_decoders = num_atten_heads = 2         ## for running in laptop

masking = True

dls = DLStudio(
                  dataroot = dataroot,
                  batch_size = 50,
                  use_gpu = True,
              )

xformer = Transformers.TransformerFG( 
                                 dl_studio = dls,
                                 dataroot = dataroot,
                                 save_checkpoints = True,
                                 data_archive = data_archive,
                                 max_seq_length = max_seq_length,
                                 embedding_size = embedding_size,
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
                                  masking = masking
                 )


xformer.run_code_for_evaluating_checkpoint(master_encoder, master_decoder, checkpoint_dir, checkpoint_index, 
                            'myoutput_' +  str(checkpoint_index) + '.txt')


