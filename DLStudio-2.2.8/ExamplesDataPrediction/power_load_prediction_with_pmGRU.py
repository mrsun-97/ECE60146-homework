#!/usr/bin/env python

##  power_load_prediction_with_pmGRU.py


"""
This script demonstrates show how you can call on the functionality of the
DataPrediction class to create a prediction framework for your time-series
data.  A time-series consists of a sequence of observations recorded at
regular intervals.  These could, for example, be the price of a stock share
recorded every hour; the hourly recordings of electrical load at your local
power utility company; the mean average temperature recorded on an annual
basis; and so on.  We want to use the past observations to predict the value
of the next one.  While data prediction has much in common with other forms of
sequence based learning, it presents certain unique challenges of its own and
those are with respect to (1) Data Normalization; (2) Input Data Chunking; and
(3) Multi-dimensional encoding of the "datetime" associated with each
observation in the time-series.

Before you can run this script, you will need to download the dataset

    dataset_for_DataPrediction.tar.gz

from the main DLStudio doc page at Purdue.  Store this data archive in the
ExamplesDataPrediction directory of the DLStudio distribution and execute the
following command in that directory:

    tar zxvf dataset_for_DataPrediction.tar.gz

This command will create a 'dataPred' subdirectory and deposit in it the
data for running this script.
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
from DataPrediction import *

import sys


dataroot  =  "./dataPred/"

dls = DLStudio(                                                                                       
                  dataroot = dataroot,
                  path_saved_model = "./saved_PredModel", 
                  learning_rate = .001,
                  epochs = 5,
                  batch_size = 1024,                                                                     
                  use_gpu = True,                                                                     
              )           

predictor = DataPrediction(
                  dlstudio = dls,
                  input_size = 5,      # means that each entry consists of one observation and 4 values for encoding datetime
                  hidden_size = 256,
                  output_size = 1,     # for the prediction 
                  sequence_length = 90,
                  ngpu = 1,    
              )

model = DataPrediction.pmGRU(predictor)
print("\n\nmodel: ", model)

num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in pmGRU: %d\n" % num_learnable_params)

dataframes = predictor.construct_dataframes_from_datafiles()
dataframes_normalized = predictor.data_normalizer( dataframes )
predictor.construct_sequences_from_data(dataframes_normalized)
dataloader = predictor.set_dataloader()

trained_model = predictor.run_code_for_training_data_predictor(dataloader, model)

print("\n\n\nFinished training.  Starting evaluation on unseen data.\n\n")
predictions, gt_for_predictions =  predictor.run_code_for_evaluation_on_unseen_data(trained_model)  
predictor.display_sample_predictions(predictions, gt_for_predictions)

