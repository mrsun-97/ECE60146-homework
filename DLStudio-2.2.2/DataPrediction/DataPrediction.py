# -*- coding: utf-8 -*-

__version__   = '2.2.2'
__author__    = "Avinash Kak (kak@purdue.edu)"
__date__      = '2022-March-5'
__url__       = 'https://engineering.purdue.edu/kak/distDLS/DLStudio-2.2.2.html'
__copyright__ = "(C) 2022 Avinash Kak. Python Software Foundation."


__doc__ = '''


  You are looking at the DataPrediction co-class file in the DLStudio module.
  For the overall documentation on DLStudio, visit:

           https://engineering.purdue.edu/kak/distDLS/



DATA PREDICTION:

    Let's say you have a sequence of observations recorded at regular
    intervals.  These could, for example, be the price of a stock share
    recorded every hour; the hourly recordings of electrical load at your
    local power utility company; the mean average temperature recorded on an
    annual basis; and so on.  We want to use the past observations to predict
    the value of the next one.  Solving these types of problems is the focus
    of the DataPrediction co-class of DLStudio.

    As a problem, data prediction has much in common with text analytics and
    seq2seq processing, in the sense that the prediction at the next time
    instant must be based on all previous observations in a manner similar to
    what we do in text analytics where the next word is understood taking into
    account all the previous words in a sentence.  However, there are three
    significant differences between purely numerical data prediction problems
    and text-based problems:

    1) Data Normalization: As you know by this time, neural networks require
       that your input data be normalized to the [0,1] interval, assuming it
       consists of non-negative numbers, or the [-1,1] interval otherwise.
       This is dictated by the design of the data processing components made
       available by the commonly used platforms.  For example, a commonly used
       activation function, Sigmoid, saturates out outside the [0,1] interval;
       and the activation function Tanh saturates out outside the [-1,1]
       interval. For multi-channel inputs, with each channel being represented
       by a different axis of the input tensor, such normalization is
       generally carried out separately for each channel.

       When solving a problem like text analytics, after you have normalized
       the input data (which is likely to consist of the numeric embeddings
       for the input words), you can forget about it.  You don't have that
       luxury when solving a data prediction problem.  As you would expect,
       the next value predicted by an algorithm must be at the same scale as
       the original input input data.  This requires that the output of a
       neural-network-based prediction algorithm must be "inverse-normalized".
       And that, in turn, requires remembering the normalization parameters
       used in each channel of the input data.

    2) Input Data Chunking: The notion of a sentence that is important in text
       analytics does not carry over to the data prediction problem.  In
       general, you would want a prediction to be made using ALL of the past
       observations.  You are also likely to believe that the longer the time
       span over which the past observations are available, the greater the
       likelihood that the prediction made for the next time instant will be
       accurate.  When the sequential data available for training a predictor
       is arbitrarily long, as is the case with numerical data in general, you
       would need to decide how to "chunk" the data --- that is, how to
       extract sub-sequences from the data for the purpose of training a
       neural network.

    3) Data Conditioning: Data prediction problems give you data as a time
       series --- meaning data values recorded at regular intervals.  A
       datafile could be as rudimentary as consisting of two comma separated
       columns, one containing the "datetime" of the observation and other the
       actual value for what's being observed.  To illustrate what I am
       saying, let's consider the entries in one of the data files for the
       utility power load prediction problem at:

            https://www.kaggle.com/robikscube/hourly-energy-consumption    

       The dataset consists of over 10-years worth of hourly electric load
       recordings made available by several utilities in the east and the
       midwest of the United States.  Shown below is the start of a datafile
       typical datafile:

             2013-12-31 01:00:00,  1861.0
             2013-12-31 02:00:00,  1835.0
             2013-12-31 03:00:00,  1841.0
             2013-12-31 04:00:00,  1872.0
             2013-12-31 05:00:00,  1934.0
             2013-12-31 06:00:00,  1995.0
             2013-12-31 07:00:00,  2101.0
             ...                   ...
             ...                   ...

       As you can see, the datafile has two comma separated columns: the left
       column is the datetime and the right the electrical load recorded in
       megawatts at that time.  As far as the data is concerned, obviously
       there's not much to be done with the actual load observations in the
       right column since those values are purely one-dimensional.

       However, it is an entirely different matter with regard to the datetime
       in the left column. You see, for the case of predicting the electric
       power load at utility company, the time of the day matters, as does the
       day of the week, what season it is (day of the year), etc.  What that
       means that it would make no sense to encode the datetime of the
       observations as a one-dimensional entity. You would need to encode it
       so that it has multiple degrees of freedom in a learning framework.  In
       a GRU-based solution for the electric load prediction problem by
       Gabriel Loye, the datetime is encoded into four numbers as follows:

         -- Hour of the day with an integer value between 0 and 23
         -- Day of the week as an integer value between 1 and 7
         -- Month of the year as an integer value between 1 and 12.
         -- Day of the year as an integer value between 1 and 365

       Data conditioning here would refer to extracting this four dimensional
       information from the datetime column shown above.  As it turns out, the
       Pandas module comes in handy for that, as demonstrated by Gabriel Loye
       in his code that you can download from:

            https://blog.floydhub.com/gru-with-pytorch/

                            ---------------------

    Now that you understand how the data prediction problem differs from, say,
    the problem of text analytics, it is time for me to state my main goal in
    defining the DataPrediction class in this file.  I actually have two
    goals:

    (a) To deepen your understanding of a GRU.  At this point, your
        understanding of a GRU is likely to be based on calling PyTorch's GRU
        in your own code.  Using a pre-programmed implementation for a GRU
        makes life easy and you also get a piece of highly optimized code that
        you can just call in your own code.  However, with a pre-programmed
        GRU, you are unlikely to get insights into how such an RNN is actually
        implemented.

    (b) To demonstrate how you can use a Recurrent Neural Network (RNN) for
        data prediction taking into account the data normalization, chunking,
        and conditioning issues mentioned earlier.

    To address the first goal above, the DataPrediction class presented in
    this file is based on my pmGRU (Poor Man's GRU).  This GRU is my
    implementation of the "Minimal Gated Unit" GRU variant that was first
    presented by Joel Heck and Fathi Salem in their paper "Simplified Minimal
    Gated Unit Variations for Recurrent Neural Networks".  Its hallmark is
    that it combines the Update and the Reset gates of a regular GRU into a
    single gate called the Forget Gate.  You could say that pmGRU is a
    lightweight version of a regular GRU and its use may therefore lead to a
    slight loss of accuracy in the predictions.  As to whether that loss would
    be acceptable would depend obviously on the application context and the
    needs of the customer.  You will find it educational to compare the
    performance you get with my pmGRU-based implementation with Gabriel Loye's
    implementation that uses PyTorch's GRU for the same dataset.  And after
    you have seen my implementation of pmGRU, you should yourself be able to
    translate the regular GRU gating equations into your own implementation
    --- just as a learning exercise.

    For measuring the accuracy of his data prediction networks, Gabriel Loye
    used the Symmetric Mean Absolute Percentage Error (sMAPE) that is the sum
    of the absolute difference between the predicted and the actual values
    divided by the average of the predicted and the actual values:

                           100                     | P_t   -   GT_t |
            sMAPE    =    -----  \SUM_{t=1}^N  ---------------------------
                            N                    | P_t   +   GT_t | / 2.0

    where N is the number of predictions made in a file.  P_t is the value
    predicted by the network at inference time and GT_t is the ground-truth
    value that actually occurred at that time.  I have used the same metric
    for estimating the accuracy of the prediction results produced by pmGRU.

@endofdocs
'''


from DLStudio import DLStudio

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import random
import sys


class DataPrediction(object):

    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''DataPrediction constructor can only be called with keyword arguments for 
                      the following keywords: dataroot, hidden_size, input_size, output_size, sequence_length,
                      path_saved_model, use_gpu, ngpu, dlstudio, device, and debug''')
        allowed_keys = 'dataroot','path_saved_model','hidden_size','input_size','output_size','sequence_length',\
                       'use_gpu','ngpu','dlstudio','debug'
        keywords_used = kwargs.keys()                                                                 
        for keyword in keywords_used:                                                                 
            if keyword not in allowed_keys:                                                           
                raise SyntaxError(keyword + ":  Wrong keyword used --- check spelling")  
        learning_rate=input_size=hidden_size=output_size=sequence_length=dataroot=path_saved_model=use_gpu=ngpu=debug=None
        if 'ngpu' in kwargs                          :   ngpu  = kwargs.pop('ngpu')           
        if 'dlstudio' in kwargs                      :   dlstudio  = kwargs.pop('dlstudio')
        if 'debug' in kwargs                         :   debug = kwargs.pop('debug')
        if 'input_size' in kwargs                    :   input_size = kwargs.pop('input_size')
        if 'hidden_size' in kwargs                   :   hidden_size = kwargs.pop('hidden_size')
        if 'output_size' in kwargs                   :   output_size = kwargs.pop('output_size')
        if 'sequence_length' in kwargs               :   sequence_length = kwargs.pop('sequence_length')
        if input_size:
            self.input_size = input_size
        if hidden_size:
            self.hidden_size = hidden_size
        if output_size:
            self.output_size = output_size
        if sequence_length:
            self.sequence_length = sequence_length
        if ngpu:
            self.ngpu = ngpu
        if dlstudio:
            self.dls = dlstudio
        self.training_sequences = []             ## Array of sequences extracted from the sequential observations
        self.training_predictions = []           ## The corresponding predictions for each of the training sequences
        self.test_sequences = {}                 ## We store the test sequences in dicts whose key values are the data files
        self.test_gt_predictions = {}            ## 'gt' stands for 'ground truth'. The predictions for the test sequences are
                                                 ##     are similary stored in dicts
        self.predictions_scaling_params = {}     ## This dict stores the scaling params for the test sequences extracted from
                                                 ##     a file.  These params are used to inverse-scale the final predictions
        if debug:
            self.debug = debug
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    
    class pmGRU(nn.Module):
        """
        This GRU implementation is based primarily on a "Minimal Gated" version of a GRU as described in
        "Simplified Minimal Gated Unit Variations for Recurrent Neural Networks" by Joel Heck and Fathi 
        Salem. The Wikipedia page on "Gated_recurrent_unit" has a summary presentation of the equations 
        proposed by Heck and Salem.
        """
        def __init__(self, predictor):
            super(DataPrediction.pmGRU, self).__init__()
            self.input_size = predictor.input_size
            self.hidden_size = predictor.hidden_size
            self.output_size = predictor.output_size
            self.batch_size = predictor.dls.batch_size
            ## for forget gate:
            self.project1 = nn.Sequential( nn.Linear(self.input_size + self.hidden_size, self.hidden_size), nn.Sigmoid() )
            ## for interim out:
            self.project2 = nn.Sequential( nn.Linear( self.input_size + self.hidden_size, self.hidden_size), nn.Tanh() ) 
            ## for final out
            self.project3 = nn.Sequential( nn.Linear( self.hidden_size, self.output_size ), nn.Tanh() )                   
    
        def forward(self, x, h, sequence_end=False):
            combined1 = torch.cat((x, h), 1)
            forget_gate = self.project1(combined1)  
            interim =  forget_gate * h
            combined2  = torch.cat((x, interim), 1)
            output_interim =  self.project2( combined2 )
            output = (1 - forget_gate) * h  +  forget_gate * output_interim
            if sequence_end == False:
                return output, output
            else:
                final_out = self.project3(output)
                return final_out, final_out
    
        def init_hidden(self):
            weight = next(self.parameters()).data                                                
            hidden = weight.new(self.batch_size, self.hidden_size).zero_()
            return hidden
    
    def construct_dataframes_from_datafiles(self):
        """
        The goal is to store contents of each CSV file in a Pandas DataFrame.  A DataFrame is a two-dimensional 
        data structure with rows and columns, like an Excel spreadsheet. Each column is represented by a separate 
        Pandas Series object. A DataFrame object has two axes: “Axis 0” and “Axis 1”.  “Axis 0” points in the 
        direction of increasing row-index values and "Axis 1" points in the direction of increasing column index.
        """
        dataframes = {}
        datafiles = os.listdir( self.dls.dataroot )
        print("\n\n[construct dataframes]  datafiles: ", datafiles)
        print("\n[construct dataframes]  file header and a few rows: ")
        print(pd.read_csv(self.dls.dataroot + datafiles[0]).head())
        for file in datafiles:
            ## Setting 'parse_dates' to '[0]' tells Pandas that datetime is in column indexed 0.
            df = pd.read_csv(self.dls.dataroot + file, parse_dates=[0], encoding='utf-8')    
            print("\n[construct dataframes]  first 10 rows of the df dataframe: ")
            print(df[:10])
            ##  The following four statements create additional columns in the tabular data stored the DataFrame 
            ##  instance "df".  The first statement adds a column with its header set to "hour".  The values in this 
            ##  column will be set to an integer between 0 and 23 depending on the hour at which the electrical 
            ##  load was measured. And so on. I have taken these statements from the code by Gabriel Loye.
            df['hour'] = df.apply(lambda x: x['Datetime'].hour, axis=1)
            df['dayofweek'] = df.apply(lambda x: x['Datetime'].dayofweek, axis=1)
            df['month'] = df.apply(lambda x: x['Datetime'].month, axis=1)
            df['dayofyear'] = df.apply(lambda x: x['Datetime'].dayofyear, axis=1)
            ##  Now sort the rows based on the values in the 'Datetime' column and then drop the 'Datetime' column:
            df = df.sort_values("Datetime").drop("Datetime", axis=1)
            print("\n[construct dataframes]  Shape of the DataFrame object for file %s: " % file, df.shape)  ##  like: (32896, 5) 
            dataframes[file] = df
        return dataframes
    
    def data_normalizer(self, dataframes):
        """
        For neural-network based processing, we want to normalize (meaning to scale) each column of the data in each
        DataFrame object to the [0,1] interval.  The data normalization step changes the value x IN EACH CHANNEL of 
        the input data into

                        x       -     x_min
                       --------------------
                        x_max   -     x_min

        where x_min is the minimum and x_max the maximum of all the values IN THAT CHANNEL. So, logically speaking,
        the data normalization is applied SEPARATELY to each channel. In our case, each channel of the input corresponds 
        to each column of the DataFrame object. This is most easily done first constructing an instance of MinMaxScaler 
        from the "sklearn.preprocessing" module as shown in Line (A) below and calling on its "fit_transform()" method 
        as shown in Line (C).  Using this functionality of "sklearn.preprocessing" saves you from having to write your 
        own code that would calculate the x_min and x_max values for each column of a DataFrame object.

        The data normalization as explained above implies that if we wanted to reverse the scaling for the column 
        that is being subject to data prediction, the x_min and x_max values for that column must be remembered for
        its use at the and of the learning process.  These values are returned by the "fit()" method of a MinMaxScaler
        instance, as shown in Line (D). The MinMaxScaler instance constructed in Line (B) is for extracting the x_min
        and the x_max values for the main observation column that is focus of predictions.

        To illustrate the normalization step with an example, consider the following 3-column (the same thing as 
        3-channel) input data:

                     X   =    [ [1.0    -1.0    2.0]
                                [2.0     0.     0. ]
                                [0.      1.0    -1.] ]

        The min values in each column are:

              X.min(axis=0)    =    [0.0   -1.0   -1.]

        Subtracting the values in each column by its min will turn all values into POSITIVE  NUMBERS:

                X  -  X.min(axis=0)    =   [ [1.   0.0   3.0]
                                             [2.   1.0   1.0]
                                             [0.   2.0   0.0] ]

        Next, we need to apply the scaling multiplier to these numbers.  For each column, the scaling multiplier 
        is
                                              1.0
                X_std        =    ---------------------------------
                                  X.max(axis=0)   -   X.min(axis=0)

                             =   [ 0.5       , 0.5       , 0.33333333]

        The multiplication

               ( X  -  X.min(axis=0) )  *  X_std

        gives a normalized version of the data with each of its columns separately mapped to the [0,1] interval.

        The explanation given so far in this comment block applies to the role of the data_scaler object defined
        Line (A) below.  About the role of a similar data scaler object, gt_predictions_scaler, in Line (B),
        that is for the purpose of remembering the x_min and x_max values for that column of the input data
        that is subject to predictions.  As already stated, we need those values for inverting the data scaling
        step at the output of the prediction network.
        """
        dataframes_normalized = {}
        for file in dataframes:
            df = dataframes[file]
            data_scaler           =   MinMaxScaler()                                                             ## (A)
            gt_predictions_scaler =   MinMaxScaler()                                                             ## (B)
            data = data_scaler.fit_transform(df.values)                                                          ## (C)
            gt_predictions_scaler.fit(df.iloc[:,0].values.reshape(-1,1))                                         ## (D)
            ##  save the MinMaxScaler instance 
            self.predictions_scaling_params[file] = gt_predictions_scaler
            dataframes_normalized[file] = data
        return dataframes_normalized
    
    def construct_sequences_from_data(self, dataframes):
        """
        The purpose of this method is to construct sequences from the electrical-load data in the DataFrame objects
        created for each file by the method "construct_dataframes_from_datafiles()". The length of the sequences 
        extracted on a running-basis from the data is set by the constructor parameter "sequence_length".  
        """
        for file in dataframes:
            num_records_in_file = len(dataframes[file])
            sequence_length = self.sequence_length
            ##  Note that the maximum number of sequences that can be extracted from a file is the number of observations
            ##  in the file minus the sequence length.  If there are, say, 1000 observations in a file and the sequence
            ##  length is 50, you will be able to construct a maximum of 950 sequences from that file.  The shape of each
            ##  sequence will also depend on how the datetime for each observation was encoded.  As currently programmed
            ##  in the method "construct_dataframes_from_datafiles()", the datetime is encoded into 4 values. Therefore,
            ##  for the case of 1000 observations in a file and with sequence length set to 50, the shape of the tensor
            ##  that stores all the sequences will be "950 x 5".
            sequences_from_file = np.zeros((num_records_in_file - sequence_length, sequence_length, dataframes[file].shape[1]))   
            predictions_from_file = np.zeros(num_records_in_file - sequence_length)            
            predictions_from_file = predictions_from_file.reshape(-1, 1)
            for i in range(sequence_length, num_records_in_file):
                sequences_from_file[i - sequence_length] = dataframes[file][i-sequence_length : i]         
                predictions_from_file[i - sequence_length] = dataframes[file][i,0]                  
            ## We save 10% of the sequences for extracted from each data file for the final evaluation of the 
            ## prediction model:
            test_portion = int(0.1*len(sequences_from_file))      
            if len(self.training_sequences) == 0:
                self.training_sequences = sequences_from_file[:-test_portion]                                             
                self.training_predictions = predictions_from_file[:-test_portion]                                         
            else:
                self.training_sequences = np.concatenate((self.training_sequences, sequences_from_file[:-test_portion]))  
                self.training_predictions = np.concatenate((self.training_predictions, predictions_from_file[:-test_portion])) 
            ##  We store the test sequences and their associated labels in the two dicts shown below that are keyed
            ##  to the file names.
            self.test_sequences[file] = (sequences_from_file[-test_portion:])         
            self.test_gt_predictions[file] = (predictions_from_file[-test_portion:])     


    def set_dataloader(self):
        train_data = TensorDataset(torch.from_numpy(self.training_sequences), torch.from_numpy(self.training_predictions)) 
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.dls.batch_size, drop_last=True)      
        return train_loader


    def run_code_for_training_data_predictor(self, train_loader, model):            
        """
        The 'model' parameter of the method is set to an instance of pmGRU.  See the following script

               power_load_prediction_with_pmGRU.py

        in the ExamplesDataPrediction directory of the DLStudio distribution for how to construct an instance
        of pmGRU and how to then supply that instance to this 'run_code_for_training' method.
        """
        device = self.dls.device
        model.to(device)
        learning_rate = self.dls.learning_rate
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print("\n\nStarting Training")
        start_time = time.perf_counter()                                           
        epochs = self.dls.epochs
        loss_tally = []
        for epoch in range(epochs):        
            avg_loss = 0.                                                   
            running_loss = 0.0
            for counter,data in enumerate(train_loader):       
                x,pred = data
                optimizer.zero_grad()
                ##  Shape of x:   (batch_size, sequence_length, input_size)
                ##  For power-load dataset and for the constructor params chosen, the shape of x is  (1024,90,5)              
                pred = pred.to(device).float()
                h = model.init_hidden().data.to(device)
                ##  The following loop steps through each element of the input sequence, one element at a
                ##  time.  Note that x.shape[1] is the sequence_length, which is 90 as set by the constructor
                ##  in the script in the ExamplesDataPrediction directory.
                for load_sample_index in range(x.shape[1]):   
                    input_sample = x[:,load_sample_index,:]
                    input_sample = torch.squeeze(input_sample)
                    input_sample = input_sample.to(device).float()
                    if load_sample_index < x.shape[1] - 1:
                        out, h = model(input_sample, h)      
                    elif load_sample_index == x.shape[1] - 1:
                        out, h = model(input_sample, h, sequence_end=True)      
                loss = criterion(out, pred)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if counter%200 == 199:                           
                    current_time = time.perf_counter()
                    elapsed_time = current_time - start_time                 
                    avg_loss = running_loss / float(200)
                    loss_tally.append(avg_loss)
                    print("\n[epoch:%d/%d  iter=%4d  elapsed_time=%3d secs]     Avg Loss: %.8f" %  (epoch+1, epochs, counter+1, elapsed_time, avg_loss))
                    running_loss = 0.0
        self.save_model(model)
        print("\nFinished Training\n")                                                                                               
        plt.figure(figsize=(10,5))
        plt.title("Loss")
        plt.plot(loss_tally)
        plt.xlabel("Training iterations over 5 epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("loss_vs_iterations_predictor_training.png")
        plt.show()
        return model
    

    def run_code_for_evaluation_on_unseen_data(self, model):
        """
        We now evaluate the quality of the pmGRU based prediction model leanred from the training data.  The 
        evaluation is carried out on the UNSEEN test data.  Recall, when the normalized load data in each file 
        was digested by the method "construct_sequences_from_data()", 
        """
        hidden_size = self.hidden_size
        device = self.dls.device
        with torch.no_grad():
            model.eval()
            model.to(device)
            all_predictions = []              ## For storing all unseen-data predictions made by the network
            all_gt_predictions = []           ## For storing the corresponding ground-truth predictions
            for file in sorted(self.test_sequences):             
                start_time = time.perf_counter()
                print("\n\n\n[eval] file: ", file)
                ## We access the MinMaxScaler instance for "unscaling" the predictions:
                scaler = self.predictions_scaling_params[file] 
                minval, maxval = torch.zeros(1,1), torch.zeros(1,1)
                minval[0,0], maxval[0,0] = scaler.data_min_[0], scaler.data_max_[0]
                minval, maxval = minval.to(device), maxval.to(device)
                testdata = TensorDataset(torch.from_numpy(self.test_sequences[file]), 
                                                                  torch.from_numpy(self.test_gt_predictions[file]))
                testdata_loader = DataLoader(testdata, batch_size=self.dls.batch_size, drop_last=True)      
                for x, gt_predictions in testdata_loader:      ## x is of shape (batch_size, sequence_length, input_size)
                    gt_predictions = gt_predictions.to(device)
                    gt_predictions_unscaled = ( gt_predictions*(maxval - minval) + minval ).reshape(-1)
                    all_gt_predictions +=  gt_predictions_unscaled.tolist()
                    h = model.init_hidden().data.to(device)
                    for load_sample_index in range(x.shape[1]):
                        input_sample = x[:,load_sample_index,:]
                        input_sample = torch.squeeze(input_sample)
                        if load_sample_index < x.shape[1] - 1:
                            out, h = model(input_sample.to(device).float(), h)      
                        elif load_sample_index == x.shape[1] - 1:
                            out, h = model(input_sample.to(device).float(), h, sequence_end=True)      
                    out_unscaled = ( out*(maxval - minval) + minval ).reshape(-1)
                    all_predictions +=  out_unscaled.tolist()
                current_time = time.perf_counter()
                evaluation_time = current_time - start_time                 
                print("evaluation time taken: ", evaluation_time)
            print("\ntotal number of predictions made by the network for evaluation: ", len(all_predictions))
            print("\ntotal number of all ground-truth predictions: ", len(all_gt_predictions))
            print("\nfirst 100 predictions made by network: ", all_predictions[:100])
            print("\nfirst 100 ground-truth predictions:     ", all_gt_predictions[:100])
            """
            Following Gabriel Loye, I'll use Symmetric Mean Absolute Percentage Error (sMAPE) to evaluate prediction 
            accuracy. sMAPE is "the sum of the absolute difference between the predicted and actual values divided 
            by the average of the predicted and actual value, therefore giving a percentage measuring the amount of error." 
            Here is the formula for sMAPE:

                               100                     | P_t   -   GT_t |
                    sMAPE  =  -----  \SUM_{t=1}^n  ---------------------------
                                n                    | P_t   +   GT_t | / 2.0
    
            where n is the number of predictions made in a file.  Note P_t is the value predicted by the network at 
            inference time and GT_t is the ground-truth load value that actually occurred at that time.
            """
            sMAPE = 0
            for i in range(len(all_predictions)):
                sMAPE += np.mean( abs(all_predictions[i] - all_gt_predictions[i]) / (all_gt_predictions[i] + all_predictions[i])/2)/len(all_predictions)
            print("\n\nsMAPE: %.8f" % (sMAPE*100))
            return all_predictions, all_gt_predictions
    
    
    def display_sample_predictions(self, predictions, predictions_gt):
        """
        The two arguments 'predictions' and the corresponding 'predictions_gt' are for solely the unseen test data.
        """
        plt.figure(figsize=(14,10))
        plt.subplot(2,2,1)
        plt.plot(predictions[:100], "-o", color="g", label="Predicted")
        plt.plot(predictions_gt[:100], color="b", label="Actual")
        plt.ylabel('Energy Consumption (MW)')
        plt.legend()
        
        plt.subplot(2,2,2)
        plt.plot(predictions[30000-50:30000], "-o", color="g", label="Predicted")
        plt.plot(predictions_gt[30000-50:30000], color="b", label="Actual")
        plt.ylabel('Energy Consumption (MW)')
        plt.legend()
        
        plt.subplot(2,2,3)
        plt.plot(predictions[60000-50:60000], "-o", color="g", label="Predicted")
        plt.plot(predictions_gt[60000-50:60000], color="b", label="Actual")
        plt.ylabel('Energy Consumption (MW)')
        plt.legend()
        
        plt.subplot(2,2,4)
        plt.plot(predictions[90000-50:90000], "-o", color="g", label="Predicted")
        plt.plot(predictions_gt[90000-50:90000], color="b", label="Actual")
        plt.ylabel('Energy Consumption (MW)')
        plt.legend()
        
        plt.show()


    def save_model(self, model):
        '''
        Save the trained prediction network to a disk file
        '''
        torch.save(model.state_dict(), self.dls.path_saved_model)

        
#_________________________  End of DataPrediciton Class Definition ___________________________

#_________________________________    Test code follows    ___________________________________

if __name__ == '__main__': 
    pass
