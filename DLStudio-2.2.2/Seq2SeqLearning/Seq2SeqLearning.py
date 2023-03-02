# -*- coding: utf-8 -*-

__version__   = '2.2.2'
__author__    = "Avinash Kak (kak@purdue.edu)"
__date__      = '2022-March-5'
__url__       = 'https://engineering.purdue.edu/kak/distDLS/DLStudio-2.2.2.html'
__copyright__ = "(C) 2022 Avinash Kak. Python Software Foundation."


__doc__ = '''


  You are looking at the Seq2SeqLearning co-class file in the DLStudio module.
  For the overall documentation on DLStudio, visit:

           https://engineering.purdue.edu/kak/distDLS/



SEQUENCE-TO-SEQUENCE LEARNING WITH ATTENTION: 

    As mentioned in the main doc page for DLStduio, sequence-to-sequence
    learning (seq2seq) is about predicting an outcome sequence from a
    causation sequence, or, said another way, a target sequence from a source
    sequence.  Automatic machine translation is probably one of the most
    popular applications of seq2seq.  DLStudio uses English-to-Spanish
    translation to illustrate the programming idioms and the PyTorch
    structures you need for seq2seq.  

    In version 2.0.9, where I first introduced seq2seq in DLStudio, I placed
    the code inside the main DLStudio class.  However, considering that
    seq2seq is an active area of research and development unto itself, it
    seems more prudent to give seq2seq its own home.  Therefore, I have moved
    that code to a co-class of DLStudio --- the class you are looking at as
    you are reading this doc page.  This class, Seq2SeqLearning, resides at
    the top-level of the distro.  That makes Seq2SeqLearning a co-class of the
    main DLStudio class.

    In Version 2.1.0, Seq2SeqLearning contains the following two inner
    classes for illustrating seq2seq:
    
        1.  Seq2SeqWithLearnableEmbeddings

        2.  Seq2SeqWithPretrainedEmbeddings

    As their names imply, the first is for seq2seq with learnable
    embeddings and the second for seq2seq with pre-trained embeddings like
    word2vec or fasttext.  The seq2seq implementations include the
    attention mechanism based on my understanding of the original paper on
    the subject by Bahdanau, Cho, and Bengio.

    As mentioned in the first para above, the specific example of seq2seq
    addressed in my implementation code is translation from English to
    Spanish. (I chose this example because learning and keeping up with
    Spanish is one of my hobbies.)  In the Seq2SeqWithLearnableEmbeddings
    class, the learning framework learns the best embedding vectors to use
    for the two languages involved. On the other hand, in the
    Seq2SeqWithPretrainedEmbeddings class, I use the word2vec embeddings
    provided by Google for the source language.  As to why I use the
    pre-training embeddings for just the source language is explained in the
    main comment doc associated with the class
    Seq2SeqWithPretrainedEmbeddings.

    To repeat what I mentioned on the main DLStudio doc page, any modern
    attempt at seq2seq must include attention.  This is done by incorporating
    a separate Attention network in the Encoder-Decoder framework needed for
    seq2seq learning.  The goal of the attention network is to modify the
    current hidden state in the decoder using the attention units produced
    previously by the encoder for the source language sentence.  

    As mentioned above, the main Attention model I have used is based on my 
    understanding of the attention mechanism proposed by Bahdanau, Cho, and 
    Bengio. You will see this attention code in a class named Attention_BCB 
    of Seq2SeqWithLearnableEmbeddings.  I have also provided another 
    attention class named Attention_SR that is my implementation of the 
    attention mechanism in the very popular NLP tutorial by Sean Robertson 
    at the PyTorch website.  The URLs to both these attention mechanisms 
    are in my Week 14 lecture material on deep learning at Purdue.

    The following two scripts in the ExamplesSeq2SeqLearning directory of the
    distribution are your main entry points for experimenting with the
    seq2seq code:

        1.  seq2seq_with_learnable_embeddings.py

        2.  seq2seq_with_pretrained_embeddings.py
    
    With the first script, the overall network will learn on its own the best
    embeddings to use for representing the words in the two languages.  And,
    with the second script, the pre-trained word2vec embeddings from Google
    are used for the source language while the system learns the embeddings
    for the target language.

@endofdocs
'''


from DLStudio import DLStudio

import sys,os,os.path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import glob                                                                                                           
import gzip
import pickle

#___________________________________  Seq2SeqLearning Class Definition  ___________________________________


class Seq2SeqLearning(object):

    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''Seq2SeqLearning constructor can only be called with keyword arguments for 
                      the following keywords: learning_rate, momentum, dataroot, 
                      path_saved_model, use_gpu, ngpu, dlstudio, device, beta1, and debug''')
        allowed_keys = 'dataroot','path_saved_model','momentum','learning_rate', \
                       'classes','use_gpu','ngpu','dlstudio', 'beta1','debug'
        keywords_used = kwargs.keys()                                                                 
        for keyword in keywords_used:                                                                 
            if keyword not in allowed_keys:                                                           
                raise SyntaxError(keyword + ":  Wrong keyword used --- check spelling")  
        learning_rate = momentum = dataroot =  path_saved_model = classes = use_gpu = ngpu = beta1 = debug = None
        if 'ngpu' in kwargs                          :   ngpu  = kwargs.pop('ngpu')           
        if 'dlstudio' in kwargs                      :   dlstudio  = kwargs.pop('dlstudio')
        if 'debug' in kwargs                         :   debug = kwargs.pop('debug')
        if ngpu:
            self.ngpu = ngpu
        if dlstudio:
            self.dlstudio = dlstudio
        if beta1:
            self.beta1 = beta1
        if debug:
            self.debug = debug
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    ###%%%
    ########################################################################################
    ############  Start Definition of Inner Class Seq2SeqWithLearnableEmbeddings  ##########

    class Seq2SeqWithLearnableEmbeddings(nn.Module):             
        """
        As the name implies, sequence-to-sequence (Seq2Seq) learning is about predicting an outcome 
        sequence from a causation sequence, or, said another way, a target sequence from a source 
        sequence.  Automatic machine translation is probably one of the most popular application 
        of Seq2Seq learning.  Since deep learning algorithms can only deal with numerical data,
        an important issue related to Seq2Seq for machine translation is representing the purely
        symbolic entities (such as words) involved with numbers. This is the same issue that was
        addressed in the TextClassification class of DLStudio.  As mentioned there, we have the 
        following choices: 

                1.  use one-hot vectors for the words

                2.  learning the embeddings directly from the training data.

                3.  use pre-trained embedding vectors for the words (as provided by word2vec 
                    and fasttext)

        As previously mentioned in the context of text classification, using one-hot vectors 
        directly is out of the question.  So that leaves us with just two options: learning the
        embeddings directly from the training data and using pre-trained embeddings. 

        The goal of this class, Seq2SeqWithLearnableEmbeddings, is to illustrate the basic 
        notions of Seq2Seq learning with learnable embeddings for the words in a vocabulary.                
        I'll use the problem of English-to-Spanish translation as a case study for the code
        shown in this class. 

        Basic to any modern implementation of Seq2Seq learning is the notion of attention.
        In general, the different grammatical units in a source-language sentence will not
        align with the corresponding units in a translation of the same sentence in the
        target language.  Consider the following English-Spanish phrasal pair:

                    the cabin roof

                    el techo de la cabaña 
        
        The word "techo" in Spanish means "roof". A word-for-word translation of the English
        phase would lead to "la cabaña techo" which is unlikely to be understood by a native
        speaker of the Spanish language.  The goal of attention is for a seq2seq framework
        to learn how to align the different parts of a pair of sentences in two different
        languages.  The attention models I will use here are explained in the slides for the
        seq2seq lecture at the deep-learning course website at Purdue.

        About the dataset I'll be using to demonstrate seq2seq, version 2.1.0 of DLStudio 
        comes with a data archive named en_es_corpus that contains a large number of 
        English-Spanish sentence pairs. This archive is a lightly curated version of the
        main dataset provided at

                 http://www.manythings.org/anki/

        The data at the above website is from the sentences_detailed.csv file at tatoeba.org:

            http://tatoeba.org/files/downloads/sentences_detailed.csv 

        The curated data archive that you can download from the DLStudio website includes the
        copyright notice from tatoeba.org.  

        My alteration to the original dataset consists mainly of expanding the contractions 
        like "it's", "I'm", "don't", "didn't", "you'll", etc., into their expansions "it is", 
        "i am", "do not", "did not", "you will", etc. The English/Spanish dataset as provided 
        at the above URL contains 417 such unique contractions.  Another alteration I made to 
        the original data archive is to surround each sentence in both English and Spanish 
        by the "SOS" and "EOS" tokens, with the former standing for "Start of Sentence" and 
        the latter for "End of Sentence".

        I have used the following convention for naming data archives at the DLStudio website:

                        en_es_N_M.tar.gz

        where N specifies the maximum number of words in the sentences in the archive and M is
        the total number sentence pairs available.  For example, the name of one of the archives
        is:
                        en_es_8_98988.tar.gz

        This archive contains a total of 98988 sentence pairs, with no sentence exceeds 8 words
        in length.

        class path:  Seq2SeqLearning  ->  Seq2SeqWithLearnableEmbeddings
        """
        def __init__(self, dl_studio, dataroot, data_archive, max_length, embedding_size, num_trials):
            super(Seq2SeqLearning.Seq2SeqWithLearnableEmbeddings, self).__init__()
            self.dl_studio = dl_studio
            self.dataroot  = dataroot
            self.data_archive = data_archive
            self.max_length = max_length
            self.embedding_size = embedding_size
            self.num_trials = num_trials
            f = gzip.open(dataroot + data_archive, 'rb')
            dataset = f.read()
            dataset,vocab_en,vocab_es = pickle.loads(dataset, encoding='latin1')
            self.dataset = dataset
            self.vocab_en = vocab_en
            self.vocab_es = vocab_es
            self.vocab_en_size = len(vocab_en)          #  includes the SOS and EOS tokens
            self.vocab_es_size = len(vocab_es)          #  includes the SOS and EOS tokens
            print("\n\nSize of the English vocab in the dataset: ", self.vocab_en_size)
            print("\nSize of the Spanish vocab in the dataset: ", self.vocab_es_size)
            self.debug = False
            if self.debug:
                print("\n\nFirst 100 elements of English vocab: ", vocab_en[:100])
                print("\n\nFirst 100 elements of Spanish vocab: ", vocab_es[:100])
            # The first two elements of both vocab_en and vocab_es are the SOS and EOS tokens
            # So the index position for SOS is 0 and for EOS is 1.
            self.en_vocab_dict = { vocab_en[i] : i  for i in range(self.vocab_en_size) }  
            self.es_vocab_dict = { vocab_es[i] : i  for i in range(self.vocab_es_size) }
            self.es_index_2_word =   { i : vocab_es[i] for i in range(self.vocab_es_size) }
            self.training_corpus = dataset

        def sentence_to_tensor(self, sentence, lang):
            """
            If there are N words in a sentence (recall each sentence starts with the 'SOS' token 
            and ends with the 'EOS' token; N is inclusive of these two marker tokens), the tensor
            produced by this function will be of shape [N,1].  To illustrate with an example, 
            consider the following English sentence from the dataset:

                          SOS they live near the school EOS

            Including the two marker tokens, this source sentence has 7 words in it.  The contract
            of the sentence_to_tensor() method is to return the following tensor of shape 
            "torch.Size([7, 1])" for such a sentence:

                         tensor([[    0],
                                 [10051],
                                 [ 5857],
                                 [ 6541],
                                 [10027],
                                 [ 8572],
                                 [    1]])

            in which each integer is the index of the corresponding word in the sorted vocabulary for
            all the English sentences in the corpus.  Note that we manually insert the tokens 'SOS'
            and 'EOS' in the sorted vocabulary.  That's why the first and the last entries in the
            tensor shown above are 0 and 1. For the other integers in the tensor, obviously, 10051 
            must be the index of the word 'they' in the sorted vocabulaty; 5857 must be the index 
            for the word 'live'; and so on.

            During training, similar tensors are constructed for the Spanish sentences. The integer
            indexes in those tensors serve as targets in the nn.NLLLoss based loss function.
            """
            list_of_embeddings = []
            words = sentence.split(' ')
            sentence_tensor = torch.zeros(len(words), 1, dtype=torch.long)
            if lang == "en":
                for i,word in enumerate(words):
                    sentence_tensor[i] = self.en_vocab_dict[word]    
            elif lang == "es":
                for i,word in enumerate(words):
                    sentence_tensor[i] = self.es_vocab_dict[word]    
            return sentence_tensor


        class EncoderRNN(nn.Module):
            """
            First recall from my lecture on RNN that, ordinarily, the main job of an RNN is 
            to create a fixed-size representation of a variable sized input sequence.  Consider
            the case when the size of the vector embeddings for the words is 256 and an input
            sentence consists of 10 words (including the 'SOS' and 'EOS' tokens). In this case, 
            the source sentence would be represented by a tensor of shape [10,256]. An 
            ordinary RNN would step through this sequence one word at a time while producing 
            an output element and a hidden state at each step.  If we were to actually use 
            an ordinary RNN, we would have no particular use for the output of the RNN, our 
            interest would be solely in the final value of the hidden state, which we would 
            feed into the decoder for generating the target sentence.

            In what you see below for the encoder, we do not use an ordinary RNN. On the other
            hand, we use a GRU (a gated RNN) that I presented in my Week 12 lecture.  As 
            discussed in the last section of my Week 12 slides, a GRU has some pretty "quirky" 
            properties that depend on the choices made for the constructor parameters and the 
            shape of the input data.  Let's again say we have an input sequence consisting of 
            10 words (including the two tokens), the GRU is going to see a tensor of shape 
            (10,1,256) at its input.  Since the dimensionality of the first axis of the input
            tensor is 10, which is greater than 1, the GRU will step through the input sequence 
            on its own.  In such a case, at its output, the GRU will emit the time evolution of 
            the hidden state and, as its hidden, it will show just the final value of the hidden 
            state.  (Obviously, the last value in the sequence of hiddens emitted at
            the GRU output will be the same as the value shown for the hidden.)  So if the 
            input is a sequence of shape (10,1,256), the output will also be a sequence of
            shape (10,1,256) assuming that we are using 256 for representing both each element 
            of the input sequence and the hidden state in the GRU.

            I will now bring in one more complication: setting the 'bidirectional' parameter of
            the GRU constructor in line (A) to True. This creates a bi-directional RNN that scans
            a sentence in both the forward (left-to-right) direction and the backward 
            (right-to-left) directions. Now each element of the time evolution of the hidden 
            state that is emitted at the GRU output is a concatenation of the hidden states 
            in the forward direction and the hidden states in the backward direction.  So for
            the case of a 10-word input sentence, the GRU output will emit a hidden sequence 
            of shape (10,1,512), with each 512-sized hidden being a concatenation of the 
            256 values for the forward direction and the 256 values for the backward direction.
            We refer to each such 512-sized value as an "attention unit".  It is so called 
            because it characterizes the local context in an input sentence at each of its
            words taking into account both the words that come before and the words that come 
            after.  As you will see later, the main job of the Attention network is to learn
            how much to draw from each attention unit for the production of each output word.

            One additional factor that is highly relevant to the action of the EncoderRNN
            defined below: the max_length parameter in line (B).  This parameter plays an 
            important role in the calculation of the attention weights that will eventually be
            needed by the DecoderRNN for producing the target sequence.  Attention weights
            tell us how much contribution each attention unit as defined above makes to
            production of each output word.

            With regard to what the encoder returns, both 'output' and 'hidden' are critical
            to the operation of the decoder as you will see later.  As explained, 'output' 
            is the time-evolution of the hidden in the GRU and 'hidden' is the final value
            of the encoder hidden state.  The former is needed for calculating the attention
            weights and the latter becomes the initial hidden for the decoder.

            ClassPath:  Seq2SeqLearning -> Seq2SeqWithLearnableEmbeddings  ->  EncoderRNN
            """
            def __init__(self, dls, s2s, embedding_size, hidden_size, max_length):
                super(Seq2SeqLearning.Seq2SeqWithLearnableEmbeddings.EncoderRNN, self).__init__()
                self.dl_studio = dls
                self.source_vocab_size = s2s.vocab_en_size
                self.embedding_size = embedding_size                                             
                self.hidden_size = hidden_size                                                   
                self.max_length = max_length                                                     
                self.embed = nn.Embedding(self.source_vocab_size, embedding_size)                
                self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True)                  ## (A)
        
            def forward(self, sentence_tensor, hidden):
                word_embeddings = torch.zeros(self.max_length, 1, 
                                       self.hidden_size).float().to(self.dl_studio.device)          ## (B)
                for i in range(sentence_tensor.shape[0]):                                       
                    word_embeddings[i] = self.embed(sentence_tensor[i].view(1, 1, -1))          
                output, hidden = self.gru(word_embeddings, hidden)                              
                return output, hidden                                                           
        
            def initHidden(self):
                return torch.zeros(2, 1, self.hidden_size).float().to(self.dl_studio.device)


        class DecoderRNN(nn.Module):
            """
            An ordinary decoder would take the final value for the hidden state as emitted by the 
            encoder and try to produce a target sequence. During training, as the decoder goes
            from one step to the next, its output would be a prediction for the next word in
            the target sequence that you would compare with the ground-truth target word at that
            position.  You would add the loss estimated from such a comparison to the 
            accumulating value of the total loss associated with a sentence.  In addition to 
            using the ground-truth target word in this manner, you would also feed it as input 
            into the decoder network where it would be used along with the evolving hidden for 
            predicting the next output word.  On the other hand, during evaluation, each output 
            word produced by the decoder would become its input at the next step.  In the past, 
            such a straightforward application of the decoder logic has only worked for very 
            short input and output sequences.

            The implementation for the decoder shown below also factors in the attention 
            weights returned by a separate Attention network that is called in line (E). The
            goal of the attention network is to modify the current hidden state in the 
            DecoderRNN taking into account all the attention units produced previously by the
            EncoderRNN.

            Note also that while we used the EncoderRNN in the bi-directional mode, the 
            DecoderRNN is being made to operate in the more traditional mode of emitting one 
            output word at a time.

            I should also draw your attention to the statements in line (C) and the commented-out
            line (D).  The call in line (C) is to the constructor of the attention class named
            Attention_BCB and the one in line (D) is to another similar class named Attention_SR.
            As mentioned in the doc sections of those classes, Attention_BCB is based on my 
            understanding the attention mechanism proposed by Bahdanau, Cho, and Bengio. The 
            other attention class, Attention_SR, is based on Sean Robertson's implementation of
            attention in his very popular NLP tutorial at the PyTorch website.

            Finally, let's talk about what the input to and the output of the DecoderRNN look
            like: The input to the decoder is specified by the first argument in line (G) and
            output by the value obtained in line (I).  Both of these must be word index values,
            that is, the integers that correspond to the word positions in the sorted vocabulary
            list for the target language.  As was done for the EncoderRNN, the input word is
            mapped to its embedding produced by the nn.Embedding layer and then supplied to the
            Attention class in line (H) that returns an "attentioned" version of the current 
            decoder hidden state.  The output returned by the GRU in line (I) is first sent 
            through a linear layer, self.out, in line (I) that maps it into a vector whose size
            equals that of the target vocabulary size.

            A most interesting thing about the the Decoder 'output' is that it is kind-of wasted
            during training.  During the evaluation phase, we apply torch.max() to the output
            of the decoder to find the integer index for the emitted output word.  Subsequently,
            this integer index becomes the input to the decoder for the production of the next
            output word.  However, during training, since the next input to the decoder will
            be the next word from the target sequence, we have no use for the current decoder
            output.

            ClassPath:  Seq2SeqLearning -> Seq2SeqWithLearnableEmbeddings  ->  DecoderRNN
            """
            def __init__(self, dls, s2s, embedding_size, hidden_size, max_length):
                super(Seq2SeqLearning.Seq2SeqWithLearnableEmbeddings.DecoderRNN, self).__init__()
                self.hidden_size = hidden_size
                self.target_vocab_size = s2s.vocab_es_size
                self.max_length = max_length
                self.embed = nn.Embedding(self.target_vocab_size, embedding_size)
                self.attn_calc = Seq2SeqLearning.Seq2SeqWithLearnableEmbeddings.Attention_BCB(
                                                              dls,embedding_size,hidden_size,max_length)         ## (C)
#                self.attn_calc = Seq2SeqLearning.Seq2SeqWithLearnableEmbeddings.Attention_SR(
#                                                               dls,embedding_size, hidden_size, max_length)     ## (D)
                self.gru = nn.GRU(self.hidden_size, self.hidden_size)                                            ## (E)
                self.out = nn.Linear(self.hidden_size, self.target_vocab_size)                                   ## (F)
        
            def forward(self, word_index, decoder_hidden, encoder_outputs):                                      ## (G)
                embedding = self.embed(word_index).view(1, 1, -1)
                attentional_hidden, attn_weights = self.attn_calc(embedding, decoder_hidden, encoder_outputs)    ## (H)
                output, hidden = self.gru(attentional_hidden, decoder_hidden)                                    ## (I)
                output = nn.LogSoftmax(dim=0)(self.out(output.view(-1)))                                         ## (J)
                output = torch.unsqueeze(output, 0)
                return output, hidden, attn_weights                                                         

        
        class Attention_BCB(nn.Module):
            """
            This model of attention is based on my interpretation of the logic presented in the
            following article by Bahdanau, Cho, and Bengio:

                     https://arxiv.org/pdf/1409.0473.pdf

            That should explain the suffix "BCB" in the name of the class.  More specifically, 
            my implementation corresponds to the Global Attention model described in Section 
            3.1 of the following paper by Luong, Pham, and Manning:

                     https://arxiv.org/pdf/1508.04025.pdf

            Eq. (7) of the paper by Luong et al. says that if h_t represents the current hidden 
            state in the DecoderRNN and if h_s_i, for i=1,..., max_length, represent the attention 
            units returned by the encoder, the contribution that each encoder h_s_i makes to the 
            decoder h_t is proportional to

                                            exp( score( h_t, h_s_i ) )    
                        c_t(s_i)   =    ----------------------------------
                                        \sum_j  exp( score( h_t, h_s_j ) ) 
                        
            where a 'general' approach to estimating the score() for a given h_t with respect to
            all the encoder attention units h_s_i, i=1,2,..., is given by 

                             score( h_t, h_s )  =    transpose( h_t ) W_a h_s
             
            In the implementation shown below, you will see two matrix products, one in line (N) 
            and the other in line (P).  The one in line (N) is what's called for by the above
            equation.  That matrix product amounts to multiplying each element of the current 
            decoder hidden with a linear combination of all the encoder attention units.  What you
            see in the first of the two equations shown above is implemented with the nn.LogSoftmax
            normalization in line (O).

            ClassPath:  Seq2SeqLearning -> Seq2SeqWithLearnableEmbeddings  ->  Attention_BCB
            """
            def __init__(self, dl_studio, embedding_size, hidden_size, max_length):
                super(Seq2SeqLearning.Seq2SeqWithLearnableEmbeddings.Attention_BCB, self).__init__()
                self.dl_studio = dl_studio
                self.max_length = max_length
                self.WC1 = nn.Linear( 2 * hidden_size, hidden_size )
                self.WC2 = nn.Linear( 2*hidden_size + embedding_size, embedding_size )

            def forward(self, prev_output_word, decoder_hidden, encoder_outputs):                            ## (K)
                contexts = torch.zeros(self.max_length).float().to(self.dl_studio.device)                    ## (L)
                for idx in range(self.max_length):                                                           ## (M)
                    contexts[idx] = decoder_hidden.view(-1) @ self.WC1(encoder_outputs[idx].view(-1))        ## (N)
                weights = nn.LogSoftmax(dim=-1)(contexts)                                                    ## (O)
                attentioned_hidden_state =  weights @ encoder_outputs                                        ## (P)
                attentioned_hidden_state = nn.Softmax(dim=-1)(attentioned_hidden_state)                      
                output = self.WC2(torch.cat( (attentioned_hidden_state.view(-1), 
                                                                     prev_output_word.view(-1)), 0 ) )
                output = torch.unsqueeze(torch.unsqueeze(output, 0), 0)
                weights = torch.unsqueeze(weights, 0)
                output = nn.ReLU()(output)
                return output, weights

        
        class Attention_SR(nn.Module):
            """
            This implementation of Attention is based on the logic used by Sean Robertson in his very 
            popular NLP tutorial:

                https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

            ClassPath:  Seq2SeqLearning -> Seq2SeqWithLearnableEmbeddings  ->  Attention_SR
            """
            def __init__(self, dl_studio, embedding_size, hidden_size, max_length):
                super(Seq2SeqLearning.Seq2SeqWithLearnableEmbeddings.Attention_SR, self).__init__()
                self.W = nn.Linear(embedding_size + hidden_size, max_length)
                self.attn_combine = nn.Linear(3*hidden_size, hidden_size)      

            def forward(self, prev_output_word, decoder_hidden, encoder_outputs):       
                contexts = self.W(torch.cat((prev_output_word[0], decoder_hidden[0]), 1)) 
                attn_weights = nn.Softmax(dim=1)( contexts )   
                attn_applied = torch.unsqueeze(attn_weights, 0) @  torch.unsqueeze(encoder_outputs, 0)
                output = torch.cat((prev_output_word[0], attn_applied[0]), 1)
                output =  torch.unsqueeze(self.attn_combine(output), 0)
                output = nn.ReLU()(output)
                return output, attn_weights
        
        
        def save_encoder(self, encoder):
            "Save the trained encoder to a disk file"       
            torch.save(encoder.state_dict(), self.dl_studio.path_saved_model["encoder"])

        def save_decoder(self, decoder):
            "Save the trained decoder to a disk file"       
            torch.save(decoder.state_dict(), self.dl_studio.path_saved_model["decoder"])
        

        def run_code_for_training_Seq2SeqWithLearnableEmbeddings(self, encoder, decoder, display_train_loss=False):        
            """
            Overall, the training consists of running the English/Spanish sentence pairs through the
            encoder-decoder combo.  For each English sentence, the encoder generates a max_length sized 
            tensor of attention units. As mentioned earlier, an attention unit is a concatenation of the
            forward hidden and the backward hidden at each position in the source sentence.  For an 
            example, if max_length equals 10 and if the size of the hidden in the encoder is 256, the
            tensor of all the attention units produced by the encoder will be of shape [10, 512]. This
            tensor is emitted at the output of the encoder in the call in line (S) shown below and becomes
            the value of the encoder_output variable.

            Here is an interesting difference between the operations of the encoder and the decoder
            during training: While both values returned by the encoder in line (S) are subsequently 
            put to use, that is not the case for the call to the decoder in line (T).  During training,
            we have no use for the 'decoder_output' returned by the decoder.  That is because, during
            training, the next input to the decoder is the next word in the target sequence, as shown
            in line (U).  However, during evaluation, it is the 'decoder_output' (after it is subject
            to torch.max() to find the index of the most probable target word) that yields the words
            for the target sequence.

            Regarding the loss function nn.NLLLoss used for training, note that using a combination
            of nn.LogSoftmax activation and nn.NLLLoss is the same thing as using nn.CrossEntropyLoss,
            which is the most commonly used loss function for solving classification problems. For a 
            neural network that is meant for solving a classification problem, the number of nodes in
            the output layer must equal the number of classes.  Applying nn.LogSoftmax activation to
            such a layer normalizes the values accumulated at those nodes so that they become a legal
            probability distribution over the classes.  Subsequently, calculating the nn.NLLLoss 
            means choosing the negative value in just that node which corresponds to the actual class 
            label of the input data. 

            That's exactly how we want to solve the problem of training the decoder here.  The number 
            of nodes in the output layer of the decoder equals the size of the target vocabulary.  For 
            example, in one of the datasets provided, the size of the Spanish vocabulary is 21823.  As 
            you can tell from line (G) in the definition of DecoderRNN shown previously, this is the 
            size of the output that will be emitted by the decoder.  In the code shown below, the 
            statement in line (V) applies nn.NLLLoss to this 21823 output vector vis-a-vis the integer 
            index for the Spanish word that was expected at the final step of the decoder for the
            input sentence in question.  The nn.NLLLoss will simply return negative of the value 
            in that node of the output which corresponds to the target word in the Spanish sentence.
            If the decoder logic did not make any prediction errors for that output word, then the 
            total probability mass accumulated at that node of the output layer will be 1. The log 
            operation of the nn.LogSoftmax activation will return the logarithm of that, which is 0.  
            And nn.NLLLoss will return the negative of the zero value as loss.
            """
            encoder.to(self.dl_studio.device)
            decoder.to(self.dl_studio.device)     
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=self.dl_studio.learning_rate)
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=self.dl_studio.learning_rate)
            encoder_scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=30, gamma=0.1, last_epoch=-1)
            decoder_scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=30, gamma=0.1, last_epoch=-1)
            criterion = nn.NLLLoss()
            accum_times = []
            start_time = time.perf_counter()
            training_loss_tally = []
            self.debug = False
            print("")
            num_sentence_pairs = len(self.training_corpus)
            print("\n\nNumber of sentence pairs in the dataset: ", num_sentence_pairs)
            print("\nNo sentence is longer than %d words (including the SOS and EOS tokens)\n\n" % self.max_length)
            running_loss = 0.0
            for iter in range(self.num_trials):
                pair = random.choice(self.training_corpus)
                ## See the doc comment for the function 'sentence_to_tensor()' for the
                ## shape of the en_tensor and es_tensor:
                en_tensor =  self.sentence_to_tensor(pair[0], 'en')                                             ## (Q)
                es_tensor =  self.sentence_to_tensor(pair[1], 'es')                                             ## (R)
                en_tensor = en_tensor.to(self.dl_studio.device)
                es_tensor = es_tensor.to(self.dl_studio.device)
                encoder_hidden = encoder.initHidden()                 
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                ## Run the bidirectional encoder to get the max_length attention units for the
                ## source sentence:
                encoder_outputs, encoder_hidden = encoder( en_tensor, encoder_hidden )                          ## (S)
                encoder_outputs = torch.squeeze(encoder_outputs)
                decoder_input = torch.tensor([[0]]).to(self.dl_studio.device)
                decoder_hidden = encoder_hidden[1]    
                decoder_hidden = torch.unsqueeze(decoder_hidden, 0)
                ## Find the number of words in the target sentence so we know the number of steps
                ## to execute with the decoder RNN:
                target_length = es_tensor.shape[0]
                loss = 0
                for di in range(target_length):
                    decoder_output, decoder_hidden,decoder_attention = decoder(decoder_input, 
                                                                          decoder_hidden, encoder_outputs)      ## (T)
                    decoder_input = es_tensor[di]                                                               ## (U)
                    loss += criterion(decoder_output, es_tensor[di])                                            ## (V)
                    if decoder_input.item() == 1:
                        break
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                loss_normed = loss.item() / target_length
                running_loss += loss_normed
                if iter % 500 == 499:    
                    avg_loss = running_loss / float(500)
                    training_loss_tally.append(avg_loss)
                    running_loss = 0.0
                    current_time = time.perf_counter()
                    time_elapsed = current_time-start_time
                    print("[iter:%4d  elapsed_time: %4d secs]     loss: %.2f" % (iter+1, time_elapsed,avg_loss))
                    accum_times.append(current_time-start_time)
            print("\nFinished Training\n")
            self.save_encoder(encoder)       
            self.save_decoder(decoder)       
            if display_train_loss:
                plt.figure(figsize=(10,5))
                plt.title("Training Loss vs. Iterations")
                plt.plot(training_loss_tally)
                plt.xlabel("iterations")
                plt.ylabel("training loss")
                plt.legend()
                plt.savefig("training_loss.png")
                plt.show()


        def run_code_for_evaluating_Seq2SeqWithLearnableEmbeddings(self, encoder, decoder):
            """
            The main difference between the training code and the evaluation code shown here
            is with regard to how we process the output of the DecoderRNN.  For the training loop,
            our goal was to use the nn.NLLLoss to choose that value from the output of the decoder
            that corresponded to the integer index of the target word.  Since nn.NLLLoss was 
            supplied with that integer index, we ourselves did not have to peer inside the output
            of the decoder.

            During the evaluation phase, however, at each step of the DecoderRNN, we must extract 
            from the decoder output the integer index of the most probable word in the target 
            language. This is illustrated in lines (c) through (f) of the code shown below. We 
            must call on torch.max() for the output emitted by the decoder to identify the node 
            in the output layer that has the highest accumulated probability mass.  If the index 
            of this node is 1, that means that the decoder has encountered the the end-of-sentence 
            token, EOS, and we must break the loop being executed by the decoder RNN.  If the 
            index returned by torch.max() for the largest value is other than 1, we identify the 
            corresponding output word in line (f), which is subsequently added to the output 
            sentence under construction.  

            A cool thing to do during the evaluation phase is to see how well the attention 
            mechanism is working for aligning the corresponding words and phrases between the
            source sentence and the target sentence.  In the implementation shown below, this
            is done with the help of the decoder_attentions tensor, of size [max_length, 
            max_length]. As shown in line (c), each row of this tensor stores the max_length 
            attention weights returned by the Attention Network via the decoder output for the
            corresponding word emitted for the output sentence.
            """
            encoder.load_state_dict(torch.load(self.dl_studio.path_saved_model['encoder']))
            decoder.load_state_dict(torch.load(self.dl_studio.path_saved_model['decoder']))
            encoder.to(self.dl_studio.device)
            decoder.to(self.dl_studio.device)     
            with torch.no_grad():
                for iter in range(20):
                    pair = random.choice(self.training_corpus)
                    en_tensor =  self.sentence_to_tensor(pair[0], 'en')
                    en_tensor = en_tensor.to(self.dl_studio.device)
                    encoder_hidden = encoder.initHidden()
                    ## encoder_outputs is the time-evolution of the encoder hidden state and encoder
                    ## hidden are the two final states for the R2L and L2R scans of the source sentence:
                    encoder_outputs, encoder_hidden = encoder( en_tensor, encoder_hidden )                      ## (W)
                    encoder_outputs = torch.squeeze(encoder_outputs)
                    decoder_input = torch.tensor([[0]]).to(self.dl_studio.device)                               ## (X)
                    ## We set the initial value of decoder_hidden to the final value of encoder_hidden:
                    decoder_hidden = encoder_hidden[1]                                                          ## (Y)  
                    decoder_hidden = torch.unsqueeze(decoder_hidden, 0)
                    decoded_words = []
                    ## For each word that is generated in the target language, we want to record the attention
                    ## vector that was used for that generation.  This is to allow for the visualization of the
                    ## alignment between the source words and the target words:
                    decoder_attentions = torch.zeros(self.max_length, self.max_length)                          ## (Z)
                    for di in range(self.max_length):
                        decoder_output, decoder_hidden,decoder_attention = decoder(decoder_input,              
                                                                       decoder_hidden, encoder_outputs)         ## (a)
                        decoder_attentions[di] = decoder_attention                                              ## (b)
                        _, idx_max =  torch.max(decoder_output, 1)                                              ## (c)
                        ## 1 is the word index for the EOS token:
                        if idx_max.item() == 1:                                                                 ## (d) 
                            decoded_words.append('EOS')                                                         ## (e)
                            break
                        else:
                            decoded_words.append(self.es_index_2_word[idx_max.item()])                          ## (f)
                        decoder_input = torch.squeeze(idx_max)
                    output_sentence = " ".join(decoded_words)
                    print("\n\n\nThe input sentence pair: ", pair)
                    print("\nThe translation produced by Seq2Seq: ", output_sentence)
                    self.show_attention(pair[0], decoded_words, decoder_attentions)


        def show_attention(self, input_sentence, output_words, attentions):
            input_words = input_sentence.split(' ')
            attentions_main_part =  attentions[1:len(output_words)-1, 1:len(input_words)-1]
            fig = plt.figure()
            ax = fig.subplots()
            cax = ax.matshow(attentions_main_part.numpy(), cmap='bone')
            fig.colorbar(cax)
            ## Mark the positions of the tick marks but subtract 2 to exclude the 
            ## SOS and EOS tokens from the sentnece
            ax.set_xticks(np.arange(len(input_words) - 2))
            ax.set_yticks(np.arange(len(output_words) - 2))
            ax.set_xticklabels(input_words[1:-1], rotation=90, fontsize=16)
            ax.set_yticklabels(output_words[1:-1], fontsize=16)
            plt.show()

        def show_attention2(self, input_sentence, output_words, attentions):
            fig = plt.figure()
            ax = fig.subplots()
            cax = ax.matshow(attentions.numpy(), cmap='bone')
            fig.colorbar(cax)
            ## mark the positions of the tick marks:
            ax.set_xticks(np.arange(self.max_length))
            ax.set_yticks(np.arange(self.max_length))
            input_words = input_sentence.split(' ')
            ## We need to take care of the possibilities that that the input and the 
            ## output sentences will be shorter than the value of self.max_length:
            while len(input_words) < self.max_length:
                input_words.append(' ')
            while len(output_words) < self.max_length:
                output_words.append(' ')
            ax.set_xticklabels(input_words, rotation=90)
            ax.set_yticklabels(output_words)
            plt.show()



    ###%%%
    ########################################################################################
    #########  Start Definition of Inner Class Seq2SeqWithPretrainedEmbeddings  ############

    class Seq2SeqWithPretrainedEmbeddings(nn.Module):             
        """
        Please read the doc section of the previous Seq2SeqLearning class, 
        Seq2SeqWithLearnableEmbeddings, for the basic documentation that also applies to 
        the class being presented here. 

        While the previous class shows how to carry out Seq2Seq learning when you allow the
        framework to learn their own numeric embeddings for the words, in the class shown in
        this section of Seq2SeqLearning we use the pre-trained word2vec embeddings from 
        Google for the source language sentences.

        At the moment, I am using the pre-trained embeddings for only the source language
        sentence because of the constraints on the fast memory that come into existence 
        when you use pre-trained embeddings for multiple languages simultaneously.  My 
        original plan was to use word2vec embeddings for the source language English and
        the Fasttext embeddings for the target language Spanish.  The pre-trained word2vec
        embeddings for English occupy nearly 4GB of RAM and the pre-trained Fasttext 
        embeddings another 8GB.  The two objects co-residing in the fast memory brings 
        down to heel a 32GB machine.
        
        Another interesting thing to keep in mind is the two different ways in which the
        target language is used in seq2seq learning.  In addition to the word embeddings 
        needed for the decoder GRU, you also use the integer word indexes directly for the 
        following reason:  You see, one would like to use nn.LogSoftmax for the final 
        activation in the overall network and nn.NLLLoss for the loss.  These choices 
        allow you to use the classifier-network principles for training.  That is, you 
        ask the decoder to correctly label the next output word by giving it a class 
        label that is an integer index spanning the size of the target vocabulary. 
        With nn.NLLLoss, for the target needed by the loss function, all you need to is 
        to supply it with the integer index of the ground-truth target word.

        For the classifier based logic mentioned above to work, you need to ensure that 
        the output layer of the decoder network has the same number of nodes as the size 
        of the target vocabulary. As mentioned above, during training, for calculating the 
        loss, the nn.NLLLoss is supplied with the integer index of the target word at that 
        step of the decoder RNN. The loss function returns the negative of the value 
        stored in the corresponding output node of the network.  Recall, the values in
        the output nodes would be produced by the application of nn.LogSoftmax to the
        values calculated by there by forward propagation.

        An alternative to using classifier-network based principles for guiding the design
        of the decoder would be to cast the problem of predicting the output word as an
        exercise in regression (when using pre-trained embeddings for both the source and
        the target languages).  I have played with that approach.  Eventually, I gave up
        on it because it yielded poor results even on short sequences.

        I should also mention that the attention mechanism used in this class is exactly 
        the same as for the case of learnable embeddings and the need for attention the same.

        I have used the same dataset for the demonstrations that follow as in the previous
        class with learnable embeddings.  Please see the doc section of 
        Seq2SeqWithLearnableEmbeddings for the dataset related information.

        ClassPath:  Seq2SeqLearning -> Seq2SeqWithPretrainedEmbeddings
        """
        def __init__(self, dl_studio, dataroot, data_archive, path_to_saved_embeddings_en, 
                                             embeddings_type, max_length, embedding_size, num_trials):
            super(Seq2SeqLearning.Seq2SeqWithPretrainedEmbeddings, self).__init__()
            self.dl_studio = dl_studio
            self.dataroot  = dataroot
            self.data_archive = data_archive
            self.path_to_saved_embeddings_en = path_to_saved_embeddings_en
            self.max_length = max_length
            self.embedding_size = embedding_size
            self.num_trials = num_trials
            f = gzip.open(dataroot + data_archive, 'rb')
            dataset = f.read()
            dataset,vocab_en,vocab_es = pickle.loads(dataset, encoding='latin1')
            self.dataset = dataset
            self.vocab_en = vocab_en
            self.vocab_es = vocab_es
            self.vocab_en_size = len(vocab_en)          #  encludes the SOS and EOS tokens
            self.vocab_es_size = len(vocab_es)          #  encludes the SOS and EOS tokens
            print("\n\nSize of the English vocab in dataset: ", self.vocab_en_size)
            print("\nSize of the Spanish vocab in dataset: ", self.vocab_es_size)
            self.debug = False
            if self.debug:
                print("\n\nFirst 100 elements of English vocab: ", vocab_en[:100])
                print("\n\nFirst 100 elements of Spanish vocab: ", vocab_es[:100])
            # The first two elements of both vocab_en and vocab_es are the SOS and EOS tokens
            # So the index position for SOS is 0 and for EOS is 1.
            self.en_vocab_dict = { vocab_en[i] : i  for i in range(self.vocab_en_size) }  
            self.es_vocab_dict = { vocab_es[i] : i  for i in range(self.vocab_es_size) }
            self.es_index_2_word =   { i : vocab_es[i] for i in range(self.vocab_es_size) }
            self.training_corpus = dataset

            if embeddings_type == 'word2vec':         
                import gensim.downloader as genapi
                from gensim.models import KeyedVectors
                if os.path.exists(path_to_saved_embeddings_en + 'vectors.kv'):
                    self.word_vectors_en = KeyedVectors.load(path_to_saved_embeddings_en + 'vectors.kv')
                else:
                    print("""\n\nSince this is your first time to install the word2vec embeddings, it may take"""
                          """\na couple of minutes. The embeddings occupy around 3.6GB of your disk space.\n\n""")
                    self.word_vectors_en = genapi.load("word2vec-google-news-300")               
                    self.word_vectors_en.save(path_to_saved_embeddings_en + 'vectors.kv')    
            elif embeddings_type == 'fasttext':                
                import fasttext.util
                if os.path.exists(path_to_saved_embeddings_en + "cc.en.300.bin"):
                    self.word_vectors_en = fasttext.load_model(path_to_saved_embeddings_en + "cc.en.300.bin")
                else:
                    print("""\n\nSince this is your first time to install the English fastText embeddings, """
                      """\nit may take a couple of minutes. The embeddings occupy around 3.6GB of your """
                      """ disk space.\n\n""")
                    os.chdir(path_to_saved_embeddings_en)
                    fasttext.util.download_model('en', if_exists='ignore')
                    os.chdir(".")
                    self.word_vectors_en = fasttext.load_model(path_to_saved_embeddings_en + "cc.en.300.bin")

            self.sos_tensor_en = torch.zeros( embedding_size, dtype=float )
            self.sos_tensor_en[0] = 1.0
            self.eos_tensor_en = torch.zeros( embedding_size, dtype=float )
            self.eos_tensor_en[1] = 1.0


        def sentence_to_tensor(self, sentence, lang):
            """
            First read the doc comment for the same method in the Seq2SeqWithLearnableEmbeddings 
            part of Seq2SeqLearning.  You are currently in the Seq2SeqWithPretrainedEmbeddings part.

            The implementation shown below is partly different on account of the fact that now
            we want to use the word2vec embeddings for the source language sentences.

            For a sentence in the source language, for a sentence consisting of N words, this 
            function returns a tensor of shape [N, 300] since 300 is the size of the word2vec 
            embeddings I have used here.  For a sentence consisting of N words in the target
            language, this method returns a tensor of shape [N,1], with the N values in the
            tensor corresponding to the integer indices of the words in the target language 
            vocab.
            """
            list_of_embeddings = []
            words = sentence.split(' ')
            if lang == "en":
                ## The corpus sentences come with prefixed 'SOS' and 'EOS' tokens. We need to
                ##   drop them for now and later insert their embedding-like tensor equivalents:
                words = words[1:-1]                
                for i,word in enumerate(words):
                    if word in self.word_vectors_en.key_to_index:
                        embedding = self.word_vectors_en[word]
                        list_of_embeddings.append(np.array(embedding))
                list_of_embeddings.insert(0,self.sos_tensor_en.numpy())  
                list_of_embeddings.append(self.eos_tensor_en.numpy()) 
                sentence_tensor = torch.FloatTensor( list_of_embeddings )
            elif lang == "es":
                sentence_tensor = torch.zeros(len(words), 1, dtype=torch.long)
                for i,word in enumerate(words):
                    sentence_tensor[i] = self.es_vocab_dict[word]    
            return sentence_tensor


        class EncoderRNN(nn.Module):
            """
            First read EncoderRNN's doc comment in the Seq2SeqWithLearnableEmbeddings 
            part of Seq2SeqLearning.  You are currently in the Seq2SeqWithPretrainedEmbeddings 
            part.

            ALMOST ALL of the comments made in the long doc section associated with the other 
            EncoderRNN apply here also.

            As in the previous definition of EncoderRNN for the Seq2SeqWithLearnableEmbeddings
            class, I'll again use a bi-directional GRU for the encoder. If the number of words
            in a sentence is N and the size of the hidden state is, say, 300, the output of
            the encoder will emit a time-evolution of the hidden represented by a tensor of
            shape [N, 600] in which each 600-valued tensor is a concatenation of the forward
            hidden and the backward hidden during the forward and the backward scan of the 
            input sentence.

            With regard to what the encoder returns, both 'output' and 'hidden' are critical
            to the operation of the decoder, as you will see later.  As explained, 'output' 
            is the time-evolution of the hidden in the GRU and 'hidden' is the final value
            of the encoder hidden state.  The former is needed for calculating the attention
            weights and the latter becomes the initial hidden for the decoder.

            ClassPath:  Seq2SeqLearning -> Seq2SeqWithPretrainedEmbeddings ->  EncoderRNN
            """
            def __init__(self, dls, s2s, embedding_size, hidden_size, max_length):
                super(Seq2SeqLearning.Seq2SeqWithPretrainedEmbeddings.EncoderRNN, self).__init__()
                self.dl_studio = dls
                self.source_vocab_size = s2s.vocab_en_size
                self.embedding_size = embedding_size
                self.hidden_size = hidden_size
                self.max_length = max_length
                self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True)
        
            def forward(self, sentence_tensor, hidden):
                word_embeddings = torch.zeros(self.max_length, 1, self.hidden_size).float().to(self.dl_studio.device)
                for i in range(sentence_tensor.shape[0]):
                    word_embeddings[i] = sentence_tensor[i].view(1, 1, -1)
                output, hidden = self.gru(word_embeddings, hidden)
                return output, hidden
        
            def initHidden(self):
                return torch.zeros(2, 1, self.hidden_size).float().to(self.dl_studio.device)
        
        
        class DecoderRNN(nn.Module):
            """
            First read DecoderRNN's doc comment in the Seq2SeqWithLearnableEmbeddings 
            part of Seq2SeqLearning.  You are currently in the Seq2SeqWithPretrainedEmbeddings 
            part.

            The decoder presented below for the case of pre-trained embeddings is identical 
            to the one that was presented previously for the case of learnable embeddings.  
            That should not be surprising because I am using the pre-trained embeddings for
            just the source language for reasons explained in the main comment doc associated 
            with the class Seq2SeqWithPretrainedEmbeddings.

            Recall that we are using attention in the decoder to modify the value of the 
            decoder hidden state using the attention units provided by the encoder for the
            source-language sentence.

            It is also important to remember that while we used the EncoderRNN in the 
            bi-directional mode, the DecoderRNN is being made to operate in the more traditional 
            mode of emitting one output word at a time.

            To remind the reader again, both the input to the decoder as specified by the first 
            argument of forward() and the output that is emitted by the GRU must be word index 
            values, that is, the integers that correspond to the word positions in the sorted 
            vocabulary list for the target language.  

            ClassPath:  Seq2SeqLearning -> Seq2SeqWithPretrainedEmbeddings ->  DecoderRNN
            """
            def __init__(self, dls, s2s, embedding_size, hidden_size, max_length):
                super(Seq2SeqLearning.Seq2SeqWithPretrainedEmbeddings.DecoderRNN, self).__init__()
                self.hidden_size = hidden_size
                self.target_vocab_size = s2s.vocab_es_size
                self.max_length = max_length
                self.embed = nn.Embedding(self.target_vocab_size, embedding_size)
                self.attn_calc = Seq2SeqLearning.Seq2SeqWithPretrainedEmbeddings.Attention_BCB(dls, 
                                                               embedding_size, hidden_size, max_length)  
#                self.attn_calc = Seq2SeqLearning.Seq2SeqWithPretrainedEmbeddings.Attention_SR(dls,
#                                                               embedding_size, hidden_size, max_length)
                self.gru = nn.GRU(self.hidden_size, self.hidden_size)
                self.out = nn.Linear(self.hidden_size, self.target_vocab_size)
        
            def forward(self, word_index, decoder_hidden, encoder_outputs):                                      
                embedding = self.embed(word_index).view(1, 1, -1)
                attentional_hidden, attn_weights = self.attn_calc(embedding, decoder_hidden, encoder_outputs)    
                output, hidden = self.gru(attentional_hidden, decoder_hidden)                                    
                output = nn.LogSoftmax(dim=0)(self.out(output.view(-1)))                                         
                output = torch.unsqueeze(output, 0)
                return output, hidden, attn_weights                                                         


        class Attention_BCB(nn.Module):
            """
            First read the doc comment for Attention_BCB in the Seq2SeqWithLearnableEmbeddings 
            part of Seq2SeqLearning.  You are currently in the Seq2SeqWithPretrainedEmbeddings 
            part.

            As I mentioned previously, the model of attention shown below is based on my 
            interpretation of the logic presented in the paper by Bahdanau, Cho, and Bengio.
            To recall the salient points of the more detailed explanation provided earlier,
            using attention means comparing the current hidden state in the decoder with each 
            of the attention units provided by the encoder for the source language sentence. 
            Through this comparison, we want the system to learn as to what extent it should let 
            each attention unit of the source sentence influence the current hidden state in 
            the decoder.

            ClassPath:  Seq2SeqLearning -> Seq2SeqWithPretrainedEmbeddings ->  Attention_BCB
            """
            def __init__(self, dl_studio, embedding_size, hidden_size, max_length):
                super(Seq2SeqLearning.Seq2SeqWithPretrainedEmbeddings.Attention_BCB, self).__init__()
                self.dl_studio = dl_studio
                self.max_length = max_length
                self.WC1 = nn.Linear( 2 * hidden_size, hidden_size )
                self.WC2 = nn.Linear( 2*hidden_size + embedding_size, embedding_size )

            def forward(self, prev_output_word, decoder_hidden, encoder_outputs):                          
                contexts = torch.zeros(self.max_length).float().to(self.dl_studio.device)                  
                for idx in range(self.max_length):                                                         
                    contexts[idx] = decoder_hidden.view(-1) @ self.WC1(encoder_outputs[idx].view(-1))      
                weights = nn.LogSoftmax(dim=-1)(contexts)                                                  
                attentioned_hidden_state =  weights @ encoder_outputs                                      
                attentioned_hidden_state = nn.Softmax(dim=-1)(attentioned_hidden_state)                      
                output = self.WC2(torch.cat( (attentioned_hidden_state.view(-1), 
                                                                     prev_output_word.view(-1)), 0 ) )
                output = torch.unsqueeze(torch.unsqueeze(output, 0), 0)
                weights = torch.unsqueeze(weights, 0)
                output = nn.ReLU()(output)
                return output, weights

        class Attention_SR(nn.Module):
            """
            First read the doc comment for Attention_BCB in the Seq2SeqWithLearnableEmbeddings 
            part of Seq2SeqLearning.  You are currently in the Seq2SeqWithPretrainedEmbeddings part.

            This implementation of Attention is based on the logic used by Sean Robertson in his 
            NLP tutorial.

            ClassPath:  Seq2SeqLearning -> Seq2SeqWithPretrainedEmbeddings ->  Attention_SR
            """
            def __init__(self, dl_studio, embedding_size, hidden_size, max_length):
                super(Seq2SeqLearning.Seq2SeqWithPretrainedEmbeddings.Attention_SR, self).__init__()
                self.W = nn.Linear(embedding_size + hidden_size, max_length)
                self.attn_combine = nn.Linear(3*hidden_size, hidden_size)      

            def forward(self, prev_output_word, decoder_hidden, encoder_outputs):       
                contexts = self.W(torch.cat((prev_output_word[0], decoder_hidden[0]), 1)) 
                attn_weights = nn.Softmax(dim=1)( contexts )   
                attn_applied = torch.unsqueeze(attn_weights, 0) @  torch.unsqueeze(encoder_outputs, 0)
                output = torch.cat((prev_output_word[0], attn_applied[0]), 1)
                output =  torch.unsqueeze(self.attn_combine(output), 0)
                output = nn.ReLU()(output)
                return output, attn_weights
        

        def save_encoder(self, encoder):
            "Save the trained encoder to a disk file"       
            torch.save(encoder.state_dict(), self.dl_studio.path_saved_model["encoder"])

        def save_decoder(self, decoder):
            "Save the trained decoder to a disk file"       
            torch.save(decoder.state_dict(), self.dl_studio.path_saved_model["decoder"])


        def run_code_for_training_Seq2SeqWithPretrainedEmbeddings(self, encoder, decoder, display_train_loss=False):        
            """
            First read the doc comment for the training method in the Seq2SeqWithLearnableEmbeddings 
            part of Seq2SeqLearning.  You are currently in the Seq2SeqWithPretrainedEmbeddings part.

            As mentioned in the doc section of the training method for the learnable embeddings case,
            overall, the training consists of running the English/Spanish sentence pairs through the
            encoder-decoder combo.  For each English sentence, the encoder generates a max_length sized 
            tensor of the attention units. As to what is meant by an attention unit was explained in
            the doc comment for the version of the training method for the case of learnable embeddings.

            To remind the reader again, we have no use for 'decoder_output' returned by the decoder
            during the training phase.  That is because the next input to the decoder is the next 
            word in the target sequence.  However, during evaluation, it is the decoder_output that 
            yields the words for the target sequence.

            See the doc comment for the training method for the learnable embeddings case for an
            explanation of why you need to use nn.NLLLoss for the loss function.
            """
            encoder.to(self.dl_studio.device)
            decoder.to(self.dl_studio.device)     
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=self.dl_studio.learning_rate)
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=self.dl_studio.learning_rate)
            encoder_scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=30, gamma=0.1, last_epoch=-1)
            decoder_scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=30, gamma=0.1, last_epoch=-1)
            criterion = nn.NLLLoss()
            accum_times = []
            start_time = time.perf_counter()
            training_loss_tally = []
            self.debug = False
            print("")
            num_sentence_pairs = len(self.training_corpus)
            print("\n\nNumber of sentence pairs in the dataset: ", num_sentence_pairs)
            print("\nNo sentence is longer than %d words (including the SOS and EOS tokens)\n\n" % self.max_length)
            running_loss = 0.0
            for iter in range(self.num_trials):
                pair = random.choice(self.training_corpus)
                ## See the doc comment for the function 'sentence_to_tensor()' for the
                ## shape of the en_tensor and es_tensor:
                en_tensor =  self.sentence_to_tensor(pair[0], 'en')                                         
                es_tensor =  self.sentence_to_tensor(pair[1], 'es')                                         
                en_tensor = en_tensor.to(self.dl_studio.device)
                es_tensor = es_tensor.to(self.dl_studio.device)
                encoder_hidden = encoder.initHidden()                 
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                ## Run the bidirectional encoder to get the max_length attention units for the
                ## source sentence:
                encoder_outputs, encoder_hidden = encoder( en_tensor, encoder_hidden )                      
                encoder_outputs = torch.squeeze(encoder_outputs)
                decoder_input = torch.tensor([[0]]).to(self.dl_studio.device)
                decoder_hidden = encoder_hidden[1]    
                decoder_hidden = torch.unsqueeze(decoder_hidden, 0)
                ## Find the number of words in the target sentence so we know the number of steps
                ## to execute with the decoder RNN:
                target_length = es_tensor.shape[0]
                loss = 0
                for di in range(target_length):
                    decoder_output, decoder_hidden,decoder_attention = decoder(decoder_input, 
                                                                          decoder_hidden, encoder_outputs)  
                    decoder_input = es_tensor[di]                                                           
                    loss += criterion(decoder_output, es_tensor[di])                                        
                    if decoder_input.item() == 1:
                        break
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                loss_normed = loss.item() / target_length
                running_loss += loss_normed
                if iter % 500 == 499:    
                    avg_loss = running_loss / float(500)
                    training_loss_tally.append(avg_loss)
                    running_loss = 0.0
                    current_time = time.perf_counter()
                    time_elapsed = current_time-start_time
                    print("[iter:%4d  elapsed_time: %4d secs]     loss: %.2f" % (iter+1, time_elapsed,avg_loss))
                    accum_times.append(current_time-start_time)
            print("\nFinished Training\n")
            self.save_encoder(encoder)       
            self.save_decoder(decoder)       
            if display_train_loss:
                plt.figure(figsize=(10,5))
                plt.title("Training Loss vs. Iterations")
                plt.plot(training_loss_tally)
                plt.xlabel("iterations")
                plt.ylabel("training loss")
                plt.legend()
                plt.savefig("training_loss.png")
                plt.show()


        def run_code_for_evaluating_Seq2SeqWithPretrainedEmbeddings(self, encoder, decoder):
            """
            First read the doc comment for the evaluation method in the Seq2SeqWithLearnableEmbeddings 
            part of Seq2SeqLearning.  You are currently in the Seq2SeqWithPretrainedEmbeddings part.

            As I mentioned in the learning embeddings version of the evaluation method, the main 
            difference between the training code and the evaluation code is with regard to how we 
            process the output of the DecoderRNN.  For the training loop, our goal was to use
            nn.NLLLoss to choose that value from the output of the decoder that correspondeds to 
            the integer index of the target word.  Since nn.NLLLoss was supplied with that integer 
            index, we ourselves did not have to peer inside the output of the decoder. During the 
            evaluation phase, however, at each step of the DecoderRNN, we must extract from the 
            decoder output the integer index of the most probable word in the target language.

            To remind the reader again, a cool thing to do during the evaluation phase is to see 
            how well the attention mechanism is working for aligning the corresponding words and 
            phrases between the source sentence and the target sentence. 
            """
            encoder.load_state_dict(torch.load(self.dl_studio.path_saved_model['encoder']))
            decoder.load_state_dict(torch.load(self.dl_studio.path_saved_model['decoder']))
            encoder.to(self.dl_studio.device)
            decoder.to(self.dl_studio.device)     
            with torch.no_grad():
                for iter in range(20):
                    pair = random.choice(self.training_corpus)
                    en_tensor =  self.sentence_to_tensor(pair[0], 'en')
                    en_tensor = en_tensor.to(self.dl_studio.device)
                    encoder_hidden = encoder.initHidden()
                    ## encoder_outputs is the time-evolution of the encoder hidden state and encoder
                    ## hidden are the two final states for the R2L and L2R scans of the source sentence:
                    encoder_outputs, encoder_hidden = encoder( en_tensor, encoder_hidden )              
                    encoder_outputs = torch.squeeze(encoder_outputs)
                    decoder_input = torch.tensor([[0]]).to(self.dl_studio.device)                       
                    ## We set the initial value of decoder_hidden to the final value of encoder_hidden:
                    decoder_hidden = encoder_hidden[1]                                                  
                    decoder_hidden = torch.unsqueeze(decoder_hidden, 0)
                    decoded_words = []
                    ## For each word that is generated in the target language, we want to record the attention
                    ## vector that was used for that generation.  This is to allow for the visualization of the
                    ## alignment between the source words and the target words:
                    decoder_attentions = torch.zeros(self.max_length, self.max_length)                  
                    for di in range(self.max_length):
                        decoder_output, decoder_hidden,decoder_attention = decoder(decoder_input,              
                                                                       decoder_hidden, encoder_outputs) 
                        decoder_attentions[di] = decoder_attention                                      
                        _, idx_max =  torch.max(decoder_output, 1)                                      
                        ## 1 is the word index for the EOS token:
                        if idx_max.item() == 1:                                                         
                            decoded_words.append('EOS')                                                 
                            break
                        else:
                            decoded_words.append(self.es_index_2_word[idx_max.item()])                  
                        decoder_input = torch.squeeze(idx_max)
                    output_sentence = " ".join(decoded_words)
                    print("\n\n\nThe input sentence pair: ", pair)
                    print("\nThe translation produced by Seq2Seq: ", output_sentence)
                    self.show_attention(pair[0], decoded_words, decoder_attentions)


        def show_attention(self, input_sentence, output_words, attentions):
            input_words = input_sentence.split(' ')
            attentions_main_part =  attentions[1:len(output_words)-1, 1:len(input_words)-1]
            fig = plt.figure()
            ax = fig.subplots()
            cax = ax.matshow(attentions_main_part.numpy(), cmap='bone')
            fig.colorbar(cax)
            ## Mark the positions of the tick marks but subtract 2 to exclude the 
            ## SOS and EOS tokens from the sentnece
            ax.set_xticks(np.arange(len(input_words) - 2))
            ax.set_yticks(np.arange(len(output_words) - 2))
            ax.set_xticklabels(input_words[1:-1], rotation=90, fontsize=16)
            ax.set_yticklabels(output_words[1:-1], fontsize=16)
            plt.show()

        def show_attention2(self, input_sentence, output_words, attentions):
            fig = plt.figure()
            ax = fig.subplots()
            cax = ax.matshow(attentions.numpy(), cmap='bone')
            fig.colorbar(cax)
            ## mark the positions of the tick marks:
            ax.set_xticks(np.arange(self.max_length))
            ax.set_yticks(np.arange(self.max_length))
            input_words = input_sentence.split(' ')
            ## We need to take care of the possibilities that that the input and the 
            ## output sentences will be shorter than the value of self.max_length:
            while len(input_words) < self.max_length:
                input_words.append(' ')
            while len(output_words) < self.max_length:
                output_words.append(' ')
            ax.set_xticklabels(input_words, rotation=90)
            ax.set_yticklabels(output_words)
            plt.show()


#_________________________  End of Seq2SeqLearning Class Definition ___________________________

#______________________________    Test code follows    _________________________________

if __name__ == '__main__': 
    pass
