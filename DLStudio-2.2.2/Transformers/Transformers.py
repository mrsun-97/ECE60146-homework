# -*- coding: utf-8 -*-

__version__   = '2.2.2'
__author__    = "Avinash Kak (kak@purdue.edu)"
__date__      = '2022-March-5'
__url__       = 'https://engineering.purdue.edu/kak/distDLS/DLStudio-2.2.2.html'
__copyright__ = "(C) 2022 Avinash Kak. Python Software Foundation."


__doc__ = '''


  You are looking at the Transformers co-class file in the DLStudio module.
  For the overall documentation on DLStudio, visit:

           https://engineering.purdue.edu/kak/distDLS/



TRANSFORMERS:

    There are two different versions of transformers in this file:

         TransformerFG                 [FG stands for "First Generation"]
         TransformerPreLN              [PreLN stands for "Pre Layer Norm"]
 
    The very small difference between the two has important consequences with
    regard to how fast and how stably the learning takes place in the two
    implementations.  I could have easily combined the two implementations
    with a small number of conditional statements to account for the
    differences, however I have chosen to keep them separate in order to make
    it easier for the two to evolve separately and to be used differently for
    educational purposes.

    The TransformerFG implementation is based on the transformers as first
    envisioned in the seminal paper "Attention is All You Need" by Vaswani et
    el.  And, the network code in TransformerPreLN incorporates the
    modifications suggested by "On Layer Normalization in the Transformer
    Architecture" by Xiong et al.  Both these papers are now so well known
    that just entering their title in a Google search window will take you
    directly to their PDFs at arxiv.org.

    I should also mention that the Transformers class (the main class defined
    in this file) is the container class for both TransformerFG and
    TransformerPreLN.  The top-level Transformers class is a co-class in the
    DLStudio module.  Recall that, in the DLStudio module, a co-class resides
    at the same level of abstraction as the main DLStudio class.

    The sequence-to-sequence problem addressed in both TransformerFG and
    TransformerPreLN is the English-to-Spanish translation problem, which is
    the same problem as in the Seq2SeqLearning co-class in the DLStudio
    module.  The seq2seq learning in the Seq2SeqLearning class was based on
    using recurrence along with the notion of attention.  Recurrence was used
    to generate a time evolution of the hidden state as a sentence was scanned
    left-to-right and again from right-to-left.  Each state in the time
    evolution stood for a summarized representation of the sentence up to that
    point in the scan.  Concatenating the forward-scan hidden state at each
    word position in a sentence with the backward-scan hidden state at the
    same position yielded a sequence of attention units for a sentence.
    Subsequently, the neural network had to learn the relationship between
    these attention units for a source-language sentence and the word that
    needed to be produced as each position in the target language sentence.
    With that as a summary of how attention was used in the Seq2SeqLearning
    class, we now turn to the subject of transformers as presented in the
    inner classes TransformerFG and TransformerPreLN here.

    The main difference between how I used attention in the Seq2SeqLearning
    class and how I'll use it in the two transformers here is that now we have
    no use for recurrence.  That language modeling could be carried out without
    using recurrence was first demonstrated in "Attention is All You Need"
    (https://arxiv.org/pdf/1706.03762.pdf) by Vaswani, et el.  As shown in
    that paper, with transformers, you need to use only the attention
    mechanism to determine how the different parts of a source sequence relate
    to one another with regard to the production of the next word in the
    target sequence.  The intra-sentence relationships between the words in
    the source and the target languages are referred to as self-attention and
    the inter-sentence relationships between the words in two different
    languages are referred to as the cross-attention.  This is further
    explained in the documentation associated with the inner classes
    TransformerFG and TransformerPreLN below.

    About the dataset I'll be using to demonstrate transformers, version 2.2.2
    of DLStudio comes with a data archive named en_es_corpus_for_transformers
    that contains the following archive

           en_es_xformer_8_90000.tar.gz

    As for the name of the archive, the number 8 refers to the maximum number
    of words in a sentence, which translates into sentences with a maximum
    length of 10 when you include the SOS and EOS tokens at the two ends of a
    sentence.  The number 90,000 is for how many English-Spanish sentence
    pairs are there in the archive.

    The following two scripts in the ExamplesTransformers directory of the
    distribution are your main entry points for experimenting with the
    Transformers code:

        seq2seq_with_transformerFG.py
        seq2seq_with_transformerPreLN.py

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

#_____________________________________________  Transformers Class Definition  _______________________________________________

class Transformers(object):

    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''Transformers constructor can only be called with keyword arguments for 
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
    ##################################  Start Definition of Inner Class TransformerFG  ####################################

    class TransformerFG(nn.Module):             
        """
        TransformerFG stands for "Transformer First Generation".

        The goal of this class is to serve as an instructional aid in understanding the basic 
        concepts of attention-based learning with neural networks --- as laid out in the 
        original paper "Attention is All You Need" (https://arxiv.org/pdf/1706.03762.pdf) 
        by Vaswani, et el.  This paper created a paradigm shift in the deep learning 
        area by showing that it was possible to replace convolution and recurrence as 
        the basic architectural elements of a neural network design by just attention. In the 
        context of seq2seq learning, attention takes two forms: self-attention and 
        cross-attention.  Self-attention means for a neural network to figure out on its own 
        what parts of a sentence contribute jointly to the generation of the words in the
        target language. To elaborate, consider the following sentence in English:

              "I was talking to my friend about his old car to find out if it was still 
               running reliably." 

        For a machine to understand this sentence, it has to figure out that the pronoun "it"
        is strongly related to the noun "car" occurring earlier in the sentence.  A neural 
        network with self-attention would be able to do that and would therefore be able to
        answer the question: "What is the current state of Charlie's old car?" assuming that
        system already knows that "my friend" in the sentence is referring to Charlie.  For
        another example, consider the following Spanish translation for the above sentence:

              "Yo estaba hablando con mi amigo sobre su viejo coche para averiguar si 
               todavía funcionaba de manera confiable."     

        In Spanish-to-English translation, the phrase "su viejo coche" could go into "his
        old car", "her old car", or "its old car". Choosing the correct form would require
        for the neural-network based translation system to have established the relationship 
        between the phrase "su viejo coche" and the phrase "mi amigo".  A neural network
        endowed with self-attention would be able to do that.

        While self-attention allows a neural network to establish the sort of intra-sentence
        word-level and phrase-level relationships mentioned above, a seq2seq translation
        network also needs what's known as cross-attention.  Cross attention means discovering
        what parts of a sentence in the source language are relevant to the production of 
        each word in the target language. In the English-to-Spanish translation example 
        mentioned above, the Spanish word "averiguar" has several nuances in what it means: 
        it can stand for "to discover", "to figure out", "to find out", etc.  It is obvious 
        that for successful translation the neural network would need to know that the
        context for "averiguar"  --- a conversation with a friend --- requires that the
        nuance "to find out" would be most appropriate translation to use for "averiguar".
        Along the same lines, in English-to-Spanish translation, ordinarily the English 
        word "running" would be translated into the gerund "corriendo" in Spanish, but
        on account of the context "car", a more appropriate Spanish translation would
        be based on the verb "funcionar".
       
        ClassPath:  Transformers  ->  TransformerFG
        """
        def __init__(self, dl_studio, dataroot, data_archive, max_seq_length, embedding_size, num_warmup_steps, 
                                                                      optimizer_params,  save_model_end_of_epoch):
            super(Transformers.TransformerFG, self).__init__()
            self.dl_studio = dl_studio
            self.dataroot  = dataroot
            self.data_archive = data_archive
            self.max_seq_length = max_seq_length
            self.embedding_size = embedding_size
            self.num_warmup_steps = num_warmup_steps
            self.optimizer_params = optimizer_params
            self.save_model_end_of_epoch = save_model_end_of_epoch
            f = gzip.open(dataroot + data_archive, 'rb')
            dataset = f.read()
            dataset,vocab_en,vocab_es = pickle.loads(dataset, encoding='latin1')
            ## the dataset is a dictionary whose keys are the integer index values associated with each entry
            self.training_corpus  = list(dataset.values())
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


        def sentence_with_words_to_ints(self, sentences, lang):
            """
            First consider the case when the parameter 'sentences' shown above is a single 
            sentence:

            This function returns a tensor of integers for the input sentence, with each integer
            being the index position of the corresponding word in the sentence in an alphabetized
            sort of the vocabulary.  The length of this sequence of integers will always be
            self.max_seq_length regardless of the actual length of the sentence.  Consider the
            case when the input sentence is 

                          [SOS they live near the school EOS]

            Recall from the dataset construction that the two tokens SOS and EOS are inserted at
            the beginning of the alphabetized sort of the actual vocabulary. So the integer 
            representation of the SOS token is always 0 and that of SOS always 1.  For the
            sentence shown above, the tensor returned by this function will be

                         tensor([[0, 10051, 5857, 6541, 10027, 8572, 1, 1, 1, 1]])

            where the additional 1's at the end represent a padding of the input sentence with
            the EOS token so that its length is always max_seq_length.

            As for why the object returned by this function is a tensor of two axis, the reason
            for that is to allow for using batches of sentences, as opposed to just one sentence
            at a time.  Suppose the batch_size is 2, the argument for 'sentences' will now be
            like

                    [[SOS they live near the school EOS],  [SOS answer the question EOS]]

            In this case, the tensor returned by this function will look like

                 tensor([[0, 10051,  5857,  6541, 10027, 8572,   1,   1,   1,   1],
                         [0,   423, 10027,  7822,     1,    1,   1,   1,   1,   1]])

            During training, similar tensors are constructed for the Spanish sentences. The integer
            indexes in those tensors serve as targets in the nn.NLLLoss based loss function.
            """
            sentence_to_ints = torch.ones(len(sentences), self.max_seq_length, dtype=torch.long)
            for i in range(len(sentences)):
                words = sentences[i].split(' ')
                for j,word in enumerate(words):
                    sentence_to_ints[i,j] = self.en_vocab_dict[word] if lang=="en" else self.es_vocab_dict[word]
            return sentence_to_ints


        class EmbeddingsGenerator(nn.Module):
            """
            A sentence begins its life as a tensor of word_vocab_index integers, with each integer 
            representing the index of a word in the alphabetized vocabulary for the language. This
            integer based representation of a sentence, created by the function 'sentence_with_words_to_ints()'
            function shown above, is fed to an instance of EmbeddingsGenerator in Line (B) below.
            The purpose of the EmbeddingsGenerator class is to translate the word_vocab_index based 
            representation to a learned word-embedding-vector based representation. The learning of
            the embedding vector for each word of the vocabulary takes place through the "nn.Embedding"
            layer that is created in Line (A). 

            ClassPath:  Transformers   ->   TransformerFG   ->   EmbeddingsGenerator
            """
            def __init__(self, xformer, lang, embedding_size):
                super(Transformers.TransformerFG.EmbeddingsGenerator, self).__init__()
                self.vocab_size = xformer.vocab_en_size if lang=="en" else xformer.vocab_es_size
                self.embedding_size = embedding_size                                             
                self.max_seq_length = xformer.max_seq_length                                                     
                self.embed = nn.Embedding(self.vocab_size, embedding_size)                                                 ## (A)
        
            def forward(self, sentence_tensor):                                                                            ## (B)
                ## Let's say your batch_size is 4 and that each sentence has a max_seq_length of 10.
                ## The sentence_tensor argument will now be of shape [4,10].  If the embedding size is
                ## is 512, the following call will return a tensor of shape [4,10,512)
                word_embeddings = self.embed(sentence_tensor)
                position_coded_word_embeddings = self.apply_positional_encoding( word_embeddings )
                return position_coded_word_embeddings


            def apply_positional_encoding(self, sentence_tensor):
                """
                Positional encoding is applied to the embedding-vector representation of a sentence 
                tensor that is returned by EmbeddingsGenerator. Positional encoding is applied by 
                explicitly calling the function 'apply_positional_encoding()' and supplying to it as 
                argument the tensor returned by the forward() of EmbeddingsGenerator.  

                The main goal of positional encoding is to sensitize a neural network to the position 
                of each word in a sentence and also to each embedding-vector cell for each word by 
                first constructing an array of floating-point values as described in this comment 
                section and then adding that array of numbers to the sentence tensor returned by the 
                forward() of EmbeddingsGenerator. The alternating columns of this 2D array are filled 
                using sine and cosine functions whose periodicities vary with the column index as shown 
                below.  Note that whereas the periodicities are column-specific, the values of the 
                sine and cosine functions are word-position-specific.  In the depiction shown below,
                each row is an embedding vector for a specific word:
                                       
                                        
                                       along the embedding vector index  i -->
    
                            i=0  |  i=1  |  i=2 |  i=3 |        ...........               | i=511
                          -------------------------------------------------------------------------
            w   pos=0            |       |      |      |                                  |
            o             -------------------------------------------------------------------------
            r   pos=1            |       |      |      |                                  |
            d             -------------------------------------------------------------------------
                pos=2            |       |      |      |                                  |
            i             -------------------------------------------------------------------------                     
            n                                              .
            d                                              .
            e                                              .
            x             -------------------------------------------------------------------------
                pos=9            |       |      |      |                                  |
                          -------------------------------------------------------------------------
                                            |      |
                                            |      |
                                            |      |_________________
                                            |                        |
                                            |                        |
                                            V                        V
                                           pos                      pos                                                    ## (D)
                                  sin( ------------- )     cos( ------------- )
                                        100^{2i/512}             100^{2i/512}                                              ## (E)
    

                To further explain the idea of positional encoding, let's say we have a sentence 
                consisting of 10 words and an embedding size of 512.  For such a sentence, 
                the forward() of EmbeddingsGenerator will return a tensor of shape [10,512].  So the 
                array of positional-encoding numbers we need to construct will also be of shape 
                [10,512].  We need to fill the alternating columns of this [10,512] array with sin() 
                and cos() values as shown above.  To appreciate the significance of these values, first 
                note that one period of a sinusoidal function like sin(pos) is 2*pi with respect to the 
                word index pos. [That would amount to only about six words. That is, there would only 
                be roughly six words in one period if we just use sin(pos) above.]  On the other hand, 
                one period of a sinusoidal function like sin(pos/k) is 2*pi*k with respect to the word 
                index pos.  So if k=100, we have a periodicity of about 640 word positions along the 
                pos axis.  The important point is that every individual column in the 2D pattern shown
                above gets a unique periodicity and that the alternating columns are characterized by 
                sine and cosine functions.
                """
                position_encodings = torch.zeros_like( sentence_tensor,  dtype=float )
                ## Calling unsqueeze() with arg 1 causes the "row tensor" to turn into a "column tensor"
                ##    which is needed in the products in lines (F) and (G). We create a 2D pattern by 
                ##    taking advantage of how PyTorch has overloaded the definition of the infix '*' 
                ##    tensor-tensor multiplication operator.  It in effect creates an output-product of
                ##    of what is essentially a column vector with what is essentially a row vector.
                word_positions = torch.arange(0, self.max_seq_length).unsqueeze(1)            
                div_term =  1.0 / (100.0 ** ( 2.0 * torch.arange(0, self.embedding_size, 2) / float(self.embedding_size) ))
                position_encodings[:, :, 0::2] =  torch.sin(word_positions * div_term)                                     ## (F)
                position_encodings[:, :, 1::2] =  torch.cos(word_positions * div_term)                                     ## (G)
                return sentence_tensor + position_encodings

        ###%%%
        #####################################   Encoder Code TransformerFG  ###############################################

        class MasterEncoder(nn.Module):
            """
            The purpose of the MasterEncoder is to invoke a stack of BasicEncoder instances on a
            source-language sentence tensor. The output of each BasicEncoder is fed as input to the 
            next BasicEncoder in the cascade, as illustrated in the loop in Line (B) below.  The stack
            of BasicEncoder instances is constructed in Line (A).

            ClassPath:  Transformers   ->   TransformerFG   ->   MasterEncoder
            """
            def __init__(self, dls, xformer, how_many_basic_encoders, num_atten_heads):
                super(Transformers.TransformerFG.MasterEncoder, self).__init__()
                self.max_seq_length = xformer.max_seq_length
                self.basic_encoder_arr = nn.ModuleList( [xformer.BasicEncoder(dls, xformer,
                                                          num_atten_heads) for _ in range(how_many_basic_encoders)] )      ## (A)
            def forward(self, sentence_tensor):
                out_tensor = sentence_tensor
                for i in range(len(self.basic_encoder_arr)):                                                               ## (B)
                    out_tensor = self.basic_encoder_arr[i](out_tensor)
                return out_tensor


        class BasicEncoder(nn.Module):
            """
            The BasicEncoder in TransformerFG consists of a layer of self-attention (SA) followed
            by a purely feed-forward layer (FFN).  The job of the SA layer is for the network to
            figure out what parts of an input sentence are relevant to what other parts of the
            same sentence in the process of learning how to translate a source-language sentence into
            a target-language sentence. The output of SA goes through FFN and the output of FFN becomes
            the output of the BasicEncoder.  To mitigate the problem of vanishing gradients in the FG
            transformer design, the output of each of the two components --- SA and FFN --- is subject
            to LayerNorm and a residual connection used that wraps around both the component and the
            LayerNorm which follows. Deploying a stack of BasicEncoder instances becomes easier if the 
            output tensor from a BasicEncoder has the same shape as its input tensor.  

            The SelfAttention layer mentioned above consists of a number of AttentionHead instances, 
            with each AttentionHead making an independent assessment of what to say about the 
            inter-relationships between the different parts of an input sequence. It is the embedding
            axis that is segmented out into disjoint slices for each AttentionHead instance.The 
            calling SelfAttention layer concatenates the outputs from all its AttentionHead instances 
            and presents the concatenated tensor as its own output.

            ClassPath:  Transformers  ->  TransformerFG  ->   BasicEncoder
            """
            def __init__(self, dls, xformer, num_atten_heads):
                super(Transformers.TransformerFG.BasicEncoder, self).__init__()
                self.dls = dls
                self.embedding_size = xformer.embedding_size                                             
                self.max_seq_length = xformer.max_seq_length                                                     
                self.num_atten_heads = num_atten_heads
                self.self_attention_layer = xformer.SelfAttention(dls, xformer, num_atten_heads)                           ## (A)
                self.norm1 = nn.LayerNorm(self.embedding_size)                                                             ## (B)
                ## What follows are the linear layers for the FFN (Feed Forward Network) part of a BasicEncoder
                self.W1 =  nn.Linear( self.max_seq_length * self.embedding_size, self.max_seq_length * 2 * self.embedding_size )
                self.W2 =  nn.Linear( self.max_seq_length * 2 * self.embedding_size, self.max_seq_length * self.embedding_size ) 
                self.norm2 = nn.LayerNorm(self.embedding_size)                                                             ## (C)

            def forward(self, sentence_tensor):
                sentence_tensor = sentence_tensor.float()
                self_atten_out = self.self_attention_layer(sentence_tensor).to(self.dls.device)                            ## (D)
                normed_atten_out = self.norm1(self_atten_out + sentence_tensor)                                            ## (E)               
                basic_encoder_out =  nn.ReLU()(self.W1( normed_atten_out.view(sentence_tensor.shape[0],-1) ))              ## (F)
                basic_encoder_out =  self.W2( basic_encoder_out )                                                          ## (G)
                basic_encoder_out = basic_encoder_out.view(sentence_tensor.shape[0], self.max_seq_length, self.embedding_size ) 
                ## for the residual connection and layer norm for FC layer:
                basic_encoder_out =  self.norm2(basic_encoder_out  + normed_atten_out)                                     ## (H)
                return basic_encoder_out



        ###%%%
        ####################################  Self Attention Code TransformerFG ###########################################

        class SelfAttention(nn.Module):
            """
            As described in the doc section of the BasicEncoder class, in each BasicEncoder you have 
            a layer of SelfAttention followed by a Feed Forward Network (FFN).  The SelfAttention layer 
            concatenates the outputs from all AttentionHead instances and presents that concatenated 
            output as its own output. If the input sentence consists of W words and if the embedding 
            size is M, the sentence_tensor at the input to forward() in Line B below will be of shape 
            [B,W,M] where B is the batch size.  This tensor is sliced off into num_atten_heads sections
            along the embedding axis and each slice shipped off to a different instance of AttentionHead. 
            Therefore, the shape what is seen by each AttentionHead is [B,W,qkv_size] where qkv_size 
            equals  "M // num_atten_heads".  The slicing of the sentence tensor, shipping off of each 
            slice to an AttentionHead instance, and the concatenation of the results returned by the 
            AttentionHead instances happens in the loop in line (C).

            ClassPath:  Transformers  ->  TransformerFG  ->   SelfAttention
            """
            def __init__(self, dls, xformer, num_atten_heads):
                super(Transformers.TransformerFG.SelfAttention, self).__init__()
                self.dl_studio = dls
                self.max_seq_length = xformer.max_seq_length                                                     
                self.embedding_size = xformer.embedding_size
                self.num_atten_heads = num_atten_heads
                self.qkv_size = self.embedding_size // num_atten_heads
                self.attention_heads_arr = nn.ModuleList( [xformer.AttentionHead(dls, self.max_seq_length, 
                                        self.qkv_size, num_atten_heads)  for _ in range(num_atten_heads)] )                ## (A)

            def forward(self, sentence_tensor):                                                                            ## (B)
                concat_out_from_atten_heads = torch.zeros( sentence_tensor.shape[0], self.max_seq_length, 
                                                                      self.num_atten_heads * self.qkv_size).float()
                for i in range(self.num_atten_heads):                                                                      ## (C) 
                    sentence_embed_slice = sentence_tensor[:, :, i * self.qkv_size : (i+1) * self.qkv_size]
                    concat_out_from_atten_heads[:, :, i * self.qkv_size : (i+1) * self.qkv_size] =          \
                                                                   self.attention_heads_arr[i](sentence_embed_slice)   
                return concat_out_from_atten_heads


        class AttentionHead(nn.Module):
            """
            In the explanation that follows, I will use the phrase "embedding slice" for the slice 
            of the embedding axis of a sentence tensor that is supplied to an instance of 
            AttentionHead.

            An AttentionHead (AH) instance does its job by first matrix-multiplying its assigned 
            embedding slice FOR EACH WORD in the input sentence by the following three matrices:

            -- a matrix Wq of size "qkv_size x qkv_size" to produce the query Vector q of size
               qkv_size at each word position in the input sentence.
      
            -- a matrix Wk, also of size "qkv_size x qkv_size", to produce a key vector k of 
               size qkv_size at each word position in the input sentence.

            -- a matrix Wv, also of size "qkv_size x qkv_size", to produce a value vector v of 
               size qkv_size at each word position in the input sentence. 

            Since in reality what the AttentionHead sees is a packing of the embedding slices
            for all the W words in an input sentence in the form of a tensor X of shape 
            [B,W,qkv_size], instead of learning the matrices Wq, Wk, Wv for the individual 
            words, it learns the tensors WQ, WK, and WV for all the words in the input sentence 
            simultaneously. Ignoring the batch axis for a moment, each of these tensors will 
            be of shape  [W, qkv_size, qkv_size] where again W is the number of words, and qkv_size 
            the size of both the embedding slice, and the size of the query, key, and value 
            vectors mentioned above.  We can mow write the following equations for the Query Q, 
            Key K, and Value V tensors calculated by an AttentionHead instance:

                 Q  =   X . WQ            K  =  X . WK             V  =  X . WV                                            ## (A)

            where the input X is of shape [W, qkv_size] and the tensors Q, K, and V tensors 
            are also of shape [W, qkv_size] --- ignoring again the batch axis.

            Going back to the individual word based query, key, and value vectors, the basic 
            idea in self-attention is to first take the dot-product of EACH query-vector q_i 
            with EVERY key vector k_j in the input sentence (i and j are word indexes).  So if 
            you have W words in an input sentence, at EACH word position in the input sentence, 
            you will have W of these dot-product  "q_i k'_j"  scalar values where  k'_j  stands 
            for the transpose of the key vector at the j^th position in the sentence.

            From all the W dot-products at each word position in a W-word input sentence, 
            calculated as explained above, you want to subject each W-element dot product 
            to a nn.Softmax normalization to yield a W-element list of probabilities.  

            To extend the above explanation to operations with the sentence based Q, K, 
            and V tensors, what's interesting  is that a matrix-matrix dot-product of the 
            Q and K tensors directly carries out all the dot-products at each word position
            in the input sentence.  Since Q and K are each of shape [W,qkv_size] for a W-word 
            sentence, the inner-product  "Q . K^T"  is of shape WxW, whose first W-element
            row contains the values obtained by taking the dot-product of the first-word 
            query vector q_1 with each of the k_1, k_2, k_3, ..., k_W key vectors for each
            of the W words in the sentence.  The second row of  "Q . K^T" will likewise 
            represent the dot product of the second query vector with every key vector, and
            so on.

            In the code shown below, all the dot-products mentioned above are calculated in
            line (N).  Next, as shown in line (O) we apply the nn.Softmax normalization to 
            each row of the WxW sized "Q . K^T" dot-products calculated in line (N). The 
            resulting WxW matrix is then used to multiply the  "W x qkv_size" matrix V
            as shown in line (V).  The operations carried out in lines (M) through (Q) of
            the code shown below can be expressed more compactly as:
                           
                                  nn.Softmax(Q .  K^T) 
                      Z   =      ----------------------- . V
                                            sqrt(M)

            At this point, the shape of Z will be "W x qkv_size"  --- ignoring again the
            batch axis.  This is the shape of the data object returned by each AttentionHead
            instance.

            ClassPath:  Transformers   ->   TransformerFG   ->   AttentionHead

            """
            def __init__(self, dl_studio, max_seq_length, qkv_size, num_atten_heads):
                super(Transformers.TransformerFG.AttentionHead, self).__init__()
                self.dl_studio = dl_studio
                self.qkv_size = qkv_size
                self.max_seq_length = max_seq_length
                self.WQ =  nn.Linear( max_seq_length * self.qkv_size, max_seq_length * self.qkv_size )                     ## (B)
                self.WK =  nn.Linear( max_seq_length * self.qkv_size, max_seq_length * self.qkv_size )                     ## (C)
                self.WV =  nn.Linear( max_seq_length * self.qkv_size, max_seq_length * self.qkv_size )                     ## (D)
                self.softmax = nn.Softmax(dim=1)                                                                           ## (E)

            def forward(self, sent_embed_slice):           ## sent_embed_slice == sentence_embedding_slice                 ## (F)
                Q = self.WQ( sent_embed_slice.reshape(sent_embed_slice.shape[0],-1).float() )                              ## (G)
                K = self.WK( sent_embed_slice.reshape(sent_embed_slice.shape[0],-1).float() )                              ## (H)
                V = self.WV( sent_embed_slice.reshape(sent_embed_slice.shape[0],-1).float() )                              ## (I)
                Q = Q.view(sent_embed_slice.shape[0], self.max_seq_length, self.qkv_size)                                  ## (J)
                K = K.view(sent_embed_slice.shape[0], self.max_seq_length, self.qkv_size)                                  ## (K)
                V = V.view(sent_embed_slice.shape[0], self.max_seq_length, self.qkv_size)                                  ## (L)
                A = K.transpose(2,1)                                                                                       ## (M)
                QK_dot_prod = Q @ A                                                                                        ## (N)
                rowwise_softmax_normalizations = self.softmax( QK_dot_prod )                                               ## (O)
                Z = rowwise_softmax_normalizations @ V                                                                     ## (P)
                coeff = 1.0/torch.sqrt(torch.tensor([self.qkv_size]).float()).to(self.dl_studio.device)                    ## (Q)
                Z = coeff * Z                                                                                              ## (R)
                return Z


        ###%%%
        ####################################  Cross Attention Code TransformerFG  #########################################

        class CrossAttention(nn.Module):
            """
            To understand the implementation of cross attention, a good starting point would be to 
            go through the doc comment block provided for the SelfAttention and AttentionHead
            classes.  Whereas self-attention consists of taking dot products the query vectors for
            the individual words in a sentence with the key vectors for all the words in order to
            discover the inter-word relevancies in a sentence, in cross-attention we take the dot 
            products of the query vectors for the individual words in the target-language sentence
            with the key vectors at the output of the master encoder for a given source-language
            sentence.

            Let X_enc represent the tensor at the output of the MasterEncoder.  Its shape will be 
            the same as that of the source sentence supplied to the MasterEncoder instance.  If W is
            the number of words in a sentence (in either language), the X tensor that is input into 
            the MasterEncoder will be of shape [B,W,M] where B is the batch size, and M the size of the 
            embedding vectors for the words. Therefore, the shape of the output of the MasterEncoder, 
            X_enc, is also [B,W,M]. Now let X_target represent the tensor form of the corresponding
            target language sentences. Its shape will also be [B,W,M].

            The ideas of CrossAttention is to ship off the embedding-axis slices of the X_enc and 
            X_target tensors to CrossAttentionHead instances for the calculation of the dot products 
            and, subsequently, for the output of the dot produces to modify the Value vectors as 
            explained in the doc section of the next class.

            ClassPath:  Transformers  ->  TransformerFG  ->   CrossAttention
            """
            def __init__(self, dls, xformer, num_atten_heads):
                super(Transformers.TransformerFG.CrossAttention, self).__init__()
                self.dl_studio = dls
                self.max_seq_length = xformer.max_seq_length
                self.embedding_size = xformer.embedding_size
                self.num_atten_heads = num_atten_heads
                self.qkv_size = self.embedding_size // num_atten_heads
                self.attention_heads_arr = nn.ModuleList( [xformer.CrossAttentionHead(dls, self.max_seq_length, 
                                                 self.qkv_size, num_atten_heads)  for _ in range(num_atten_heads)] )              
            def forward(self, basic_decoder_out, final_encoder_out):                                                     
                concat_out_from_atten_heads = torch.zeros( basic_decoder_out.shape[0], self.max_seq_length, 
                                                                        self.num_atten_heads * self.qkv_size).float()
                for i in range(self.num_atten_heads):                                                                    
                    basic_decoder_slice = basic_decoder_out[:, :, i * self.qkv_size : (i+1) * self.qkv_size]
                    final_encoder_slice = final_encoder_out[:, :, i * self.qkv_size : (i+1) * self.qkv_size]
                    concat_out_from_atten_heads[:, :, i * self.qkv_size : (i+1) * self.qkv_size] =          \
                                            self.attention_heads_arr[i](basic_decoder_slice, final_encoder_slice)    
                return concat_out_from_atten_heads


        class CrossAttentionHead(nn.Module):
            """
            CrossAttentionHead works the same as the regular AttentionHead described earlier, 
            except that now, in keeping with the explanation for the CrossAttention class, the dot 
            products involve the query vector slices from the target sequence and the key vector 
            slices from the MasterEncoder output for the source sequence.  The dot products 
            eventually modify the value vector slices that are also from the MasterEncoder output
            for the source sequence.  About the word "slice" here, as mentioned earlier, what each 
            attention head sees is a slice along the embedding axis for the words in a sentence.

            If X_target and X_source represent the embedding-axis slices of the target sentence 
            tensor and the MasterEncoder output for the source sentences, each CrossAttentionHead
            will compute the following dot products:

                Q  =   X_target . WQ          K  =  X_source . WK           V  =  X_source . WV                            ## (A)

            Note that the Queries are derived from the target sentences, whereas the Keys and the
            Values come from the source sentences.  The operations carried out in lines (N)
            through (R) can be described more compactly as:

                           
                                 nn.Softmax(Q .  K^T)
                      Z   =      ---------------------  . V
                                        sqrt(M)

            ClassPath:  Transformers  ->  TransformerFG  ->   CrossAttention
            """  
            def __init__(self, dl_studio, max_seq_length, qkv_size, num_atten_heads):
                super(Transformers.TransformerFG.CrossAttentionHead, self).__init__()
                self.dl_studio = dl_studio
                self.qkv_size = qkv_size
                self.max_seq_length = max_seq_length
                self.WQ =  nn.Linear( max_seq_length * self.qkv_size, max_seq_length * self.qkv_size )                     ## (B)
                self.WK =  nn.Linear( max_seq_length * self.qkv_size, max_seq_length * self.qkv_size )                     ## (C)
                self.WV =  nn.Linear( max_seq_length * self.qkv_size, max_seq_length * self.qkv_size )                     ## (D)
                self.softmax = nn.Softmax(dim=1)                                                                           ## (E)

            def forward(self, basic_decoder_slice, final_encoder_slice):                                                   ## (F)
                Q = self.WQ( basic_decoder_slice.reshape(final_encoder_slice.shape[0],-1).float() )                        ## (G)
                K = self.WK( final_encoder_slice.reshape(final_encoder_slice.shape[0],-1).float() )                        ## (H)
                V = self.WV( final_encoder_slice.reshape(final_encoder_slice.shape[0],-1).float() )                        ## (I)
                Q = Q.view(final_encoder_slice.shape[0], self.max_seq_length, self.qkv_size)                               ## (J)
                K = K.view(final_encoder_slice.shape[0], self.max_seq_length, self.qkv_size)                               ## (K)
                V = V.view(final_encoder_slice.shape[0], self.max_seq_length, self.qkv_size)                               ## (L)
                A = K.transpose(2,1)                                                                                       ## (M)
                QK_dot_prod = Q @ A                                                                                        ## (N)
                rowwise_softmax_normalizations = self.softmax( QK_dot_prod )                                               ## (O)
                Z = rowwise_softmax_normalizations @ V                                                                     ## (P)
                coeff = 1.0/torch.sqrt(torch.tensor([self.qkv_size]).float()).to(self.dl_studio.device)                    ## (Q)
                Z = coeff * Z                                                                                              ## (R)
                return Z



        ###%%%
        #######################################   Decoder Code TransformerFG ##############################################

        class BasicDecoderWithMasking(nn.Module):
            """
            As with the basic encoder, while a basic decoder also consists of a layer of SelfAttention 
            followed by a Feedforward Network (FFN) layer, but now there is a layer of CrossAttention 
            interposed between the two.  The output from each of these three components of a basic 
            decoder passes through a LayerNorm layer. Additionally, you have a residual connection from 
            the input at each component to the output from the LayerNorm layer.

            An important feature of the BasicDecoder is the masking of the target sentences during the
            training phase in order to ensure that each predicted word in the target language depends only 
            on the target words that have been seen PRIOR to that point. This recursive backward dependency 
            is referred to as "autoregressive masking". In the implementation shown below, the masking is 
            initiated and its updates established by MasterDecoderWithMasking.

            ClassPath:  Transformers  ->  TransformerFG  ->   BasicDecoderWithMasking
            """
            def __init__(self, dls, xformer, num_atten_heads):
                super(Transformers.TransformerFG.BasicDecoderWithMasking, self).__init__()
                self.dls = dls
                self.embedding_size = xformer.embedding_size                                             
                self.max_seq_length = xformer.max_seq_length                                                     
                self.num_atten_heads = num_atten_heads
                self.qkv_size = self.embedding_size // num_atten_heads
                self.self_attention_layer = xformer.SelfAttention(dls, xformer, num_atten_heads)
                self.norm1 = nn.LayerNorm(self.embedding_size)
                self.cross_attn_layer = xformer.CrossAttention(dls, xformer, num_atten_heads)
                self.norm2 = nn.LayerNorm(self.embedding_size)
                ## What follows are the linear layers for the FFN (Feed Forward Network) part of a BasicDecoder
                self.W1 =  nn.Linear( self.max_seq_length * self.embedding_size, self.max_seq_length * 2 * self.embedding_size )
                self.W2 =  nn.Linear( self.max_seq_length * 2 * self.embedding_size, self.max_seq_length * self.embedding_size ) 
                self.norm3 = nn.LayerNorm(self.embedding_size)

            def forward(self, sentence_tensor, final_encoder_out, mask):   
                ## self attention
                masked_sentence_tensor = sentence_tensor
                if mask is not None:
                    masked_sentence_sentence = self.apply_mask(sentence_tensor, mask, self.max_seq_length, self.embedding_size)
                Z_concatenated = self.self_attention_layer(masked_sentence_tensor).to(self.dls.device)
                Z_out = self.norm1(Z_concatenated + masked_sentence_tensor)                     
                ## for cross attention
                Z_out2  = self.cross_attn_layer( Z_out, final_encoder_out).to(self.dls.device)
                Z_out2 = self.norm2( Z_out2 )
                ## for FFN:
                basic_decoder_out =  nn.ReLU()(self.W1( Z_out2.view(sentence_tensor.shape[0],-1) ))                  
                basic_decoder_out =  self.W2( basic_decoder_out )                                                    
                basic_decoder_out = basic_decoder_out.view(sentence_tensor.shape[0], self.max_seq_length, self.embedding_size )
                basic_decoder_out =  basic_decoder_out  + Z_out2 
                basic_decoder_out = self.norm3( basic_decoder_out )
                return basic_decoder_out

            def apply_mask(self, sentence_tensor, mask, max_seq_length, embedding_size):  
                out = torch.zeros(sentence_tensor.shape[0], max_seq_length, embedding_size).float().to(self.dls.device) 
                out[:,:,:len(mask)] = sentence_tensor[:,:,:len(mask)] 
                return out    


        class MasterDecoderWithMasking(nn.Module):
            """
            The primary job of the MasterDecoder is to orchestrate the invocation of a stack of BasicDecoders.
            The number of BasicDecoder instances used is a user-defined parameter.  The masking that is used
            in each BasicDecoder instance is set here by the MasterDecoder.  In Line (B), we define the 
            BasicDecoder instances needed. The linear layer in Line (C) is needed because what the decoder side
            produces must ultimately be mapped as a probability distribution over the entire vocabulary for the
            target language.  With regard to the data flow through the network, note how the mask is initialized
            in Line (D).  The mask is supposed to be a vector of one's that grows with the prediction for each
            output word. We start by setting it equal to just a single-element vector containing a single "1".
            Lines (E) and (F) declare the tensors that will store the final output of the master decoder. This
            final output consists of two tensors: One tensor holds the integer index to the target-language 
            vocabulary word where the output log-prob is maximum. [This index is needed at inference time to 
            output the words in the translation.]  The other tensor holds the log-probs over the target language
            vocabulary. The log-probs are produced by the nn.LogSoftmax in Line (L).

            ClassPath:  Transformers  ->  TransformerFG  ->   MasterDecoderWithMasking
            """
            def __init__(self, dls, xformer, how_many_basic_decoders, num_atten_heads):
                super(Transformers.TransformerFG.MasterDecoderWithMasking, self).__init__()
                self.dls = dls
                self.max_seq_length = xformer.max_seq_length
                self.embedding_size = xformer.embedding_size
                self.target_vocab_size = xformer.vocab_es_size                                                                   ## (A)
                self.basic_decoder_arr = nn.ModuleList([xformer.BasicDecoderWithMasking( dls, xformer,
                                                               num_atten_heads) for _ in range(how_many_basic_decoders)])        ## (B)
                ## Need the following layer because we want the prediction of each target word to be a probability 
                ## distribution over the target vocabulary. The conversion to probs would be done by the criterion 
                ## nn.CrossEntropyLoss in the training loop:
                self.out = nn.Linear(self.embedding_size, self.target_vocab_size)                                                ## (C)

            def forward(self, sentence_tensor, final_encoder_out):                                                   
                mask = torch.ones(1, dtype=int)                         ## initialize the mask                                   ## (D)
                ##  A tensor with two axes, one for the batch instance and the other for storing the predicted 
                ##  word ints for that batch instance. We initialize by filling the tensor with "EOS" tokens (==1).
                predicted_word_index_values = torch.ones(sentence_tensor.shape[0], self.max_seq_length,          
                                                                            dtype=torch.long).to(self.dls.device)                ## (E)
                ##  A tensor with two axes, one for the batch instance and the other for storing the log-prob 
                ##  of predictions for that batch instance. The log_probs for each predicted word are defined over
                ##  the entire target vocabulary:
                predicted_word_logprobs  = torch.zeros( sentence_tensor.shape[0], self.max_seq_length, 
                                                            self.target_vocab_size, dtype=float).to(self.dls.device)             ## (F)
                for mask_index in range(1, sentence_tensor.shape[1]):
                    masked_target_sentence = self.apply_mask(sentence_tensor, mask, self.max_seq_length, self.embedding_size)    ## (G)
                    ## out_tensor will start as just the first word, then two first words, etc.
                    out_tensor = masked_target_sentence                                                                          ## (H)
                    for i in range(len(self.basic_decoder_arr)):                                                                 ## (I)
                        out_tensor = self.basic_decoder_arr[i](out_tensor, final_encoder_out, mask)                              ## (J)
                    last_word_tensor = out_tensor[:,mask_index]                                                           
                    last_word_onehot = self.out(last_word_tensor.view(sentence_tensor.shape[0],-1))                              ## (K)
                    output_word_logprobs = nn.LogSoftmax(dim=1)(last_word_onehot)                                                ## (L)
                    _, idx_max = torch.max(output_word_logprobs, 1)                                                              ## (M)
                    predicted_word_index_values[:,mask_index] = idx_max                                                          ## (N)
                    predicted_word_logprobs[:,mask_index] = output_word_logprobs                                                 ## (O)
                    mask = torch.cat( ( mask, torch.ones(1, dtype=int) ) )                                                       ## (P)
                return predicted_word_logprobs, predicted_word_index_values                                                      ## (Q)

            def apply_mask(self, sentence_tensor, mask, max_seq_length, embedding_size):  
                out = torch.zeros_like(sentence_tensor).float().to(self.dls.device)
                out[:,:len(mask),:] = sentence_tensor[:,:len(mask),:] 
                return out    

        ###%%%
        ##############################  Training and Evaluation for TransformerFG  ########################################

        def save_encoder(self, encoder):
            "Save the trained encoder to a disk file"       
            torch.save(encoder.state_dict(), self.dl_studio.path_saved_model["encoder"])

        def save_decoder(self, decoder):
            "Save the trained decoder to a disk file"       
            torch.save(decoder.state_dict(), self.dl_studio.path_saved_model["decoder"])
        

        def run_code_for_training_TransformerFG(self, dls, master_encoder, master_decoder, display_train_loss=False):        
            """
            A particular feature of TransformerFG is how it is trained. Training a transformer as 
            conceptualized in the paper by Vaswani et al. requires a carefully designed warm-up stage 
            at the beginning in which you start with a very small learning rate that is gradually 
            increased to a set maximum value in a specific number of training iterations.  The 
            training is sensitive to both the maximum value chosen for the learning rate and the
            number of iterations used for the ramp-up to the maximum.  Choosing inappropriate
            values for these two parameters can make the training unstable.  I have used Yu-Hsiang 
            Huang's ScheduledOptim class for the scheduling of the learning rate.  This class is
            defined at the end of the TransformerFG section of this file.  What you see in Lines
            (A) and (B) are calls to the constructor of this class.  The definition of the 
            ScheduledOptim class is at the end of TransformerFG section of this file.

            Since we did not construct a dataloader by subclassing from torch.utils.data.DataLoader, we
            need the statements in Lines (E) through (I) to deal with the cases when the size of the
            training data is not an exact multiple of the batch size, etc.  The code shown in these
            lines ensures that every bit of the available training data is used even if that means 
            that the last batch will not have the expected number of training samples in it.

            Overall, the training consists of running the English/Spanish sentence pairs as generated 
            in Lines (H) or (I) through the MasterEncoder-MasterDecoder combo as shown in Lines (N) and
            (O).  We must first convert the words in these sentences into their int values in Lines (J) 
            and (K) and subsequently generate the embeddings for each word in Lines (L) and (M).

            For a given English sentence, at each word position in a max_seq_length of positions in 
            Spanish, the encoder-decoder combo generates a tensor of log probabilities (logprobs) over 
            the target vocabulary. That is what is returned in Line (Q). [The logprobs are generated by the
            final nn.LogSoftmax activation function in the decoder code.]  At each word position in the
            target language sequence, the nn.NLLLoss() criterion then picks the negative of that logprob 
            that corresponds to the index of the groundtruth target word as a measure of loss for that word 
            position.

            Regarding the loss function nn.NLLLoss used for training, note that using a combination
            of nn.LogSoftmax activation and nn.NLLLoss is the same thing as using nn.CrossEntropyLoss,
            which is the most commonly used loss function for solving classification problems. For a 
            neural network that is meant for solving a classification problem, the number of nodes in
            the output layer must equal the number of classes.  Applying nn.LogSoftmax activation to
            such a layer normalizes the values accumulated at those nodes so that they become a legal
            probability distribution over the classes.  Subsequently, calculating the nn.NLLLoss 
            means choosing the negative value at just that node which corresponds to the actual class 
            label of the input data.                                                                    
            """
            master_encoder.to(self.dl_studio.device)
            master_decoder.to(self.dl_studio.device)     
            embeddings_generator_en = self.EmbeddingsGenerator(self, 'en', self.embedding_size).to(self.dl_studio.device)
            embeddings_generator_es = self.EmbeddingsGenerator(self, 'es', self.embedding_size).to(self.dl_studio.device)
            beta1,beta2,epsilon = self.optimizer_params['beta1'], self.optimizer_params['beta2'], \
                                                                                           self.optimizer_params['epsilon']     
            master_encoder_optimizer = self.ScheduledOptim(optim.Adam(master_encoder.parameters(), betas=(beta1,beta2), eps=epsilon),  
                                                lr_mul=2, d_model=self.embedding_size, n_warmup_steps=self.num_warmup_steps)     ## (A)
            master_decoder_optimizer = self.ScheduledOptim(optim.Adam(master_decoder.parameters(), betas=(beta1,beta2), eps=epsilon),
                                                lr_mul=2, d_model=self.embedding_size, n_warmup_steps=self.num_warmup_steps)     ## (B)
            ##  Note that by default, nn.NLLLoss averages over the instances in a batch
            criterion = nn.NLLLoss()                                                                                   
            accum_times = []
            start_time = time.perf_counter()
            training_loss_tally = []
            self.debug = False
            print("")
            num_sentence_pairs = len(self.training_corpus)
            print("\n\nNumber of sentence pairs in the dataset: ", num_sentence_pairs)
            print("\nNo sentence is longer than %d words (including the SOS and EOS tokens)\n\n" % self.max_seq_length)
            batch_size = self.dl_studio.batch_size
            max_iters = len(self.training_corpus) // batch_size                                                                  ## (C)
            max_iters = max_iters if len(self.training_corpus) % batch_size == 0 else max_iters + 1                              ## (D)
            print("\n\nMaximum number of training iterations in each epoch: %d\n\n" % max_iters)
            for epoch in range(self.dl_studio.epochs):                                                                           ## (E)
                print("")
                random.shuffle(self.training_corpus)                                                                             ## (F)
                running_loss = 0.0
                for iter in range(max_iters):
                    if self.debug:
                        print("\n\n\n========================== starting batch indexed: %d ================================\n\n\n" % iter)
                    if (iter+1)*batch_size <= num_sentence_pairs:                                                                ## (G)
                        batched_pairs = self.training_corpus[iter*batch_size : (iter+1)*batch_size]                              ## (H)
                    else:
                        batched_pairs = self.training_corpus[iter*batch_size : ]                                                 ## (I)
                    if self.debug:                 ##  MUST use a batch size of 5 in your script for this to work
                        batched_pairs =  [['SOS i will kill him EOS', 'SOS le mataré EOS'], ['SOS what was that EOS', 'SOS qué era eso EOS'], ['SOS he soon left the new job EOS', 'SOS él dejó pronto el nuevo empleo EOS'], ['SOS i go into the city every day EOS', 'SOS yo voy a la ciudad todos los días EOS'], ['SOS she might come tomorrow EOS', 'SOS puede que ella venga mañana EOS']]
                    source_sentences = [pair[0] for pair in batched_pairs]
                    target_sentences = [pair[1] for pair in batched_pairs]
                    if self.debug:
                        print("\n\nfirst source sentence in batch: ", source_sentences[0])
                        print("\nfirst target sentence in batch: ", target_sentences[0])
                        print("\nlast source sentence in batch: ", source_sentences[-1])
                        print("\nlast target sentence in batch: ", target_sentences[-1])
                    en_tensor =  self.sentence_with_words_to_ints(source_sentences, 'en')                                        ## (J)
                    es_tensor =  self.sentence_with_words_to_ints(target_sentences, 'es')                                        ## (K)
                    en_tensor = en_tensor.to(self.dl_studio.device)
                    es_tensor = es_tensor.to(self.dl_studio.device)
                    en_sentence_tensor = embeddings_generator_en(en_tensor)                                                      ## (L)
                    es_sentence_tensor = embeddings_generator_es(es_tensor)                                                      ## (M)
                    master_encoder_optimizer.zero_grad()
                    master_decoder_optimizer.zero_grad()
                    master_encoder_output = master_encoder( en_sentence_tensor )                                                 ## (N)
                    predicted_word_logprobs, predicted_word_index_values = master_decoder(es_sentence_tensor, 
                                                                                                master_encoder_output)           ## (O)
                    loss = torch.tensor(0.0).to(self.dl_studio.device)                                                           ## (P)
                    for di in range(es_tensor.shape[1]):                                                                         ## (Q)
                        loss += criterion(predicted_word_logprobs[:,di], es_tensor[:,di])                                        ## (R)
                    loss.backward()                                                                                              ## (S)
                    master_encoder_optimizer.step_and_update_lr()                                                                ## (T)
                    master_decoder_optimizer.step_and_update_lr()                                                                ## (U)
                    loss_normed = loss.item() / es_tensor.shape[0]
                    running_loss += loss_normed
                    if iter % 200 == 199:    
                        avg_loss = running_loss / float(200)
                        training_loss_tally.append(avg_loss)
                        running_loss = 0.0
                        current_time = time.perf_counter()
                        time_elapsed = current_time-start_time
                        print("[epoch:%2d  iter:%4d  elapsed_time: %4d secs]     loss: %.4f" % (epoch+1, iter+1, time_elapsed,avg_loss)) 
                        accum_times.append(current_time-start_time)
                if self.save_model_end_of_epoch:
                    self.save_encoder(master_encoder)
                    self.save_decoder(master_decoder)
                    print("Model saved at the end of epoch %d" % (epoch+1))
            print("\nFinished Training\n")
            self.save_encoder(master_encoder)       
            self.save_decoder(master_decoder)       
            if display_train_loss:
                plt.figure(figsize=(10,5))
                plt.title("Training Loss vs. Iterations")
                plt.plot(training_loss_tally)
                plt.xlabel("iterations")
                plt.ylabel("training loss")
                plt.legend()
                plt.savefig("training_loss.png")
                plt.show()


        def run_code_for_evaluating_TransformerFG(self, master_encoder, master_decoder):
            """
            The main difference between the training code shown in the previous function and the
            evaluation code shown here is with regard to the input to MasterDecoder and how we
            process its output. As shown in the previous function, for the training loop, the
            input to MasterDecoder consists of the both the target sentence and the output of
            the MasterEncoder for the source sentence.  However, at inference time (that is, in
            the evaluation loop shown below), the target sentence at the input to the MasterDecoder
            is replaced by an encoding of a "starter stub" output sentence as defined in line (B).
            The main message conveyed by the stub in line (B) is that we want to start the 
            translation with the first work of the output as being the token "SOS".  The encoding
            for the stub is generated in lines (F) and (G).

            The second significant difference between the training and the testing code is 
            with regard to how we process the output of the MasterDecoder.  As you will recall
            from the docstring associated with MasterDecoder, it returns two things: (1) the
            predicted log probabilities (logprob) over the target vocabulary for every word 
            position in the target language; and (2) for each target-language word position, 
            the word_vocab_index at which the logprob is maximum.  The loss calculation in
            the training code was based on the former.  ON the other hand, as shown in line (H)
            below, it is the latter that lets us do the the translations in the target words
            in line (I).
            """
            master_encoder.load_state_dict(torch.load(self.dl_studio.path_saved_model['encoder']))
            master_decoder.load_state_dict(torch.load(self.dl_studio.path_saved_model['decoder']))
            master_encoder.to(self.dl_studio.device)
            master_decoder.to(self.dl_studio.device)     
            embeddings_generator_en = self.EmbeddingsGenerator(self, 'en', self.embedding_size).to(self.dl_studio.device)
            embeddings_generator_es = self.EmbeddingsGenerator(self, 'es', self.embedding_size).to(self.dl_studio.device)
            self.debug = False
            with torch.no_grad():
                for iter in range(20):
                    starter_stub_for_translation = ['SOS', 'EOS', 'EOS', 'EOS', 'EOS', 'EOS', 'EOS', 'EOS', 'EOS', 'EOS']      ## (A)
                    batched_pairs = random.sample(self.training_corpus, 1)                                                     ## (B)
                    source_sentences = [pair[0] for pair in batched_pairs]
                    target_sentences = [pair[1] for pair in batched_pairs]
                    if self.debug:
                        print("\n\nsource sentences: ", source_sentences)
                        print("\ntarget sentences: ", target_sentences)
                    en_sent_ints =  self.sentence_with_words_to_ints(source_sentences, 'en').to(self.dl_studio.device)         ## (C)
                    if self.debug:
                        print("\n\nsource sentence tensor: ", en_sent_ints)
                        print("\n\ntarget sentence tensor: ", es_sent_ints)
                    en_sentence_tensor = embeddings_generator_en(en_sent_ints).to(self.dl_studio.device)                       ## (D)
                    master_encoder_output = master_encoder( en_sentence_tensor )                                               ## (E)
                    starter_stub_as_ints = self.sentence_with_words_to_ints(
                                                  [" ".join(starter_stub_for_translation)], 'es').to(self.dl_studio.device)    ## (F)
                    starter_stub_tensor = embeddings_generator_es(starter_stub_as_ints)                                        ## (G)
                    _, predicted_word_index_values = master_decoder(starter_stub_tensor, master_encoder_output)                ## (H)
                    predicted_word_index_values = predicted_word_index_values[0].unsqueeze(1)
                    decoded_words = [self.es_index_2_word[predicted_word_index_values[di].item()] 
                                                                            for di in range(self.max_seq_length)]              ## (I)
                    output_sentence = " ".join(decoded_words)                                                                 
                    print("\n\n\nThe input sentence pair: ", source_sentences, target_sentences)                             
                    print("\nThe translation produced by TransformerFG: ", output_sentence)



        class ScheduledOptim():
            """
            For the scheduling of the learning rate during the warm-up phase of training TransformerFG, I 
            have borrowed the class shown below from the GitHub code made available by Yu-Hsiang Huang at:

                    https://github.com/jadore801120/attention-is-all-you-need-pytorch
            """
            def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
                self._optimizer = optimizer
                self.lr_mul = lr_mul
                self.d_model = d_model
                self.n_warmup_steps = n_warmup_steps
                self.n_steps = 0

            def step_and_update_lr(self):
                "Step with the inner optimizer"
                self._update_learning_rate()
                self._optimizer.step()
        
            def zero_grad(self):
                "Zero out the gradients with the inner optimizer"
                self._optimizer.zero_grad()
        
            def _get_lr_scale(self):
                d_model = self.d_model
                n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
                return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))
        
            def _update_learning_rate(self):
                ''' Learning rate scheduling per step '''
                self.n_steps += 1
                lr = self.lr_mul * self._get_lr_scale()
                for param_group in self._optimizer.param_groups:
                    param_group['lr'] = lr

    ##################################  END Definition of Inner Class TransformerFG  ####################################
    #####################################################################################################################
    #####################################################################################################################



    ###%%%
    ###############################  Start Definition of Inner Class TransformerPreLN  ##################################
    class TransformerPreLN(nn.Module):             
        """
        TransformerPreLN stands for "Transformer Pre Layer Norm".

        As with the TransformerFG class, the goal of TransformerPreLN is to serve as an 
        instructional aid in understanding the basic concepts of attention-based learning with
        neural networks.

        At this moment, there is only a small (but crucial) difference between the implementation 
        for TransformerFG shown previously in this file and the code for TransformerPreLN that 
        follows.  This difference is based on the observation made by Xiong et al. in their paper 
        "On Layer Normalization in the Transformer Architecture" that using LayerNorm after each 
        residual connection in the Vaswani et al. design for the transformers contributed 
        significantly to the stability problems with training.  This they attributed as the main
        reason for why TransformerFG needs a warm-up phase in learning.  To get around this 
        problem, Xiong et al. advocated changing the point at which the LayerNorm is invoked in 
        the original design.  In the two keystroke diagrams shown below, the one at left is for
        the encoder layout in TransformerFG and the one on right for the same in TransformerPreLN.

        In both approaches, the fundamental components of the basic encoder remain the same: a
        multi-head attention layer followed by an FFN.  However, in TransformerFG, each of these
        two components is followed with a residual connection that wraps around the component
        and the residual connection is followed by LayerNorm. On the other hand, in 
        TransformerPreLN, the LayerNorm for each component is used prior to the component and 
        the residual connection wraps around both the LayerNorm layer and the component, as shown 
        at right below.




         output of basic encoder                            output of basic encoder
                  ^                                                  ^
                  |                                                  |
                  |                                                  + <---------  residual
                  |                                                  |           | connection
             -----------                                        -----------      |    
            | LayerNorm |                                      |           |     |
             -----------                                       |    FFN    |     |
                  ^                                            |           |     |
                  |             residual                        -----------      |
                  + <---------  connection                           ^           |
                  |           |                                      |           |   
             -----------      |                                 -----------      |
            |           |     |                                | LayerNorm |     |   
            |    FFN    |     |                                 -----------      |
            |___________|     |                                      ^           |
                  ^           |                                      |           |
                  |           |                                      |-----------
                  |-----------                                       |
                  |                                                  + <---------  residual
             -----------                                             |           | connection
            | LayerNorm |                                     ---------------    |
             -----------                                     |               |   |
                  |             residual                     |   Multi-Head  |   |
                  + <---------  connection                   |   Attention   |   |   
                  |           |                              |               |   |  
           ---------------    |                               ---------------    |
          |               |   |                                      ^           |          
          |   Multi-Head  |   |                                      |           |
          |   Attention   |   |                                 -----------      |
          |               |   |                                | LayerNorm |     |
           ---------------    |                                 -----------      |
                  ^           |                                      ^           |
                  |           |                                      |           |                        
                  |-----------                                       |-----------
                  |                                                  |
                  |                                                  |
        input sentence tensor                               input sentence tensor
                 or                                                  or  
   output from previous basic encoder                output from previous basic encoder


           TransformerFG                                      TransformerPreLN


        While the above explanation about the difference between TransformerFG and 
        TransformerPreLN specifically addresses the basic encoder, the same difference 
        carries over to the decoder side.  Inside each basic decoder, you will have three 
        invocations of LayerNorm, one before the self-attention layer, another one before
        the call to cross-attention and, finally, one more application of LayerNorm prior
        to the FFN layer.
 
        ClassPath:  Transformers  ->  TransformerPreLN
        """
        def __init__(self, dl_studio, dataroot, data_archive, max_seq_length, embedding_size, save_model_end_of_epoch):
            super(Transformers.TransformerPreLN, self).__init__()
            self.dl_studio = dl_studio
            self.dataroot  = dataroot
            self.data_archive = data_archive
            self.max_seq_length = max_seq_length
            self.embedding_size = embedding_size
            self.save_model_end_of_epoch = save_model_end_of_epoch
            f = gzip.open(dataroot + data_archive, 'rb')
            dataset = f.read()
            dataset,vocab_en,vocab_es = pickle.loads(dataset, encoding='latin1')
            ##  The dataset is a dictionary whose keys are the integer index values associated with each entry.
            ##  We are only interested in the values themselves:
            self.training_corpus  = list(dataset.values())
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


        def sentence_with_words_to_ints(self, sentences, lang):
            """
            First consider the case when 'sentences' is a single sentence:

            This function returns a tensor of integers for the input sentence, with each integer
            being the index position of the corresponding word in the sentence in an alphabetized
            sort of the vocabulary.  The length of this sequence of integers will always be
            self.max_seq_length regardless of the actual length of the sentence.  Consider the
            case when the input sentence is 

                          [SOS they live near the school EOS]

            Recall from the dataset construction that the two tokens SOS and EOS are inserted at
            the beginning of the alphabetized sort of the actual vocabulary. So the integer 
            representation of the SOS token is always 0 and that of SOS always 1.  For the
            sentence shown above, the tensor returned by this function will be

                         tensor([[0, 10051, 5857, 6541, 10027, 8572, 1, 1, 1, 1]])

            where the additional 1's at the end represent a padding of the input sentence with
            the EOS token so that its length is always max_seq_length.

            As for why the object returned by this function is a tensor of two axis, the reason
            for that is to allow for using batches of sentences, as opposed to just one sentence
            at a time.  Suppose the batch_size is 2, the argument for 'sentences' will now be
            like

                    [[SOS they live near the school EOS],  [SOS answer the question EOS]]

            In this case, the tensor returned by this function will look like

                 tensor([[0, 10051,  5857,  6541, 10027, 8572,   1,   1,   1,   1],
                         [0,   423, 10027,  7822,     1,    1,   1,   1,    1,  1]])

            During training, similar tensors are constructed for the Spanish sentences. The integer
            indexes in those tensors serve as targets in the nn.NLLLoss based loss function.
            """
            sentence_tensor = torch.ones(len(sentences), self.max_seq_length, dtype=torch.long)
            for i in range(len(sentences)):
                words = sentences[i].split(' ')
                for j,word in enumerate(words):
                    sentence_tensor[i,j] = self.en_vocab_dict[word] if lang=="en" else self.es_vocab_dict[word]
            return sentence_tensor


        class EmbeddingsGenerator(nn.Module):
            """
            A sentence begins its life as a tensor of word_vocab_index integers, with each integer 
            representing the index of a word in the alphabetized vocabulary for the language. This
            integer based representation of a sentence, created by the function 'sentence_with_words_to_ints()'
            function shown above, is fed to an instance of EmbeddingsGenerator in Line B below.
            The purpose of the EmbeddingsGenerator class is to translate the word_vocab_index based 
            representation to a learned word-embedding-vector based representation. The learning of
            the embedding vector for each word of the vocabulary takes place through the "nn.Embedding"
            layer that is created in Line A. Line C shows us assembling the sentence tensor one word
            at a time translated into its embedding vector.

            ClassPath:  Transformers  ->  TransformerPreLN  ->  EmbeddingsGenerator
            """
            def __init__(self, xformer, lang, embedding_size):
                super(Transformers.TransformerPreLN.EmbeddingsGenerator, self).__init__()
                self.vocab_size = xformer.vocab_en_size if lang=="en" else xformer.vocab_es_size
                self.embedding_size = embedding_size                                             
                self.max_seq_length = xformer.max_seq_length                                                     
                self.embed = nn.Embedding(self.vocab_size, embedding_size)                                                 ## (A)
        
            def forward(self, sentence_as_ints):                                                                           ## (B)
                ## Let's say your batch_size is 4 and that each sentence has a max_seq_length of 10.
                ## The sentence_tensor argument will now be of shape [4,10].  If the embedding size is
                ## is 512, the following call will return a tensor of shape [4,10,512)
                word_embeddings = self.embed(sentence_as_ints)
                position_coded_word_embeddings = self.apply_positional_encoding( word_embeddings )
                return position_coded_word_embeddings


            def apply_positional_encoding(self, sentence_tensor):
                """
                Positional encoding is applied to the embedding-vector representation of a sentence 
                tensor that is returned by a callable instance of EmbeddingsGenerator. [Positional 
                encoding is applied by explicitly calling the function 'apply_positional_encoding()'
                and supplying to it as argument the tensor returned by the forward() of 
                EmbeddingsGenerator.  The main goal of positional encoding is to sensitize a
                neural network to the position of each word in a sentence and also to each
                embedding-vector cell for each word by first constructing an array of floating-point
                values as described in this comment section and then adding that array of numbers
                to the sentence tensor returned by the forward() of EmbeddingsGenerator. The 
                alternating columns of this 2D array are filled using sine and cosine functions whose 
                periodicities vary with the column index as shown below.  Note that whereas the 
                periodicities are column-specific, the values of the sine and cosine functions are 
                word-position-specific.  In the depiction that follows, each row is an embedding
                vector for a specific word:
                                       
                                        
                                       along the embedding vector index  i -->
    
                            i=0  |  i=1  |  i=2 |  i=3 |        ...........               | i=511
                          -------------------------------------------------------------------------
            w   pos=0            |       |      |      |                                  |
            o             -------------------------------------------------------------------------
            r   pos=1            |       |      |      |                                  |
            d             -------------------------------------------------------------------------
                pos=2            |       |      |      |                                  |
            i             -------------------------------------------------------------------------                     
            n                                              .
            d                                              .
            e                                              .
            x             -------------------------------------------------------------------------
                pos=9            |       |      |      |                                  |
                          -------------------------------------------------------------------------
                                            |      |
                                            |      |
                                            |      |_________________
                                            |                        |
                                            |                        |
                                            V                        V
                                           pos                      pos                                                    ## (D)
                                  sin( ------------- )     cos( ------------- )
                                        100^{2i/512}             100^{2i/512}                                              ## (E)
    

                To further explain the idea of positional encoding, let's say we have a sentence 
                consisting of 10 words and an embedding size of 512.  For such a sentence, 
                the forward() of EmbeddingsGenerator will return a tensor of shape [10,512].  So the 
                array of positional-encoding numbers we need to construct will also be of shape 
                [10,512].  We need to fill the alternating columns of this [10,512] array with sin() 
                and cos() values as shown above.  To appreciate the significance of these values, first 
                note that one period of a sinusoidal function like sin(pos) is 2*pi with respect to the 
                word index pos. [That would amount to only about six words. That is, there would only 
                be roughly six words in one period if we just use sin(pos) above.]  On the other hand, 
                one period of a sinusoidal function like sin(pos/k) is 2*pi*k with respect to the word 
                index pos.  So if k=100, we have a periodicity of about 640 word positions along the 
                pos axis.  The important point is that every individual column in the 2D pattern shown
                above gets a unique periodicity and that the alternating columns are characterized by 
                sine and cosine functions.
                """
                position_encodings = torch.zeros_like( sentence_tensor,  dtype=float )
                ## Calling unsqueeze() with arg 1 causes the "row tensor" to turn into a "column tensor"
                ##    which is needed in the products in lines (F) and (G). We create a 2D pattern by 
                ##    taking advantage of how PyTorch has overloaded the definition of the infix '*' 
                ##    tensor-tensor multiplication operator.  It in effect creates an outer-product of
                ##    of what is essentially a column vector with what is essentially a row vector.
                word_positions = torch.arange(0, self.max_seq_length).unsqueeze(1)            
                div_term =  1.0 / (100.0 ** ( 2.0 * torch.arange(0, self.embedding_size, 2) / float(self.embedding_size) ))
                position_encodings[:, :, 0::2] =  torch.sin(word_positions * div_term)                                     ## (F)
                position_encodings[:, :, 1::2] =  torch.cos(word_positions * div_term)                                     ## (G)
                return sentence_tensor + position_encodings


        ###%%%
        #####################################   Encoder Code TransformerPreLN  ###############################################

        class MasterEncoder(nn.Module):
            """
            The purpose of the MasterEncoder is to invoke a stack of BasicEncoder instances on a
            source-language sentence tensor. The output of each BasicEncoder is fed as input to the 
            next BasicEncoder in the cascade, as illustrated in the loop in Line B below.  The stack
            of BasicEncoder instances is constructed in Line A.

            ClassPath:  Transformers   ->   TransformerPreLN   ->   MasterEncoder
            """
            def __init__(self, dls, xformer, how_many_basic_encoders, num_atten_heads):
                super(Transformers.TransformerPreLN.MasterEncoder, self).__init__()
                self.max_seq_length = xformer.max_seq_length
                self.basic_encoder_arr = nn.ModuleList( [xformer.BasicEncoder(dls, xformer,
                                                          num_atten_heads) for _ in range(how_many_basic_encoders)] )      ## (A)
            def forward(self, sentence_tensor):
                out_tensor = sentence_tensor
                for i in range(len(self.basic_encoder_arr)):                                                               ## (B)
                    out_tensor = self.basic_encoder_arr[i](out_tensor)
                return out_tensor


        class BasicEncoder(nn.Module):
            """
            The BasicEncoder in TransformerPreLN consists of a layer of self-attention (SA) followed
            by a purely feed-forward layer (FFN).  The job of the SA layer is for the network to
            figure out what parts of an input sentence are relevant to what other parts of the
            same sentence in the process of learning how to translate a source-language sentence into
            a target-language sentence. The output of SA goes through FFN and the output of FFN becomes
            the output of the BasicEncoder.  To mitigate the problem of vanishing gradients in the PreLN
            transformer design, the input to each of the two components of a BasicEncoder --- SA and 
            FFN --- is subject to LayerNorm and a residual connection used that wraps around both the
            LayerNorm and the component as shown in the keystroke diagram in the comment block associated
            with the definition of TranformerPreLN. Deploying a stack of BasicEncoder instances becomes
            easier if the output tensor from a BasicEncoder has the same shape as its input tensor.  

            The SelfAttention layer mentioned above consists of a number of AttentionHead instances, 
            with each AttentionHead making an independent assessment of what to say about the 
            inter-relationships between the different parts of an input sequence. It is the embedding
            axis that is segmented out into disjoint slices for each AttentionHead instance.The 
            calling SelfAttention layer concatenates the outputs from all its AttentionHead instances 
            and presents the concatenated tensor as its own output.

            ClassPath:  Transformers  ->  TransformerPreLN  ->   BasicEncoder
            """
            def __init__(self, dls, xformer, num_atten_heads):
                super(Transformers.TransformerPreLN.BasicEncoder, self).__init__()
                self.dls = dls
                self.embedding_size = xformer.embedding_size                                             
                self.qkv_size = self.embedding_size // num_atten_heads
                self.max_seq_length = xformer.max_seq_length                                                     
                self.num_atten_heads = num_atten_heads
                self.self_attention_layer = xformer.SelfAttention(dls, xformer, num_atten_heads)                           ## (A)
                self.norm1 = nn.LayerNorm(self.embedding_size)                                                             ## (C)
                self.W1 =  nn.Linear( self.max_seq_length * self.embedding_size, self.max_seq_length * 2 * self.embedding_size )
                self.W2 =  nn.Linear( self.max_seq_length * 2 * self.embedding_size, self.max_seq_length * self.embedding_size ) 
                self.norm2 = nn.LayerNorm(self.embedding_size)                                                             ## (E)

            def forward(self, sentence_tensor):
                input_for_self_atten = sentence_tensor.float()
                normed_input_self_atten = self.norm1( input_for_self_atten )
                output_self_atten = self.self_attention_layer(normed_input_self_atten).to(self.dls.device)                            ## (F)
                input_for_FFN = output_self_atten + input_for_self_atten
                normed_input_FFN = self.norm2(input_for_FFN)                                      ## (I)               
                basic_encoder_out =  nn.ReLU()(self.W1( normed_input_FFN.view(sentence_tensor.shape[0],-1) ))                         ## (K)
                basic_encoder_out =  self.W2( basic_encoder_out )                                                          ## (L)
                basic_encoder_out = basic_encoder_out.view(sentence_tensor.shape[0], self.max_seq_length, self.embedding_size ) 
                basic_encoder_out =  basic_encoder_out  + input_for_FFN
                return basic_encoder_out



        ###%%%
        ####################################  Self Attention Code TransformerPreLN ###########################################

        class SelfAttention(nn.Module):
            """
            As described in the doc section of the BasicEncoder class, in each BasicEncoder you have 
            a layer of SelfAttention followed by a Fully-Connected layer.  The SelfAttention layer 
            concatenates the outputs from all AttentionHead instances and presents that concatenated 
            output as its own output. If the input sentence consists of W words at most and if the 
            embedding size is M, the sentence_tensor at the input to forward() in Line B below will 
            be of shape [W,M].  This tensor will be fed into each AttentionHead instance constructed
            in Line A.  If K is the size of the output from an AttentionHead instance, the output of each 
            such instance will be of shape [W,K].  The SelfAttention instance concatenates A of those
            AttentionHead outputs and returns a tensor of shape [W,K*A] to the BasicEncoder.  This
            concatenation is carried in the loop in Lines C and D below.

            ClassPath:  Transformers  ->  TransformerPreLN  ->   SelfAttention
            """
            def __init__(self, dls, xformer, num_atten_heads):
                super(Transformers.TransformerPreLN.SelfAttention, self).__init__()
                self.dl_studio = dls
                self.max_seq_length = xformer.max_seq_length                                                     
                self.embedding_size = xformer.embedding_size
                self.num_atten_heads = num_atten_heads
                self.qkv_size = self.embedding_size // num_atten_heads
                self.attention_heads_arr = nn.ModuleList( [xformer.AttentionHead(dls, self.max_seq_length, 
                                        self.qkv_size, num_atten_heads)  for _ in range(num_atten_heads)] )                ## (A)
            def forward(self, sentence_tensor):                                                                            ## (B)
                concat_out_from_atten_heads = torch.zeros( sentence_tensor.shape[0], self.max_seq_length, 
                                                                      self.num_atten_heads * self.qkv_size).float()
                for i in range(self.num_atten_heads):                                                                      ## (C) 
                    sentence_tensor_portion = sentence_tensor[:, :, i * self.qkv_size : (i+1) * self.qkv_size]
                    concat_out_from_atten_heads[:, :, i * self.qkv_size : (i+1) * self.qkv_size] =          \
                                                                 self.attention_heads_arr[i](sentence_tensor_portion)      ## (D)
                return concat_out_from_atten_heads


        class AttentionHead(nn.Module):
            """
            An AttentionHead (AH) instance does its job by first matrix-multiplying the embedding
            vector FOR EACH WORD in an input sentence by the following three matrices:

            -- a matrix Wq of size PxM to produce the query Vector q where P is the desired size 
               for the matrix-vector product and M the size of the embedding.  For each word, this 
               will yield a query vector of size P elements at each word position in the input 
               sentence.
      
            -- a matrix Wk, also of size PxM, to produce a key vector k of size P elements at
               each word position

            -- a matrix Wv, also of size PxM, to produce a value vector v of size P elements at
               each word position.

               NOTE:  In the implementation shown below, P is the same thing as qkv_size

            Since in reality what the AttentionHead sees is a packing of the embedding vectors 
            for all the W words in an input sentence in the form of a tensor X of shape [W,M], 
            instead of learning the matrices Wq, Wk, Wv for the individual words, it learns the 
            tensors WQ, WK, and WV for all the words in the input sentence simultaneously. Each 
            of these tensors will be of shape [W,M,P] where again W is the number of words, M 
            the size of the embedding, and P the size of the query, key, and value vectors 
            mentioned above.  We can mow write the following equations for the Query Q, Key K,
            and Value V tensors calculated by an AttentionHead instance:

                 Q  =   X . WQ            K  =  X . WK             V  =  X . WV                                            ## (A)

            The Q, K, and V tensors will be of shape [W,P].

            Going back to the individual word based query, key, and value vectors, the basic 
            idea in self-attention is to first take the dot-product of EACH query-vector q_i 
            with EVERY key vector k_j in the input sentence (i and j are word indexes).  So if 
            you have W words in an input sentence, at EACH word position in the input sentence, 
            you will have W of these dot-product  "q_i k'_j"  scalar values where  k'_j  stands 
            for the transpose of the key vector at the j^th position in the sentence.

            From all the W dot-products at each word position in a W-word input sentence, 
            calculated as explained above, you want to subject each W-element dot product 
            to a nn.Softmax normalization to yield a W-element list of probabilities, as 
            shown in Line N below. Applying torch.max to such a probability vector, as
            shown in Line O, yields the largest value of the probability that you retain 
            at each word position in the input sentence.  Since the torch.max() function is 
            applied to the WxW matrix of dot products, that is, to the dot products at every
            word position in the input sentence, what is returned in Line O is the max 
            probability at every word position in the input sentence.  In other words, the
            output in Line O is a W-element vector of max probabilities.

            To extend the above explanation to operations with the sentence based Q, K, 
            and V tensors, what's interesting  is that a matrix-matrix dot-product of the 
            Q and K tensors directly carries out all the dot-products at each word position
            in the input sentence.  Since Q and K are each of shape [W,P] for a W-word 
            sentence, the inner-product  "Q . K^T"  is of shape WxW, whose first W-element
            row contains the values obtained by taking the dot-product of the first-word 
            query vector q_1 with each of the k_1, k_2, k_3, ..., k_W key vectors for each
            of the W words in the sentence.  The second row of  "Q . K^T" will likewise 
            represent the dot product of the second query vector with every key vector, and
            so on.

            Next what we want to do is to apply nn.Softmax operator to each row of the 
            "Q . K^T" product to retain just a single scalar value for that word position.  
            That is, at the i^th work position in the input, we want to retain:

            torch.max(nn.Softmax( q_i . k'_1    q_i . k'_2    q_i . k'_3     .....    q_i . k'_W)

            In other words, we want to apply the softmax operator separately to each row
            of the tensor-tensor product Q . K^T.

            The following formula expresses the above more compactly:

                           
                                 torch.max( nn.Softmax(Q .  K^T) )
                      Z   =      ---------------------------------  . V
                                            sqrt(M)

            Remember that, for a W-word input sentence, the numerator will yield W numbers,
            one for each word position in the input sentence.  Each of these serve as a 
            coefficient for the corresponding element of the value tensor V.  Recall that 
            the shape of V for a W-word input sentence is [W,P].

            A final note about the implementation of the tensor-tensor inner products 
            shown in Line A above:

              Q  =   X . WQ            K  =  X . WK             V  =  X . WV     

            where WQ, WK, and WV are tensors of learnable weights.  Consider a 10-word input 
            sentence, and the embedding size of M, X will of shape [10,M].  On the other 
            hand, the tensors Q, K, and V are each of shape [10,P].  That is, for EACH WORD, 
            the input vector is of size M and the output of size P.  If we "flatten" X for 
            the purpose of using nn.Linear for learning the weight matrices WQ, WK, and WV, 
            X will go into a 1-D array of length 10*M and Q will go into a 1-D array of size 
            10*P. So the learnable weights will be learned by nn.Linear(10*M, 10*P).  In 
            general we are going to need a 

                nn.Linear( max_seq_length * embedding_size, max_seq_length * qkv_size)

                                                NOTE:  qdv_size is the same thing as P
           
            for learning the weights in the matrices WQ, WK, and WV where max_seq_length is the 
            max number of words in an input sentence

            We now retain the max value in each of the WxW product of the Q and K^T.  The 
            sequence of these max values will be W elements long.

            We obtain  Z by multiplying each row of WxP matrix V by its corresponding element
            of the max_val vector. Therefore, Z itself is a WxP matrix. 

            Note again that P is represented by "qkv_size" in the code. 

            ClassPath:  Transformers   ->   TransformerPreLN   ->   AttentionHead
            """
            def __init__(self, dl_studio, max_seq_length, qkv_size, num_atten_heads):
                super(Transformers.TransformerPreLN.AttentionHead, self).__init__()
                self.dl_studio = dl_studio
                self.qkv_size = qkv_size
                self.max_seq_length = max_seq_length
                self.WQ =  nn.Linear( max_seq_length * self.qkv_size, max_seq_length * self.qkv_size )                     ## (B)
                self.WK =  nn.Linear( max_seq_length * self.qkv_size, max_seq_length * self.qkv_size )                     ## (C)
                self.WV =  nn.Linear( max_seq_length * self.qkv_size, max_seq_length * self.qkv_size )                     ## (D)
                self.softmax = nn.Softmax(dim=1)                                                                           ## (E)

            def forward(self, sentence_portion):                                                                           ## (F)
                Q = self.WQ( sentence_portion.reshape(sentence_portion.shape[0],-1).float() ).to(self.dl_studio.device)    ## (G)
                K = self.WK( sentence_portion.reshape(sentence_portion.shape[0],-1).float() ).to(self.dl_studio.device)    ## (H)
                V = self.WV( sentence_portion.reshape(sentence_portion.shape[0],-1).float() ).to(self.dl_studio.device)    ## (I)
                Q = Q.view(sentence_portion.shape[0], self.max_seq_length, self.qkv_size)                                  ## (J)
                K = K.view(sentence_portion.shape[0], self.max_seq_length, self.qkv_size)                                  ## (K)
                V = V.view(sentence_portion.shape[0], self.max_seq_length, self.qkv_size)                                  ## (L)
                A = K.transpose(2,1)                                                                                       ## (M)
                QK_dot_prod = Q @ A                                                                                        ## (N)
                rowwise_softmax_normalizations = self.softmax( QK_dot_prod )                                               ## (O)
                Z = rowwise_softmax_normalizations @ V
                coeff = 1.0/torch.sqrt(torch.tensor([self.qkv_size]).float()).to(self.dl_studio.device)                    ## (S)
                Z = coeff * Z                                                                                              ## (T)
                return Z


        ###%%%
        ####################################  Cross Attention Code TransformerPreLN  #########################################

        class CrossAttention(nn.Module):
            """
            To understand the implementation of cross attention, a good starting point would be to 
            go through the documentation section provided for the SelfAttention and AttentionHead
            classes.  Whereas self-attention consists of taking dot products the query vectors for
            the individual words in a sentence with the key vectors for all the words in order to
            discover the inter-word relevancies in a sentence, in cross-attention we take the dot 
            products of the query vectors for the individual words in the target-language sentence
            with the key vectors at the output of the master encoder for a given source-language
            sentence.

            More specifically, let X_enc represent the tensor at the output of the MasterEncoder.
            Its shape will be that of the input-sentence to the MasterEncoder instance.  If W is
            the largest number of words allowed in a sentence (in either language), the X tensor 
            that is input into the MasterEncoder will be of shape [W,M] where M is the size of the 
            embedding vectors used for the words. Therefore, in our implementation, the shape of 
            the output of the MasterEncoder, X_enc, is also [W,M]. Now let X_target represent the
            tensor form of the target-language sentence. Its shape in our implementation is also
            [W,M].  The cross attention layer first calculates the following tensors in Line

                Q  =   X_target . WQ          K  =  X_source . WK           V  =  X_source . WV                            ## (A)

            and then computes the cross-attention weights in very much the same manner as the
            self-attention weights shown earlier.

                           
                                 torch.max( nn.Softmax(Q .  K^T) )
                      Z   =      ---------------------------------  . V
                                            sqrt(M)

            This computation is done in Lines K through R shown below.

            ClassPath:  Transformers  ->  TransformerPreLN  ->   CrossAttention
            """
            def __init__(self, dls, xformer, num_atten_heads):
                super(Transformers.TransformerPreLN.CrossAttention, self).__init__()
                self.dl_studio = dls
                self.max_seq_length = xformer.max_seq_length
                self.embedding_size = xformer.embedding_size
                self.num_atten_heads = num_atten_heads
                self.qkv_size = self.embedding_size // num_atten_heads
                self.attention_heads_arr = nn.ModuleList( [xformer.CrossAttentionHead(dls, self.max_seq_length, 
                                        self.qkv_size, num_atten_heads)  for _ in range(num_atten_heads)] )              
            def forward(self, basic_decoder_out, final_encoder_out):                                                     
                concat_out_from_atten_heads = torch.zeros( basic_decoder_out.shape[0], self.max_seq_length, 
                                                                      self.num_atten_heads * self.qkv_size).float()
                for i in range(self.num_atten_heads):                                                                    
                    basic_decoder_portion = basic_decoder_out[:, :, i * self.qkv_size : (i+1) * self.qkv_size]
                    final_encoder_portion = final_encoder_out[:, :, i * self.qkv_size : (i+1) * self.qkv_size]
                    concat_out_from_atten_heads[:, :, i * self.qkv_size : (i+1) * self.qkv_size] =          \
                                            self.attention_heads_arr[i](basic_decoder_portion, final_encoder_portion)    
                return concat_out_from_atten_heads


        class CrossAttentionHead(nn.Module):
            def __init__(self, dl_studio, max_seq_length, qkv_size, num_atten_heads):
                super(Transformers.TransformerPreLN.CrossAttentionHead, self).__init__()
                self.dl_studio = dl_studio
                self.qkv_size = qkv_size
                self.max_seq_length = max_seq_length
                self.WQ =  nn.Linear( max_seq_length * self.qkv_size, max_seq_length * self.qkv_size )                   
                self.WK =  nn.Linear( max_seq_length * self.qkv_size, max_seq_length * self.qkv_size )                   
                self.WV =  nn.Linear( max_seq_length * self.qkv_size, max_seq_length * self.qkv_size )                   
                self.softmax = nn.Softmax(dim=1)                                                                         

            def forward(self, basic_decoder_portion, final_encoder_portion):
                Q = self.WQ( basic_decoder_portion.reshape(final_encoder_portion.shape[0],-1).float() ).to(self.dl_studio.device)   
                K = self.WK( final_encoder_portion.reshape(final_encoder_portion.shape[0],-1).float() ).to(self.dl_studio.device)   
                V = self.WV( final_encoder_portion.reshape(final_encoder_portion.shape[0],-1).float() ).to(self.dl_studio.device)   
                Q = Q.view(final_encoder_portion.shape[0], self.max_seq_length, self.qkv_size)                           
                K = K.view(final_encoder_portion.shape[0], self.max_seq_length, self.qkv_size)                           
                V = V.view(final_encoder_portion.shape[0], self.max_seq_length, self.qkv_size)                           
                A = K.transpose(2,1)                                                                                     
                QK_dot_prod = Q @ A                                                                                      
                rowwise_softmax_normalizations = self.softmax( QK_dot_prod )                                             
                Z = rowwise_softmax_normalizations @ V
                coeff = 1.0/torch.sqrt(torch.tensor([self.qkv_size]).float()).to(self.dl_studio.device)                  
                Z = coeff * Z                                                                                            
                return Z


        ###%%%
        #######################################   Decoder Code TransformerPreLN ##############################################

        class BasicDecoderWithMasking(nn.Module):
            """
            As with the basic encoder, while a basic decoder also consists of a layer of SelfAttention 
            followed by a Feedforward Network (FFN) layer, but now there is a layer of CrossAttention 
            interposed between the two.  In the PreLN decoder, the input to each of these three 
            components is first subject to LayerNorm and, for each component, the residual connection 
            wraps around both the LayerNorm at the input and the component itself.

            An important feature of the BasicDecoder is masking --- which is also called autoregressive 
            masking or, sometimes more informally, as triangle masking ---in order to ensure that the 
            predicted word at each position in the output depends only on the words that have already 
            been predicted at all previous positions in output sentence.

            ClassPath:  Transformers  ->  TransformerPreLN  ->   BasicDecoderWithMasking
            """
            def __init__(self, dls, xformer, num_atten_heads):
                super(Transformers.TransformerPreLN.BasicDecoderWithMasking, self).__init__()
                self.dls = dls
                self.embedding_size = xformer.embedding_size                                             
                self.max_seq_length = xformer.max_seq_length                                                     
                self.num_atten_heads = num_atten_heads
                self.qkv_size = self.embedding_size // num_atten_heads
                self.self_attention_layer = xformer.SelfAttention(dls, xformer, num_atten_heads)
                self.norm1 = nn.LayerNorm(self.embedding_size)
                self.cross_attn_layer = xformer.CrossAttention(dls, xformer, num_atten_heads)
                self.norm2 = nn.LayerNorm(self.embedding_size)
                self.W1 =  nn.Linear( self.max_seq_length * self.embedding_size, self.max_seq_length * 2 * self.embedding_size )
                self.W2 =  nn.Linear( self.max_seq_length * 2 * self.embedding_size, self.max_seq_length * self.embedding_size ) 
                self.norm3 = nn.LayerNorm(self.embedding_size)

            def forward(self, sentence_tensor, final_encoder_out, mask):   
                masked_sentence_tensor = sentence_tensor
                if mask is not None:
                    masked_sentence_tensor = self.apply_mask(sentence_tensor, mask, self.max_seq_length, self.embedding_size)
                normed_masked_sentence_tensor = self.norm1( masked_sentence_tensor.float() )
                output_self_atten = self.self_attention_layer(normed_masked_sentence_tensor).to(self.dls.device)
                input_for_xatten = output_self_atten + masked_sentence_tensor                     
                normed_input_xatten = self.norm2(input_for_xatten)
                output_of_xatten  = self.cross_attn_layer( normed_input_xatten, final_encoder_out).to(self.dls.device)
                input_for_FFN  = output_of_xatten + input_for_xatten
                normed_input_FFN = self.norm3(input_for_FFN)
                basic_decoder_out =  nn.ReLU()(self.W1( normed_input_FFN.view(sentence_tensor.shape[0],-1) ))                  
                basic_decoder_out =  self.W2( basic_decoder_out )                                                    
                basic_decoder_out = basic_decoder_out.view(sentence_tensor.shape[0], self.max_seq_length, self.embedding_size )
                basic_decoder_out =  basic_decoder_out  + input_for_FFN
                return basic_decoder_out

            def apply_mask(self, sentence_tensor, mask, max_seq_length, embedding_size):  
                out = torch.zeros(sentence_tensor.shape[0], max_seq_length, embedding_size).float().to(self.dls.device) 
                out[:,:,:len(mask)] = sentence_tensor[:,:,:len(mask)] 
                return out    


        class MasterDecoderWithMasking(nn.Module):
            """
            The primary job of the MasterDecoder is to orchestrate the invocation of a stack of BasicDecoders.
            The number of BasicDecoder instances used is a user-defined parameter.  The masking that is used
            in each BasicDecoder instance is set here by the MasterDecoder.  In Line (B), we define the 
            BasicDecoder instances needed. The linear layer in Line (C) is needed because what the decoder side
            produces must ultimately be mapped as a probability distribution over the entire vocabulary for the
            target language.  With regard to the data flow through the network, note how the mask is initialized
            in Line (D).  The mask is supposed to be a vector of one's that grows with the prediction for each
            output word. We start by setting it equal to just a single-element vector containing a single "1".
            Lines (E) and (F) declare the tensors that will store the final output of the master decoder. This
            final output consists of two tensors: One tensor holds the integer index to the target-language 
            vocabulary word where the output log-prob is maximum. [This index is needed at inference time to 
            output the words in the translation.]  The other tensor holds the log-probs over the target language
            vocabulary. The log-probs are produced by the nn.LogSoftmax in Line (L).

            ClassPath:  Transformers  ->  TransformerPreLN  ->   MasterDecoderWithMasking
            """
            def __init__(self, dls, xformer, how_many_basic_decoders, num_atten_heads):
                super(Transformers.TransformerPreLN.MasterDecoderWithMasking, self).__init__()
                self.dls = dls
                self.max_seq_length = xformer.max_seq_length
                self.embedding_size = xformer.embedding_size
                self.target_vocab_size = xformer.vocab_es_size                                                                   ## (A)
                self.basic_decoder_arr = nn.ModuleList([xformer.BasicDecoderWithMasking( dls, xformer,
                                                                   num_atten_heads) for _ in range(how_many_basic_decoders)])    ## (B)
                ## Need the following layer because we want the prediction of each target word to be a probability distribution
                ## over the target vocabulary:
                self.out = nn.Linear(self.embedding_size, self.target_vocab_size)                                                ## (C)

            def forward(self, sentence_tensor, final_encoder_out):
                mask = torch.ones(1, dtype=int)                                                                                  ## (D)
                ##  A tensor with two axes, one for the batch instance and the other for storing the predicted 
                ##  word ints for that batch instance. We initialize by filling the tensor with "EOS" tokens (==1).
                predicted_word_index_values = torch.ones(sentence_tensor.shape[0], self.max_seq_length, 
                                                                                 dtype=torch.long).to(self.dls.device)           ## (E)
                predicted_word_index_values[:,0] = 0      ## change the first word to the "SOS" token
                ##  A tensor with two axes, one for the batch instance and the other for storing the log-prob of predictions 
                ##  for that batch instance. The log_probs for each predicted word over the entire target vocabulary:
                predicted_word_logprobs  = torch.zeros( sentence_tensor.shape[0], self.max_seq_length, 
                                                              self.target_vocab_size, dtype=float).to(self.dls.device)           ## (F)
                for mask_index in range(1, sentence_tensor.shape[1]):
                    masked_target_sentence = self.apply_mask(sentence_tensor, mask, self.max_seq_length, self.embedding_size)    ## (G)
                    out_tensor = masked_target_sentence   ## it will start as just the first word, then two first words, etc.    ## (H)
                    for i in range(len(self.basic_decoder_arr)):                                                                 ## (I)
                        out_tensor = self.basic_decoder_arr[i](out_tensor, final_encoder_out, mask)                              ## (J)
                    last_word_tensor = out_tensor[:,mask_index]                                                            
                    last_word_onehot = self.out(last_word_tensor.view(sentence_tensor.shape[0],-1))                              ## (K)
                    output_word_logprobs = nn.LogSoftmax(dim=1)(last_word_onehot)                                                ## (L)
                    _, idx_max = torch.max(output_word_logprobs, 1)                                                              ## (M)
                    predicted_word_index_values[:,mask_index] = idx_max                                                          ## (N)
                    predicted_word_logprobs[:,mask_index] = output_word_logprobs                                                 ## (O)
                    mask = torch.cat( ( mask, torch.ones(1, dtype=int) ) )                                                       ## (P)
                return predicted_word_logprobs, predicted_word_index_values                                                      ## (Q)


            def apply_mask(self, sentence_tensor, mask, max_seq_length, embedding_size):  
                out = torch.zeros_like(sentence_tensor).float().to(self.dls.device)
                out[:,:len(mask),:] = sentence_tensor[:,:len(mask),:] 
                return out    

        ###%%%
        ##############################  Training and Evaluation for TransformerPreLN  ########################################

        def save_encoder(self, encoder):
            "Save the trained encoder to a disk file"       
            torch.save(encoder.state_dict(), self.dl_studio.path_saved_model["encoder"])

        def save_decoder(self, decoder):
            "Save the trained decoder to a disk file"       
            torch.save(decoder.state_dict(), self.dl_studio.path_saved_model["decoder"])
        

        def run_code_for_training_TransformerPreLN(self, dls, master_encoder, master_decoder, display_train_loss=False):        
            """
            Since we did not construct a dataloader by subclassing from torch.utils.data.DataLoader, we
            need the statements in Lines (A) through (G) to deal with the cases when the size of the
            training data is not an exact multiple of the batch size, etc.  The code shown in these
            lines ensures that every bit of the available training data is used even if that means 
            that the last batch will not have the expected number of training samples in it.

            Overall, the training consists of running the English/Spanish sentence pairs as generated 
            in Lines (F) or (G) through the MasterEncoder-MasterDecoder combo as shown in Lines (P) and
            (Q).  We must first convert the words in these sentences into their int values in Lines (J) 
            and (K) and subsequently generate the embeddings for each word in Lines (L) and (M).

            For a given English sentence, at each word position in a max_seq_length of positions in 
            Spanish, the encoder-decoder combo generates a tensor of log probabilities (logprobs) over 
            the target vocabulary. That is what is returned in Line (Q). [The logprobs are generated by the
            final nn.LogSoftmax activation function in the decoder code.]  At each word position in the
            target language sequence, the nn.NLLLoss() criterion then picks the negative of that logprob 
            that corresponds to the index of the groundtruth target word as a measure of loss for that word 
            position.

            Regarding the loss function nn.NLLLoss used for training, note that using a combination
            of nn.LogSoftmax activation and nn.NLLLoss is the same thing as using nn.CrossEntropyLoss,
            which is the most commonly used loss function for solving classification problems. For a 
            neural network that is meant for solving a classification problem, the number of nodes in
            the output layer must equal the number of classes.  Applying nn.LogSoftmax activation to
            such a layer normalizes the values accumulated at those nodes so that they become a legal
            probability distribution over the classes.  Subsequently, calculating the nn.NLLLoss 
            means choosing the negative value at just that node which corresponds to the actual class 
            label of the input data.                                                                    
            """
            master_encoder.to(self.dl_studio.device)
            master_decoder.to(self.dl_studio.device)     
            embeddings_generator_en = self.EmbeddingsGenerator(self, 'en', self.embedding_size).to(self.dl_studio.device)
            embeddings_generator_es = self.EmbeddingsGenerator(self, 'es', self.embedding_size).to(self.dl_studio.device)
            master_encoder_optimizer = optim.Adam(master_encoder.parameters(), lr=self.dl_studio.learning_rate)
            master_decoder_optimizer = optim.Adam(master_decoder.parameters(), lr=self.dl_studio.learning_rate)
            ##  Note that by default, nn.NLLLoss averages over the instances in a batch
            criterion = nn.NLLLoss()                                                                                           
            accum_times = []
            start_time = time.perf_counter()
            training_loss_tally = []
            self.debug = False
            print("")
            num_sentence_pairs = len(self.training_corpus)
            print("\n\nNumber of sentence pairs in the dataset: ", num_sentence_pairs)
            print("\nNo sentence is longer than %d words (including the SOS and EOS tokens)\n\n" % self.max_seq_length)
            batch_size = self.dl_studio.batch_size
            max_iters = len(self.training_corpus) // batch_size                                                                  ## (A)
            max_iters = max_iters if len(self.training_corpus) % batch_size == 0 else max_iters + 1                              ## (B)
            print("\n\nMaximum number of training iterations in each epoch: %d\n\n" % max_iters)
            for epoch in range(self.dl_studio.epochs):                                                                           ## (C)
                print("")
                random.shuffle(self.training_corpus)                                                                             ## (D)
                running_loss = 0.0
                for iter in range(max_iters):
                    if self.debug:
                        print("\n\n\n========================== starting batch indexed: %d ================================\n\n\n" % iter)
                    if (iter+1)*batch_size <= num_sentence_pairs:                                                                ## (E)
                        batched_pairs = self.training_corpus[iter*batch_size : (iter+1)*batch_size]                              ## (F)
                    else:
                        batched_pairs = self.training_corpus[iter*batch_size : ]                                                 ## (G)
                    if self.debug:                 ##  MUST use a batch size of 5 in your script for this to work
                        batched_pairs =  [['SOS i will kill him EOS', 'SOS le mataré EOS'], ['SOS what was that EOS', 'SOS qué era eso EOS'], ['SOS he soon left the new job EOS', 'SOS él dejó pronto el nuevo empleo EOS'], ['SOS i go into the city every day EOS', 'SOS yo voy a la ciudad todos los días EOS'], ['SOS she might come tomorrow EOS', 'SOS puede que ella venga mañana EOS']]
                    source_sentences = [pair[0] for pair in batched_pairs]                                                       ## (H)
                    target_sentences = [pair[1] for pair in batched_pairs]                                                       ## (I)
                    if self.debug:
                        print("\n\nfirst source sentence in batch: ", source_sentences[0])
                        print("\nfirst target sentence in batch: ", target_sentences[0])
                        print("\nlast source sentence in batch: ", source_sentences[-1])
                        print("\nlast target sentence in batch: ", target_sentences[-1])
                    en_sent_ints =  self.sentence_with_words_to_ints(source_sentences, 'en').to(self.dl_studio.device)           ## (J)
                    es_sent_ints =  self.sentence_with_words_to_ints(target_sentences, 'es').to(self.dl_studio.device)           ## (K)
                    en_sentence_tensor = embeddings_generator_en(en_sent_ints)                                                   ## (L)
                    es_sentence_tensor = embeddings_generator_es(es_sent_ints)                                                   ## (M)
                    master_encoder_optimizer.zero_grad()                                                                         ## (N)
                    master_decoder_optimizer.zero_grad()                                                                         ## (O)
                    master_encoder_output = master_encoder( en_sentence_tensor )                                                 ## (P)
                    predicted_word_logprobs, predicted_word_index_values = master_decoder(es_sentence_tensor, 
                                                                                                master_encoder_output)           ## (Q)
                    loss = torch.tensor(0.0).to(self.dl_studio.device)
                    for di in range(es_sent_ints.shape[1]):                                                                      ## (R)
                        loss += criterion(predicted_word_logprobs[:,di], es_sent_ints[:,di])                                     ## (S)
                    loss.backward()                                                                                              ## (T)
                    master_encoder_optimizer.step()                                                                              ## (U)
                    master_decoder_optimizer.step()                                                                              ## (V)
                    loss_normed = loss.item() / es_sent_ints.shape[0]                                                            ## (W)
                    running_loss += loss_normed                                                                                  ## (X)
                    if iter % 200 == 199:         
                        avg_loss = running_loss / float(200)
                        training_loss_tally.append(avg_loss)
                        running_loss = 0.0
                        current_time = time.perf_counter()
                        time_elapsed = current_time-start_time
                        print("[epoch:%2d  iter:%4d  elapsed_time: %4d secs]     loss: %.4f" % (epoch+1, iter+1, time_elapsed,avg_loss))
                        accum_times.append(current_time-start_time)
                if self.save_model_end_of_epoch:
                    self.save_encoder(master_encoder)
                    self.save_decoder(master_decoder)
                    print("Model saved at the end of epoch %d" % (epoch+1))
            print("\nFinished Training\n")
            self.save_encoder(master_encoder)       
            self.save_decoder(master_decoder)       
            if display_train_loss:
                plt.figure(figsize=(10,5))
                plt.title("Training Loss vs. Iterations")
                plt.plot(training_loss_tally)
                plt.xlabel("iterations")
                plt.ylabel("training loss")
                plt.legend()
                plt.savefig("training_loss.png")
                plt.show()

        def run_code_for_evaluating_TransformerPreLN(self, master_encoder, master_decoder):
            """
            The main difference between the training code shown in the previous function and the
            evaluation code shown here is with regard to the input to MasterDecoder and how we process 
            its output. As shown in the previous function, for the training loop, the input to 
            MasterDecoder consists of the both the target sentence and the output of the MasterEncoder 
            for the source sentence.  However, at inference time (that is, in the evaluation loop shown 
            below), the target sentence at the input to the MasterDecoder is replaced by an encoding of 
            a "starter stub" output sentence as defined in line (B).  The main message conveyed by the 
            stub in line (B) is that we want to start the translation with the first work of the output 
            as being the token "SOS".  The encoding for the stub is generated in lines (F) and (G).

            The second significant difference between the training and the testing code is with regard 
            to how we process the output of the MasterDecoder.  As you will recall from the docstring 
            associated with MasterDecoder, it returns two things: (1) the predicted log probabilities 
            (logprob) over the target vocabulary for every word position in the target language; and 
            (2) for each target-language word position, the word_vocab_index at which the logprob is 
            maximum.  The loss calculation in the training code was based on the former.  ON the other 
            hand, as shown in line (H) below, it is the latter that lets us do the the translations in 
            the target words in line (I).
            """
            master_encoder.load_state_dict(torch.load(self.dl_studio.path_saved_model['encoder']))
            master_decoder.load_state_dict(torch.load(self.dl_studio.path_saved_model['decoder']))
            master_encoder.to(self.dl_studio.device)
            master_decoder.to(self.dl_studio.device)     
            embeddings_generator_en = self.EmbeddingsGenerator(self, 'en', self.embedding_size).to(self.dl_studio.device)
            embeddings_generator_es = self.EmbeddingsGenerator(self, 'es', self.embedding_size).to(self.dl_studio.device)
            self.debug = False
            with torch.no_grad():
                for iter in range(20):
                    starter_stub_for_translation = ['SOS', 'EOS', 'EOS', 'EOS', 'EOS', 'EOS', 'EOS', 'EOS', 'EOS', 'EOS']      ## (A)
                    batched_pairs = random.sample(self.training_corpus, 1)                                                     ## (B)
                    source_sentences = [pair[0] for pair in batched_pairs]
                    target_sentences = [pair[1] for pair in batched_pairs]
                    if self.debug:
                        print("\n\nsource sentences: ", source_sentences)
                        print("\ntarget sentences: ", target_sentences)
                    en_sent_ints =  self.sentence_with_words_to_ints(source_sentences, 'en').to(self.dl_studio.device)         ## (C)
                    if self.debug:
                        print("\n\nsource sentence tensor: ", en_sent_ints)
                        print("\n\ntarget sentence tensor: ", es_sent_ints)
                    en_sentence_tensor = embeddings_generator_en(en_sent_ints).to(self.dl_studio.device)                       ## (D)
                    master_encoder_output = master_encoder( en_sentence_tensor )                                               ## (E)
                    starter_stub_as_ints = self.sentence_with_words_to_ints(
                                                  [" ".join(starter_stub_for_translation)], 'es').to(self.dl_studio.device)    ## (F)
                    starter_stub_tensor = embeddings_generator_es(starter_stub_as_ints)                                        ## (G)
                    _, predicted_word_index_values = master_decoder(starter_stub_tensor, master_encoder_output)                ## (H)
                    predicted_word_index_values = predicted_word_index_values[0].unsqueeze(1)
                    decoded_words = [self.es_index_2_word[predicted_word_index_values[di].item()] 
                                                                            for di in range(self.max_seq_length)]              ## (I)
                    output_sentence = " ".join(decoded_words)                                                                 
                    print("\n\n\nThe input sentence pair: ", source_sentences, target_sentences)                             
                    print("\nThe translation produced by TransformerPreLN: ", output_sentence)



    ################################  END Definition of Inner Class TransformerPreLN  ###################################
    #####################################################################################################################
    #####################################################################################################################


#_______________________  End of Transformers Class Definition __________________________

#______________________________    Test code follows    _________________________________

if __name__ == '__main__': 
    pass

