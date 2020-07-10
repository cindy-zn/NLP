#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
from cnn import CNN
from highway import Highway

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

# from cnn import CNN
# from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        self.vocab = vocab
        self.char_embed_size = 50
        self.max_word_length = 21
        self.dropout_rate = 0.3

        self.char_embedding = nn.Embedding(
            num_embeddings=len(vocab.char2id),
            embedding_dim=self.char_embed_size,
            padding_idx=vocab.char2id['<pad>'])
        self.CNN = CNN(
            char_embed_size=self.char_embed_size,
            number_of_filters=self.embed_size,
            max_word_length=self.max_word_length
        )
        self.Highway = Highway(
            input_size= embed_size
        )
        self.dropout = nn.Dropout(p=self.dropout_rate)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        char_embedding = self.char_embedding(input) # (sentence_length, batch_size, max_word_length, char_embed_size)
        sent_len, batch_size, _, _ = char_embedding.shape
        view_shape = (sent_len*batch_size, self.max_word_length, self.char_embed_size)
        char_embedding = char_embedding.view(view_shape).transpose(1, 2) #(sent_len*batch_size, char_embed_size, max_word_length)

        x_conv = self.CNN(char_embedding) #(sent_len*batch_size, embed_size)
        x_highway = self.Highway(x_conv) #(sent_len*batch_size, embed_size)
        x_dropout = self.dropout(x_highway)
        x_dropout = x_dropout.view(sent_len, batch_size, self.embed_size)
        return x_dropout
        ### END YOUR CODE

