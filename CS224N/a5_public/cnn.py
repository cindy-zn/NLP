#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch.nn.functional as F
### YOUR CODE HERE for part 1i

class CNN(nn.Module):
    def __init__(self, char_embed_size, number_of_filters, max_word_length, kernel_size=5):
        """
        Initialize CNN layer
        :param embedding_size: character embedding size
        :param number_of_filters: num of filters used in CNN, decides the number of output channel
        :param max_word_length: will be used by max pooling
        :param kernel_size:
        """
        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size
        self.number_of_filters = number_of_filters
        self.max_word_legnth = max_word_length
        self.conv = nn.Conv1d(char_embed_size, number_of_filters, kernel_size, bias=True)
        # the size of the final word embedding after CNN is (max_word_length - k +1)
        # use max-pooling to reduce it to a single vector
        # which means kernel size of maxpooling is (max_word_length-k+1)
        self.pooling = nn.MaxPool1d(max_word_length-kernel_size+1)

    def forward(self, conv_input):
        """
        :param conv_input: (batch_size, char_embedding_size, max_word_length)
        :return: (batch_size, embed_size(word_embedding))
        """
        conv_output = F.relu(self.conv(conv_input))
        conv_output = self.pooling(conv_output).squeeze()
        return conv_output

### END YOUR CODE

