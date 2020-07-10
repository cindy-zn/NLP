#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1h
class Highway(nn.Module):

    def __init__(self, input_size):
        """
        Highway layer
        :param input_size: embedding size
        """
        super(Highway, self).__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.gate = nn.Linear(input_size, input_size)
        self.gate.bias.data.fill_(-2.0)

    def forward(self, cnn_output: torch.Tensor) -> torch.Tensor:
        """
        take a batch of convolution output, get highway network output
        :param cnn_output: (batch_size, embed_size)
        :return: (batch_size, embed_size)
        """
        proj_result = F.relu(self.proj(cnn_output))
        proj_gate = torch.sigmoid(self.gate(cnn_output))
        gated_result = proj_gate * proj_result + (1-proj_gate) * cnn_output
        return gated_result
### END YOUR CODE 

# if __name__ == '__main__':
#     input_size = 5
#     batch_size = 10
#     x = torch.rand(batch_size, input_size)  # new_* methods take in sizes
#     # print(x)
#     hw = Highway(input_size)
#     output = hw.forward(x)
#     assert(output.shape[0]==batch_size)
#     assert(output.shape[1]==input_size)
#     print('correct shape')