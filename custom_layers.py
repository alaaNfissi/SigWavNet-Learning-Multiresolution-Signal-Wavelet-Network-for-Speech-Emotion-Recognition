#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Title: Custom Layers for Speech Emotion Recognition Model (SigWavNet)
Author: Alaa Nfissi
Date: March 31, 2024
Description: This file defines custom neural network layers and attention mechanisms 
specifically designed for enhancing the speech emotion recognition model's performance.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

d = 0
device = torch.device(f"cuda:{d}")

class Kernel(nn.Module):
    
    """
    Represents a learnable kernel, which can be initialized either randomly or with a specified array.
    """
    
    def __init__(self, kernelInit=20, trainKern=True):
        
        """
        Initializes the Kernel object.

        Parameters:
        - kernelInit (int, list): Initial size of the kernel or the initial array values.
        - trainKern (bool): Specifies whether the kernel is learnable.
        """
        
        super(Kernel, self).__init__()
        self.trainKern = trainKern

        if isinstance(kernelInit, int):
            self.kernelSize = kernelInit
            self.kernel = nn.Parameter(torch.empty(self.kernelSize,), requires_grad=self.trainKern)
            nn.init.normal_(self.kernel)
        else:
            self.kernelSize = len(kernelInit)
            self.kernel = nn.Parameter(torch.Tensor(kernelInit).view(self.kernelSize,), requires_grad=self.trainKern)

    def forward(self, inputs):
        
        """
        Forward pass for the Kernel. It returns the kernel itself since it's a learnable parameter.
        """
        
        return self.kernel


class LowPassWave(nn.Module):
    
    """
    Performs low-pass filtering on the input signal using a convolution operation with stride 2.
    """

    def __init__(self):
        super(LowPassWave, self).__init__()

    def forward(self, inputs):
        
        """
        Applies low-pass filtering by convolving the input with the kernel.

        Parameters:
        - inputs (tuple): A tuple containing the input signal and the filter kernel.

        Returns:
        - The low-pass filtered signal.
        """
        
        return F.conv1d(inputs[0], inputs[1].view(1,1,-1).to(device), padding=0, stride=2)


class HighPassWave(nn.Module):
    
    """
    Performs high-pass filtering by convolving the input signal with a reversed and sign-flipped kernel.
    """

    def __init__(self):
        super(HighPassWave, self).__init__()

    def initialize_qmfFlip(self, input_shape):
        
        """
        Initializes the sign-flipping tensor used for high-pass filtering.

        Parameters:
        - input_shape (tuple): Shape of the input signal.
        """
        
        qmfFlip = torch.tensor([(-1) ** i for i in range(input_shape[0])], dtype=torch.float32)
        self.qmfFlip = nn.Parameter(qmfFlip.view(1, 1, -1), requires_grad=False).to(device)

    def forward(self, inputs):
        
        """
        Applies high-pass filtering by convolving the input with a reversed and sign-flipped kernel.

        Parameters:
        - inputs (tuple): A tuple containing the input signal and the filter kernel.

        Returns:
        - The high-pass filtered signal.
        """
        
        if not hasattr(self, 'qmfFlip'):
            
            self.initialize_qmfFlip(inputs[1].shape)
            
        return F.conv1d(inputs[0], torch.mul(torch.flip(inputs[1], [0]).to(device), self.qmfFlip), padding=0, stride=2)



class HardThresholdAssym(nn.Module):
    
    """
    Implements an asymmetrical hard-thresholding function that is learnable and can be applied to input signals.
    """

    def __init__(self, init=None, alpha=None, trainBias=True):
        
        """
        Initializes the HardThresholdAssym object.

        Parameters:
        - init (float, optional): Initial threshold value.
        - alpha (float, optional): Sharpness parameter of the sigmoid function.
        - trainBias (bool): Specifies whether the threshold values are learnable.
        """
        
        super(HardThresholdAssym, self).__init__()
        self.trainBias = trainBias

        if isinstance(init, float) or isinstance(init, int):
            self.init = torch.tensor([init], dtype=torch.float32)
        else:
            self.init = torch.ones(1, dtype=torch.float32)
        
        if isinstance(alpha, float) or isinstance(alpha, int):
            self.alpha = torch.tensor([alpha], dtype=torch.float32)
        else:
            self.alpha = torch.ones(1, dtype=torch.float32)
        
        if torch.cuda.is_available():
            self.init = self.init.to(device)
            self.alpha = self.alpha.to(device)
        
        self.thrP = nn.Parameter(self.init, requires_grad=self.trainBias)
        self.thrN = nn.Parameter(self.init, requires_grad=self.trainBias)
        
        self.alpha = nn.Parameter(self.alpha, requires_grad=self.trainBias)
        
    def forward(self, inputs):
        
        """
        Applies the asymmetric hard thresholding to the input signals.

        Parameters:
        - inputs (Tensor): Input signal tensor.

        Returns:
        - Thresholded signal tensor.
        """

        return inputs * (
            torch.sigmoid(self.alpha * (inputs - self.thrP))
            + torch.sigmoid(-self.alpha * (inputs + self.thrN))
        )
    
class SpatialAttentionBlock(nn.Module):
    
    """
    Implements a linear attention block that applies spatial attention mechanism over 1D signals.
    """
    
    def __init__(self, in_features, normalize_attn=True):
        
        """
        Initializes the LinearAttentionBlock object.

        Parameters:
        - in_features (int): Number of input features.
        - normalize_attn (bool): Specifies whether to normalize attention weights using softmax.
        """
        
        super(SpatialAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv1d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        
        """
        Applies attention to the input features.

        Parameters:
        - l (Tensor): Local input features.
        - g (Tensor): Global context.

        Returns:
        - A tuple of attention scores and attended feature map.
        """
        
        N, C, W = l.size()
        c = self.op(l+g) # batch_sizex1xW
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool1d(g, (1,1)).view(N,C)
        return c.view(N,1,W), g
    

class TemporalAttn(nn.Module):
    
    """
    Implements temporal attention mechanism over sequences of hidden states.
    """
    
    def __init__(self, hidden_size):
        
        """
        Initializes the TemporalAttn object.

        Parameters:
        - hidden_size (int): Dimensionality of the hidden states.
        """
        
        super(TemporalAttn, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        
        """
        Applies temporal attention to the sequence of hidden states.

        Parameters:
        - hidden_states (Tensor): A tensor of shape (batch_size, time_steps, hidden_size) containing the hidden states.

        Returns:
        - The attention vector and attention weights.
        """
        
        # (batch_size, time_steps, hidden_size)
        score_first_part = self.fc1(hidden_states)
        # (batch_size, hidden_size)
        h_t = hidden_states[:,-1,:]
        # (batch_size, time_steps)
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(score, dim=1)
        # (batch_size, hidden_size)
        context_vector = torch.bmm(hidden_states.permute(0,2,1), attention_weights.unsqueeze(2)).squeeze(2)
        # (batch_size, hidden_size*2)
        pre_activation = torch.cat((context_vector, h_t), dim=1)
        # (batch_size, hidden_size)
        attention_vector = self.fc2(pre_activation)
        attention_vector = torch.tanh(attention_vector)

        return attention_vector, attention_weights