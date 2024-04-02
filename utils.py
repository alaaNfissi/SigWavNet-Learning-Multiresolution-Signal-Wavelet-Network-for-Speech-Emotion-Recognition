#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Title: Utility Functions for Speech Emotion Recognition (SigWavNet)
Author: Alaa Nfissi
Date: March 31, 2024
Description: This file contains utility functions for preprocessing, loading data, and 
performing various auxiliary tasks for the speech emotion recognition project.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm.notebook import tqdm
import math
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
torch.manual_seed(123)
import random
import pywt
random.seed(123)

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from torch import cuda
import gc
import inspect

from sklearn.model_selection import StratifiedKFold

d = 0
device = torch.device(f"cuda:{d}")



def load_data(data_path):
    """
    Loads the dataset from a given CSV file path and processes it.

    Parameters:
    - data_path (str): Path to the CSV file containing the dataset.

    Returns:
    - Tuple containing the processed dataset DataFrame and a list of unique emotion classes.
    """
    data_path = os.path.abspath(data_path)
    
    data = pd.read_csv(data_path)
    data['label'] = data['label'].replace('exc', 'hap')
    data = data[data['label'].isin(['ang', 'hap', 'neu', 'sad'])].reset_index()
    del data['index']
    
    emotionclasses = sorted(list(data.label.unique()))
    
    return data, emotionclasses


def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1, target_variable=None, data_source=None):
    
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
    - df (DataFrame): The dataset to split.
    - train_split (float): Proportion of the dataset to use for training.
    - val_split (float): Proportion of the dataset to use for validation.
    - test_split (float): Proportion of the dataset to use for testing.
    - target_variable (str, optional): Name of the column containing the target variable for stratification.
    - data_source (str, optional): Name of the column containing the data source for stratification.

    Returns:
    - DataFrames for the training, validation, and test sets.
    """
    
    assert (train_split + test_split + val_split) == 1
    
    # Only allows for equal validation and test splits
    #assert val_split == test_split 
    # Shuffle
    df_sample = df.sample(frac=1, random_state=42)

    # Specify seed to always have the same split distribution between runs
    # If target variable is provided, generate stratified sets
    arr_list = []
    if target_variable is not None and data_source is not None:
        grouped_df = df_sample.groupby([data_source, target_variable])
        #arr_list = [np.split(g, [int(train_split * len(g)), int((1 - val_split) * len(g))]) for i, g in grouped_df]
        for i, g in grouped_df:
            if len(g) == 3:
                arr_list.append(np.split(g, 3))
            else:
                arr_list.append(np.split(g, [int(train_split * len(g)), int((1 - val_split) * len(g))]))
        train_ds = pd.concat([t[0] for t in arr_list])
        val_ds = pd.concat([v[1] for v in arr_list])
        test_ds = pd.concat([t[2] for t in arr_list])

    else:
        indices_or_sections = [int(train_split * len(df)), int((1 - val_split) * len(df))]
        train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)
    
    return train_ds.reset_index(drop=True), val_ds.reset_index(drop=True), test_ds.reset_index(drop=True)



class MyDataset(torch.utils.data.Dataset):
    
    """
    Custom PyTorch Dataset for loading and processing the speech emotion recognition dataset.

    Attributes:
    - paths (list): List of file paths to the audio files.
    - labels (list): List of labels corresponding to the audio files.
    - transform (callable): A function/transform that takes in an audio file and returns a transformed version.
    """
    
    def __init__(self, paths, labels, transform):
        self.files = paths
        self.labels = labels
        self.transform = transform
    def __getitem__(self, item):
        #print(self.files)
        file = self.files[item]
        label = self.labels[item]
        file, sampling_rate = torchaudio.load(file)
        file = file if file.shape[0] == 1 else file[0].unsqueeze(0)
        file = self.transform(file)
        
        return file, sampling_rate, label
    
    def __len__(self):
        return len(self.files)
    
def compute_precise_mean_std(file_paths):
    
    """
    Computes the mean and standard deviation of the waveforms in the dataset.

    Parameters:
    - file_paths (list): List of paths to the audio files in the dataset.

    Returns:
    - Tuple containing the global mean and standard deviation of the waveforms.
    """
    
    sum_waveform = 0.0
    sum_squares = 0.0
    total_samples = 0
    
    for file_path in file_paths:
        waveform, _ = torchaudio.load(file_path)
        sum_waveform += waveform.sum()
        sum_squares += (waveform ** 2).sum()
        total_samples += waveform.numel()  # Count total number of samples across all files
    
    # Compute global mean and std
    mean = sum_waveform / total_samples
    std = (sum_squares / total_samples - mean ** 2) ** 0.5
    
    return mean.item(), std.item()

class MyTransformPipeline(nn.Module):
    
    """
    Custom transform pipeline for processing audio data.
    
    Parameters:
    - train_mean (float): The mean of the training data.
    - train_std (float): The standard deviation of the training data.
    - input_freq (int): The original frequency of the audio data.
    - resample_freq (int): The target frequency to resample the audio data.
    """
    
    def __init__(
        self,
        train_mean = 0,
        train_std = 1,
        input_freq=16000,
        resample_freq=16000,
    ):
        super().__init__()
        
        self.train_mean = train_mean
        self.train_std = train_std
        self.input_freq = input_freq
        self.resample_freq = resample_freq
        
        self.resample = torchaudio.transforms.Resample(orig_freq=self.input_freq, new_freq=self.resample_freq).to(device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Resample the input
        waveform = waveform.to(device)
        resampled = self.resample(waveform)
        normalized_waveform = (resampled - self.train_mean) / self.train_std

        return normalized_waveform
    

def count_parameters(model):
    
    """
    Counts the number of trainable parameters in a model.
    
    Parameters:
    - model (torch.nn.Module): The model to count parameters for.
    
    Returns:
    - The number of trainable parameters.
    """
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if device == f"cuda{d}":
    num_workers = 10
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False


def get_dataloaders(data, batch_size=32, num_splits=5, stratify=True):
    
    """
    Creates dataloaders for cross-validation, with optional stratification.

    Parameters:
    - data (DataFrame): The dataset to be loaded into the dataloaders.
    - batch_size (int): Size of batches.
    - num_splits (int): Number of folds for cross-validation.
    - stratify (bool): Whether to stratify the folds based on labels.

    Returns:
    - List of tuples containing train and validation dataloaders for each fold.
    """

    if stratify:
        kf = StratifiedKFold(n_splits=num_splits)
        split_method = kf.split(data, data['label'])
    else:
        kf = KFold(n_splits=num_splits)
        split_method = kf.split(data)
    
    dataloaders = []

    for train_idx, val_idx in split_method:
        train_data, val_data = data.iloc[train_idx], data.iloc[val_idx]
        
        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)
        
        
        train_mean, train_std = compute_precise_mean_std(train_data['path'])
        
        transform = MyTransformPipeline(input_freq=16000, resample_freq=8000, train_mean = train_mean, train_std = train_std)
        
        transform.to(device)
        
        train_dataset = MyDataset(train_data['path'], train_data['label'], transform=transform)
        val_dataset = MyDataset(val_data['path'], val_data['label'], transform=transform)
        
        # Create dataloaders for train and validation sets
        train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        dataloaders.append((train_loader, val_loader))
    
    return dataloaders

_, emotionclasses = load_data('IEMOCAP_dataset.csv')


def index_to_emotionclass(index):
    """
    Converts an index to an emotion class.

    Parameters:
    - index (int): Index of the emotion class in the list of classes.

    Returns:
    - The name of the emotion class corresponding to the given index.
    """
    
    return emotionclasses[index]

def emotionclass_to_index(emotion):
    
    """
    Converts an emotion class to its corresponding index.

    Parameters:
    - emotion (str): The emotion class.

    Returns:
    - Index of the emotion class in the list of classes.
    """
    
    return torch.tensor(emotionclasses.index(emotion))

def pad_sequence(batch):
    
    """
    Pads a batch of tensors to the same length with zeros.

    Parameters:
    - batch (list of Tensor): The batch of tensors to pad.

    Returns:
    - A tensor containing the padded batch.
    """
    
    batch = [item.t() for item in batch]
    
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn(batch):
    
    """
    Custom collate function to process batches of data.

    Parameters:
    - batch (list): A batch of data.

    Returns:
    - Processed batch of tensors and targets.
    """
    
    tensors, targets = [], []

    # Gather in lists, and encode wordclasses as indices
    for waveform, _, emotionclass, *_ in batch:
        tensors += [waveform]
        targets += [emotionclass_to_index(emotionclass)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    # stack - Concatenates a sequence of tensors along a new dimension
    targets = torch.stack(targets)

    return tensors, targets


def get_less_used_gpu(gpus=None, debug=False):
    
    """
    Finds the least utilized GPU for use.

    Parameters:
    - gpus (list): List of GPUs to consider. If None, considers all available GPUs.
    - debug (bool): Whether to print debug information.

    Returns:
    - ID of the least used GPU.
    """
    
    if gpus is None:
        warn = 'Falling back to default: all gpus'
        gpus = range(cuda.device_count())
    elif isinstance(gpus, str):
        gpus = [int(el) for el in gpus.split(',')]

    # check gpus arg VS available gpus
    sys_gpus = list(range(cuda.device_count()))
    if len(gpus) > len(sys_gpus):
        gpus = sys_gpus
        warn = f'WARNING: Specified {len(gpus)} gpus, but only {cuda.device_count()} available. Falling back to default: all gpus.\nIDs:\t{list(gpus)}'
    elif set(gpus).difference(sys_gpus):
        # take correctly specified and add as much bad specifications as unused system gpus
        available_gpus = set(gpus).intersection(sys_gpus)
        unavailable_gpus = set(gpus).difference(sys_gpus)
        unused_gpus = set(sys_gpus).difference(gpus)
        gpus = list(available_gpus) + list(unused_gpus)[:len(unavailable_gpus)]
        warn = f'GPU ids {unavailable_gpus} not available. Falling back to {len(gpus)} device(s).\nIDs:\t{list(gpus)}'

    cur_allocated_mem = {}
    cur_cached_mem = {}
    max_allocated_mem = {}
    max_cached_mem = {}
    for i in gpus:
        cur_allocated_mem[i] = cuda.memory_allocated(i)
        cur_cached_mem[i] = cuda.memory_reserved(i)
        max_allocated_mem[i] = cuda.max_memory_allocated(i)
        max_cached_mem[i] = cuda.max_memory_reserved(i)
    min_allocated = min(cur_allocated_mem, key=cur_allocated_mem.get)
    if debug:
        print(warn)
        print('Current allocated memory:', {f'cuda:{k}': v for k, v in cur_allocated_mem.items()})
        print('Current reserved memory:', {f'cuda:{k}': v for k, v in cur_cached_mem.items()})
        print('Maximum allocated memory:', {f'cuda:{k}': v for k, v in max_allocated_mem.items()})
        print('Maximum reserved memory:', {f'cuda:{k}': v for k, v in max_cached_mem.items()})
        print('Suggested GPU:', min_allocated)
    return min_allocated


def free_memory(to_delete: list, debug=False):
    
    """
    Frees up memory by deleting specified variables and collecting garbage.

    Parameters:
    - to_delete (list): List of variable names to delete.
    - debug (bool): Whether to print debug information before and after freeing memory.
    """
    
    calling_namespace = inspect.currentframe().f_back
    if debug:
        print('Before:')
        get_less_used_gpu(debug=True)

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        cuda.empty_cache()
    if debug:
        print('After:')
        get_less_used_gpu(debug=True)
        

class FocalLoss(nn.Module):
    
    """
    Implementation of the Focal Loss as a PyTorch module.

    Parameters:
    - alpha (Tensor): Weighting factor for the positive class.
    - gamma (float): Focusing parameter to adjust the rate at which easy examples contribute to the loss.
    """
    
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha.to(device)
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='mean')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss
    
    
def nr_of_right(pred, target):
    
    """
    Counts the number of correct predictions.

    Parameters:
    - pred (Tensor): Predicted labels.
    - target (Tensor): True labels.

    Returns:
    - Number of correct predictions.
    """
    
    return pred.squeeze().eq(target).sum().item()

def get_probable_idx(tensor):
    
    """
    Finds the indices of the most probable class for each element in the batch.

    Parameters:
    - tensor (Tensor): Tensor containing class probabilities for each element.

    Returns:
    - Tensor of indices for the most probable class.
    """
    
    return tensor.argmax(dim=-1)


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14, normalize=True):
    
    """
    Prints a confusion matrix using seaborn.

    Parameters:
    - confusion_matrix (ndarray): The confusion matrix to print.
    - class_names (list): List of class names corresponding to the indices of the confusion matrix.
    - figsize (tuple): Size of the figure.
    - fontsize (int): Font size for the labels.
    - normalize (bool): Whether to normalize the values in the confusion matrix.
    """
    
    fig = plt.figure(figsize=figsize)
    if normalize:
        confusion_matrix_1 = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
        print("Normalized confusion matrix")
    else:
        confusion_matrix_1 = confusion_matrix
        print('Confusion matrix, without normalization')
    df_cm = pd.DataFrame(
        confusion_matrix_1, index=class_names, columns=class_names
    )
    labels = (np.asarray(["{:1.2f} % \n ({})".format(value, value_1) for value, value_1 in zip(confusion_matrix_1.flatten(),confusion_matrix.flatten())])).reshape(confusion_matrix.shape)
    try:
        heatmap = sn.heatmap(df_cm, cmap="Blues", annot=labels, fmt='' if normalize else 'd')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
        
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')