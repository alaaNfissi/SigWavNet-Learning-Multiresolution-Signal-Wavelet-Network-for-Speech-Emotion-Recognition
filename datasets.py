#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Title: Dataset Preparation for Speech Emotion Recognition
Author: Alaa Nfissi
Date: March 31, 2024
Description: This script is responsible for preparing the speech emotion recognition datasets, 
including loading, partitioning, and preprocessing the data for model training and evaluation.
"""

import os
import pandas as pd
import torchaudio


# Define the path to the IEMOCAP dataset. This needs to be updated to the correct path where the IEMOCAP dataset is stored.
iemocap_data_path = "IEMOCAP DATA FOLDER PATH"
IEMOCAP_path = os.path.abspath(iemocap_data_path)
dir_list_IEMOCAP = os.listdir(IEMOCAP_path)

# Initialize lists to hold the paths to audio files and label files.
records = []
label_files = []

# Process each session of the IEMOCAP dataset. There are five sessions in total.
for i in range(1,6):
    # List directories for each session.
    wav_list = os.listdir(IEMOCAP_path+f'/IEMOCAP/Session{i}/sentences/wav/')
    
    # Extend the records list with paths to each wav file.
    for j in wav_list:
        records.extend([IEMOCAP_path+f'/IEMOCAP/Session{i}/sentences/wav/'+str(j)+'/'+k for k in os.listdir(IEMOCAP_path+f'/IEMOCAP/Session{i}/sentences/wav/'+str(j)+'/')])
    
    # List emotion label files for each session.
    label_list = os.listdir(IEMOCAP_path+f'/IEMOCAP/Session{i}/dialog/EmoEvaluation/')
    
    # Append label file paths to the label_files list.
    for k in label_list:
        if len(str(k).split('.')) == 2: # Check if the file name format is correct.
            label_files.append(IEMOCAP_path+f'/IEMOCAP/Session{i}/dialog/EmoEvaluation/'+str(k))

# Create a dictionary to map label files to their corresponding audio files.
dic = {}
for i in label_files:
    dic.update({i : [j for j in records if j.split('/')[14].startswith(i.split('/')[13].split('.')[0])]})

# Map audio file paths to their respective emotions.
segments_emotions = {}
for i in dic.keys():
    with open(i) as f:
        for line in f:
            if i.split('/')[13].split('.')[0] in line:
                segments_emotions.update({ [j for j in dic.get(i) if line.split('\t')[1]+'.wav' in j][0] 
                                        : line.split('\t')[2] })

# Convert the dictionary to a DataFrame and save it as a CSV file.
IEMOCAP_df = pd.DataFrame({'path': segments_emotions.keys(), 'source': 'IEMOCAP', 'label':segments_emotions.values()})
IEMOCAP_df.to_csv('IEMOCAP_dataset.csv', index=False)
IEMOCAP_df.head()