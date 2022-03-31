#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author faustineljc
#@date 2022/3/30
#@file pretrain_HuBERT_featureextractions.py

'''

tutorials
1.https://stackoverflow.com/questions/52074153/cannot-convert-list-to-array-valueerror-only-one-element-tensors-can-be-conver
2.can't convert list to array. vaule error only one element tensors can be converted to python scalars
https://stackoverflow.com/questions/52074153/cannot-convert-list-to-array-valueerror-only-one-element-tensors-can-be-conver

'''
import os
import  numpy as np
import torch
import torchaudio
import pandas as pd


#load huBERT models
bundle = torchaudio.pipelines.HUBERT_BASE
model = bundle.get_model()



def HuBERT_featureextraction(csv_files_train, out_csv_files_train):
    #tmp_feature = []

    for csv_file in os.listdir(csv_files_train):
        output_csv_files_train = out_csv_files_train + '/' + csv_file[:-4] + '.npy'

        waveform, sample_rate = torchaudio.load(csv_files_train + '/' + csv_file)
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        features, _ = model.extract_features(waveform)

        #features are list type
        # [t.size() for t in features]

        #save hubert extractor features
        with torch.no_grad():
            probs = [t.numpy() for t in features]

        np_features = np.array(probs)
        np.save(output_csv_files_train,np_features)
        print('np_features shape:', np_features.shape)
        # print(out)

        # tmp_feature.append(feature)

if __name__ == '__main__':


    #csv path
    csv_files_train = 'C:/Users/faustineljc/Desktop/dev_wav'
    out_csv_files_train = 'C:/Users/faustineljc/dev_out'
    #csv_files = ['dia0_utt%s.csv' % i for i in range(len(csv_files_train)]
    #print(csv_files)

    #HuBERT feature extractions
    HuBERT_featureextraction(csv_files_train, out_csv_files_train)