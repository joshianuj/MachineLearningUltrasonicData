#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:31:49 2019

@author: ajoshi
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
import os

def list_filenames(filename='Stopper/1st/*_80'):
    filenames = glob(filename)
    #return pd.concat([pd.read_csv(f, header=None) for f in filenames])
    return filenames

def getFolderlist(main='Stopper'):
    distance = np.arange(60, 310, 10)
    if main == 'Stopper':
      a = ['./DataCleaning/NewData/%s/*/*_%scm_AS_*.dat'%(main,x) for x in distance]
    else:
      a = ['./DataCleaning/NewData/%s/*/*/*_%scm_AS_*.dat'%(main,x) for x in distance]
    aggregate_folder = [list_filenames(filename) for filename in a]
    # remove empty list
    aggregate_folder = [x for x in aggregate_folder if x]
    return aggregate_folder

def createDataFrametoFile(filename='Stopper'):
    folders = getFolderlist(filename)
    #for stopper
    all_data = []
    for folder in folders:
        data_frames = pd.concat([pd.read_csv(f, header=None).T for f in folder])
        folder_name = folder[0].split('/')[-1].split('.')
        folder_name = folder_name[0][-14:]
        all_data = pd.concat(all_data,[data_frames])
    all_data.to_csv('./data/all/stopper.csv', index=False, header=None)

if __name__ == '__main__':
    createDataFrametoFile('Stopper')