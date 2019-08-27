#!/usr/bin/envpython3
#-*-coding: utf-8 -*-
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

def getFolderlist(main='Stopper', min=80, max=280):
    distance = np.arange(min, max, 10)
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
    for folder in folders:
        data_frames = pd.concat([pd.read_csv(f, header=None).T for f in folder])
        folder_name = folder[0].split('/')[-1].split('.')
        folder_name = folder_name[0][-14:]
        train, predict = train_test_split(data_frames, test_size=0.2)
        train.to_csv('./data/train/%s/%s.csv'%(filename,folder_name), index=False, header=None)
        predict.to_csv('./data/predict/%s/%s.csv'%(filename,folder_name), index=False, header=None)

def noSplitCreateDataFrame(filename='Stopper'):
    folders = getFolderlist(filename)
    #for stopper
    for folder in folders:
        data_frames = pd.concat([pd.read_csv(f, header=None).T for f in folder])
        folder_name = folder[0].split('/')[-1].split('.')
        folder_name = folder_name[0][-14:]
        data_frames.to_csv('./data/all/%s/%s.csv'%(filename,folder_name), index=False, header=None)

def allInOneDataframe(filename='Stopper', min=80,max=110):
    folders = getFolderlist(filename,min,max)
    #for stopper
    all_dataframe = pd.DataFrame()
    for folder in folders:
      data_frames = pd.concat([pd.read_csv(f, header=None).T for f in folder])
      folder_name = folder[0].split('/')[-1].split('.')
      folder_name = folder_name[0][-14:]
      all_dataframe = pd.concat([all_dataframe, data_frames])
    train, predict = train_test_split(all_dataframe, test_size=0.3,random_state=42)
    train.to_csv('./final_data/%s_train_%s_to_%s.csv'%(filename,min,max), index=False, header=None)
    predict.to_csv('./final_data/%s_predict_%s_to_%s.csv'%(filename,min,max), index=False, header=None)
    # all_dataframe.to_csv('./final_data/%s_overall_80to270.csv'%(filename), index=False, header=None)

def allInOneDataframe_all(filename='Stopper', min=80,max=110):
    folders = getFolderlist(filename,min,max)
    #for stopper

    all_dataframe = pd.DataFrame()
    for folder in folders:
      data_frames = pd.concat([pd.read_csv(f, header=None).T for f in folder])
      folder_name = folder[0].split('/')[-1].split('.')
      folder_name = folder_name[0][-14:]
      all_dataframe = pd.concat([all_dataframe, data_frames])
    all_dataframe.to_csv('./data/all/_all_%s_%s_to_%s.csv'%(filename,min,max), index=False, header=None)

def newData_in_one(filename='Stopper', min=80,max=110):
    filename_list = './data/all/new_data_april_last/%s_*.csv'%filename
    aggregate_folder = list_filenames(filename_list)
    # remove empty list
    folders = [x for x in aggregate_folder if x]
    #for stopper
    all_dataframe = pd.DataFrame()
    for folder in folders:
      data_frames = pd.read_csv(folder, header=None)
      all_dataframe = pd.concat([all_dataframe, data_frames])
    # train, predict = train_test_split(all_dataframe, test_size=0.3,random_state=42)
    all_dataframe.to_csv('./final_data/%s_overall.csv'%(filename), index=False, header=None)
    # predict.to_csv('./final_data/%s_predict_aggregate.csv'%(filename), index=False, header=None)

if __name__ == '__main__':
    # createDataFrametoFile('Stopper')
    # createDataFrametoFile('Wall_Collection')
    # noSplitCreateDataFrame('Stopper')
    # noSplitCreateDataFrame('Wall_Collection')
    allInOneDataframe('Stopper',80,270)
    allInOneDataframe('Wall_Collection',80,270)
    
    # allInOneDataframe_all('Stopper',80,180)
    # allInOneDataframe_all('Wall_Collection',80,180)
    # newData_in_one('Wall')