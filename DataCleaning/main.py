#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:31:49 2019

@author: ajoshi
"""
import pandas as pd
import numpy as np
from glob import glob

def list_filenames(filename='Stopper/1st/*_80'):
    filenames = glob(filename)
    #return pd.concat([pd.read_csv(f, header=None) for f in filenames])
    return filenames

def LoadWallDataFrame():
    wall_filenames = glob('NewData/Wall/1st/C1wall_60cm_AS_*.dat')
    return pd.concat([pd.read_csv(f, header=None) for f in wall_filenames])


def getFolderlist(main='Stopper'):
    distance = np.arange(60, 300, 10)
    a = ['./MachineLearningUltrasonicData/DataCleaning/NewData/%s/*/*_%scm_AS_*.dat'%(main,x) for x in distance]
    aggregate_folder = [list_filenames(filename) for filename in a]
    return aggregate_folder

def createStopperDataFrameToFile(filename='Stopper'):
    folders = getFolderlist(filename)
    
    # remove empty list
    folders = [x for x in folders if x != []]

    #for stopper
    for folder in folders:
        temp = []
        df3 = pd.DataFrame(data={})
        for file in folder:
            df = pd.read_csv(file, header=None)
            temp = df.transpose()
            df3 = df3.append(temp)
        folder_digit = folder[0].split('/')[-1]
        folder_digit = ''.join(filter(str.isdigit, folder_digit))
        df3.to_csv('./MachineLearningUltrasonicData/data/%s/%s.csv'%(filename,folder_digit), sep=',')

if __name__ == '__main__':
    createStopperDataFrameToFile()
    createStopperDataFrameToFile('Wall')
    createStopperDataFrameToFile('Wall2')


