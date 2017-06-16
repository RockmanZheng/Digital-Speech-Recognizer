# Embedded train module
# Run over all transcriptions to train all word models
# Including background noise model
import sys
from ModelIO import WriteModel,ReadModel,LoadModels
from Utility import ParseConfig, GetDictionary
from MFCC import load
import numpy as np
from sys import path
from os import getcwd
path.append(getcwd())
from x64.Release.TrainCore import VDecode,BWDecode
from glob import glob
from pdb import set_trace
from os import getcwd
from sys import argv


MAIN_DIR = getcwd() + '/'
MODEL_FOLDER = MAIN_DIR + 'model/'
DICT_DIR = MAIN_DIR + 'dict/'


#########################################################################
#                          MAIN ENTRY #
#########################################################################
if len(argv) != 3:
    exit("Usage: Tester.py <dict> <mfcc-dir>")

MFCC_FOLDER = MAIN_DIR + 'mfcc/train/'+argv[2]+'/'
words, model_id = GetDictionary(DICT_DIR + argv[1] + '.txt')

models = LoadModels(model_id,MODEL_FOLDER)

count = 0
sum = 0
# For each model
for k in range(len(model_id)):
    # Load MFCC data
    # To compute the global mean and log_var
    MFCC_filename_list = glob(MFCC_FOLDER + '*' + model_id[k] + '*.txt')
    feat_list = []
    for filename in MFCC_filename_list:
        feat_list.append(load(filename))

    dim_observation = len(feat_list[0][0])

    #print('VDecode: ' + str(k + 1) + '/' + str(len(model_id)))
    print('BWDecode: ' + str(k + 1) + '/' + str(len(model_id)))


    for i in range(len(feat_list)):
        #print('VDecode: Using ' + str(i + 1) + '/' + str(len(feat_list)) + ' recording')
        #ans = VDecode(models,feat_list[i])
        #print('VDecode: predict = '+str(ans)+', real = '+str(k))

        ans = BWDecode(models,feat_list[i])
        print('BWDecode: predict = '+str(ans)+', real = '+str(k))

        # Wrong prediction
        if ans != k:
            count+=1
        sum += 1

print('Error Rate = '+str(float(count)/sum))
print('Done\a')

