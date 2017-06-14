# Embedded train module
# Run over all transcriptions to train all word models
# Including background noise model
import sys
from ModelIO import WriteModel,ReadModel
from Utility import ParseConfig, GetDictionary
from MFCC import load
import numpy as np
from x64.Release.TrainCore import BWTrain
from glob import glob
from pdb import set_trace
from os import getcwd


MAIN_DIR = getcwd() + '/'
MFCC_FOLDER = MAIN_DIR + 'mfcc/single/'
MODEL_FOLDER = MAIN_DIR + 'model/'
DICT_DIR = MAIN_DIR + 'dict/'
CONFIG_DIR = MAIN_DIR + 'config/'

# Get iteration time limit from config
# Get configuration
conf_filename = CONFIG_DIR + sys.argv[2] + '.conf'
max_iter = int(ParseConfig(conf_filename,'MAXITER'))


#########################################################################
#                          MAIN ENTRY #
#########################################################################
if len(sys.argv) < 3:
    sys.exit("Usage: EmbeddedTrain.py <dict> <config>")

words, model_id = GetDictionary(DICT_DIR + sys.argv[1] + '.txt')

# For each model
for k in range(len(model_id)):
    # Load MFCC data
    # To compute the global mean and log_var
    MFCC_filename_list = glob(MFCC_FOLDER + '*' + model_id[k] + '*.txt')
    feat_list = []
    for filename in MFCC_filename_list:
        feat_list.append(load(filename))

    dim_observation = len(feat_list[0][0])

    # Load model
    model_filename = MODEL_FOLDER + model_id[k] + '.xml'
    name,states,num_states,num_components,dim_observation,log_trans,log_coef,mean,log_var = ReadModel(model_filename)

    print('BWTrain: ' + str(k + 1) + '/' + str(len(model_id)))

    for i in range(len(feat_list)):
        print('BWTrain: Using ' + str(i + 1) + '/' + str(len(feat_list)) + ' recording')
        log_trans, log_coef, mean, log_var = BWTrain(log_trans,log_coef,mean,log_var,feat_list[i])
        
    # Write in model file
    WriteModel(name, states, num_states, num_components, dim_observation,
               log_trans, log_coef, mean, log_var, MODEL_FOLDER + model_id[k] + '.xml')

print('Done\a')

