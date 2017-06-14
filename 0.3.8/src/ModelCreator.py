# CreateModel.py
# Create file which only specifies the network of single models
import sys
import math
from ModelIO import WritePreModel


def get_row_idx(num_states):
    row_idx = [0]
    for i in range(1, num_states - 1):
        for k in range(2):
            row_idx.append(i)
    return row_idx


def get_col_idx(num_states):
    col_idx = [1]
    for i in range(1, num_states - 1):
        for k in range(2):
            col_idx.append(i + k)
    return col_idx


MAIN_DIR = '../'
MFCC_FOLDER = MAIN_DIR + 'mfcc/single/'
MODEL_FOLDER = MAIN_DIR + 'model/'
PREMODEL_FOLDER = MAIN_DIR + 'premodel/'
DICT_DIR = MAIN_DIR + 'dict/'

#########################################################################
#                          MAIN ENTRY #
#########################################################################
if len(sys.argv) < 2:
    sys.exit("Usage: ModelCreator.py <dict>")

dict_file = open(DICT_DIR + sys.argv[1] + '.txt')
tokens_list = []

for line in dict_file:
    tokens_list.append(line.strip().split())

dict_file.close()

for tokens in tokens_list:
    ID = tokens[0]
    name = tokens[1]
    states = ['<START>']
    for i in range(3, len(tokens)):
        states.append('<' + tokens[i] + '->')
        states.append('<' + tokens[i] + '~>')
        states.append('<' + tokens[i] + '+>')

    states.append('<END>')
    num_states = len(states)

    # Get flat start log transition matrix
    row_idx = get_row_idx(num_states)
    col_idx = get_col_idx(num_states)
    #log_val = get_log_val(num_states)

    network = [row_idx, col_idx]

    pre_model_file = PREMODEL_FOLDER + tokens[0] + '.xml'

    # Write in
    WritePreModel(name, states, num_states, network, pre_model_file)
