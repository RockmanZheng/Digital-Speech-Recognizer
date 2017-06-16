# CreateModel.py
# Create file which only specifies the network of single models
import math
from ModelIO import WritePreModel
from os import getcwd
from Utility import ParseConfig
from sys import argv

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


MAIN_DIR = getcwd()+'/'
PREMODEL_FOLDER = MAIN_DIR + 'premodel/'
DICT_DIR = MAIN_DIR + 'dict/'
CONFIG_DIR = MAIN_DIR + 'config/'

#########################################################################
#                          MAIN ENTRY #
#########################################################################
if len(argv) != 3:
    exit("Usage: ModelCreator.py <dict> <config>")

dict_file = open(DICT_DIR + argv[1] + '.txt')
tokens_list = []

for line in dict_file:
    tokens_list.append(line.strip().split())

dict_file.close()


num_subphones_key = 'NUMSUBPHONES'	
# Get configuration
num_subphones = int(ParseConfig(CONFIG_DIR + argv[2] + '.conf',num_subphones_key))


for tokens in tokens_list:
    ID = tokens[0]
    name = tokens[1]
    states = ['<START>']
    for i in range(3, len(tokens)):
        for k in range(num_subphones):
            states.append('<' + tokens[i] + str(k)+'>')

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
