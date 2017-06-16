from MFCC import mfcc,cook
from os.path import join,split,isdir
from glob import glob
from SignalProcessor import Signal
from os import getcwd,mkdir
from sys import argv



MAIN_DIR = getcwd()+'/'
WAVE_FOLDER = MAIN_DIR + 'wav/'
# COOKED_FOLDER = WAVE_FOLDER + 'single-cooked/'
MFCC_DIR = MAIN_DIR + 'mfcc/train/'

TRAIN_DIR = WAVE_FOLDER+'train/'

if len(argv) != 2:
    argv('Usage: mfccCooker.py <train-dir>')

COLLECT_DIR = TRAIN_DIR+argv[1]+'/'
COOKED_DIR = TRAIN_DIR+argv[1]+'-cooked/'
OUTPUT_DIR = MFCC_DIR+argv[1]+'/'

if not isdir(OUTPUT_DIR):
    mkdir(OUTPUT_DIR)

def _cook():
    for wavfilename in glob(join(COOKED_DIR,'*.wav')):
        cook(wavfilename,OUTPUT_DIR)
    print('Done\a')

_cook()
