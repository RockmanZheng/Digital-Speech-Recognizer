from SignalProcessor import Signal,preprocess
from os.path import join,split
from glob import glob
from sys import argv
from os.path import isdir
from os import mkdir,getcwd

MAIN_DIR = getcwd()+'/'
WAVE_FOLDER = MAIN_DIR + 'wav/'
TRAIN_DIR = WAVE_FOLDER+'train/'
# SINGLE_FOLDER = WAVE_FOLDER + 'single/'
# COOKED_FOLDER = WAVE_FOLDER + 'single-cooked/'

if len(argv) != 2:
    sys.argv('Usage: SpeechCooker.py <train-dir>')

COLLECT_DIR = TRAIN_DIR+argv[1]+'/'
COOKED_DIR = TRAIN_DIR+argv[1]+'-cooked/'
if not isdir(COOKED_DIR):
    mkdir(COOKED_DIR)

# Cut: cutting click sounds when starting and closing recording
# Denoise: remove background noise
# Truncate: cut off silence
# Data augmentation: by performing the same tasks to attenuated signals
def cook():
    gain = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    for wavfilename in glob(join(COLLECT_DIR,'*.wav')):
        signal = preprocess(wavfilename)
        cookedfilename = COOKED_DIR + '0' + '-' + split(wavfilename)[1]
        signal.write(cookedfilename)
        for i in range(1,len(gain)):
            signal.attenuate(gain[i] / gain[i - 1])
            cookedfilename = COOKED_DIR + str(i) + '-' + split(wavfilename)[1]
            signal.write(cookedfilename)
    print('Done\a')

cook()