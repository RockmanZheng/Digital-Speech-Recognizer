from Utility import GetDictionary,Collect
from ModelIO import LoadModels
import sys
from sys import path
from os import getcwd
path.append(getcwd())
from x64.Release.TrainCore import VDecode,BWDecode
from MFCC import load,cook
from SignalProcessor import preprocess
from os.path import join,split
from glob import glob

MAIN_DIR = getcwd()+'/'
WAVE_FOLDER = MAIN_DIR + 'wav/'
TEST_FOLDER = WAVE_FOLDER + 'test/'
CONFIG_DIR = MAIN_DIR + 'config/'
DICT_DIR = MAIN_DIR + 'dict/'
MFCC_DIR = MAIN_DIR + 'mfcc/test/'
MODEL_FOLDER = MAIN_DIR + 'model/'

if len(sys.argv) != 2:
    sys.exit("Usage: Recognizer.py <dict>")

words, model_id = GetDictionary(DICT_DIR + sys.argv[1] + '.txt')
models = LoadModels(model_id,MODEL_FOLDER)

instruction = 'Get ready to speak (0~9) and press <Enter> to start record.\n Remember to leave 3 seconds of blank before and after the utterance.'
filename = TEST_FOLDER +'temp.wav'
Collect(filename,instruction)

signal = preprocess(filename)

cooked_filename = TEST_FOLDER+'cooked.wav'
signal.write(cooked_filename)

cook(cooked_filename,MFCC_DIR)

mfcc_filename = MFCC_DIR+split(cooked_filename)[1].replace('wav','txt')

mfcc = load(mfcc_filename)

ans = BWDecode(models,mfcc)
print('BWDecode: predict = '+str(ans))
print('BWDecode: predict = '+words[ans])