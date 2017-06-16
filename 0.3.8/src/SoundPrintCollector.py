# Prompt the user to utter scripts
# Record sound print into wave file
import pyaudio
import wave
# import sys
from sys import argv
from os import mkdir
from os.path import isdir
from Utility import Collect,ParseConfig,GetDictionary
from os import getcwd
from tkinter import *

MAIN_DIR = getcwd()+'/'
WAVE_FOLDER = MAIN_DIR + 'wav/'
# SINGLE_FOLDER = WAVE_FOLDER + 'single/'
TRAIN_DIR = WAVE_FOLDER+'train/'
CONFIG_DIR = MAIN_DIR + 'config/'
DICT_DIR = MAIN_DIR + 'dict/'

def main():
	if len(argv) != 4:
		exit("Usage: SoundPrintCollector.py <dict> <config> <output-folder>")

	num_repeat_key = 'NUMREPEAT'	
	# Get configuration
	num_repeat = ParseConfig(CONFIG_DIR + argv[2] + '.conf',num_repeat_key)

	if num_repeat != '':
			num_repeat = int(num_repeat)
	else:
			num_repeat = 1

	words,model_id = GetDictionary(DICT_DIR + argv[1] + '.txt')

	total_num = len(words) * num_repeat

	OUTPUT_DIR = TRAIN_DIR+argv[3]+'/'

	if not isdir(OUTPUT_DIR):
		mkdir(OUTPUT_DIR)

	# Collect sound print for single model
	for i in range(len(words)):
		for k in range(num_repeat):
			total_num -= 1
			print(str(total_num) + ' transcript(s) remaining.')
			if words[i].find('!') > -1:
				instruction = 'Press <Enter> to record background noise.\n' + words[i]
			else:
				instruction = 'Get ready to speak the following script and press <Enter> to start record.\n' + words[i]+'\n Remember to leave 3 seconds of blank before and after the utterance.\n'
			Collect(OUTPUT_DIR + model_id[i] + '-' + str(k) + '.wav',instruction)
	print('Done\a')

if __name__=='__main__':
	main()