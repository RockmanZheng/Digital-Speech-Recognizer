from MFCC import mfcc
from os.path import join,split
from glob import glob
from SignalProcessor import Signal

MAIN_DIR = '../'
WAVE_FOLDER = MAIN_DIR + 'wav/'
COOKED_FOLDER = WAVE_FOLDER + 'single-cooked/'
MFCC_DIR = MAIN_DIR + 'mfcc/single/'


def cook():
    for wavfilename in glob(join(COOKED_FOLDER,'*.wav')):
        print('Cooking ' + split(wavfilename)[1])
        signal = Signal(wavfilename)
        feat = mfcc(signal.data,signal.rate)
        outname = MFCC_DIR + split(wavfilename)[1].replace('wav','txt')
        out = open(outname,'w')
        # out.write(str(len(feat))+' '+str(len(feat[0]))+'\n')
        for i in range(len(feat)):
            out.write(str(list(feat[i])).strip('[]').replace(' ','') + '\n')
        out.close()
        print('Done')

cook()
