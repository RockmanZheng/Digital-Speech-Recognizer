from SignalProcessor import Signal
from os.path import join,split
from glob import glob

MAIN_DIR = '../'
WAVE_FOLDER = MAIN_DIR+'wav/'
SINGLE_FOLDER = WAVE_FOLDER+'single/'
COOKED_FOLDER = WAVE_FOLDER+'single-cooked/'


def cook():
    # Load data
    # noise = Signal(SINGLE_FOLDER+'00-0.wav')
    # noise.cut(1.0,noise.during-1.0)
    for wavfilename in glob(join(SINGLE_FOLDER,'*.wav')):
        print 'Cooking '+split(wavfilename)[1]
        signal = Signal(filename=wavfilename)
        signal.cut(1.0,signal.during-1.0)
        signal.attenuate(0.8)
        # Use last 1 sec as noise
        noise_data = signal.data[len(signal.data)-signal.rate:len(signal.data)]
        noise = Signal(data=noise_data,rate=signal.rate)

        signal.noise_removal(noise)
        signal.truncate_silence(1000000)
        cookedfilename = COOKED_FOLDER+split(wavfilename)[1]
        signal.write(cookedfilename)
        print 'Done'

cook()