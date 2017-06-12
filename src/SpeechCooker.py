from SignalProcessor import Signal
from os.path import join,split
from glob import glob

MAIN_DIR = '../'
WAVE_FOLDER = MAIN_DIR+'wav/'
SINGLE_FOLDER = WAVE_FOLDER+'single/'
COOKED_FOLDER = WAVE_FOLDER+'single-cooked/'

# Cut: cutting click sounds when starting and closing recording
# Denoise: remove background noise
# Truncate: cut off silence
# Data augmentation: by performing the same tasks to attenuated signals
def cook():
    gain = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    for wavfilename in glob(join(SINGLE_FOLDER,'*.wav')):
        print 'Cooking '+split(wavfilename)[1]
        signal = Signal(filename=wavfilename)
        signal.cut(1.0,signal.during-1.0)
        signal.attenuate(0.8)
        # Use last 1 sec as noise
        noise_data = signal.data[len(signal.data)-signal.rate:len(signal.data)]
        noise = Signal(data=noise_data,rate=signal.rate)
        signal.noise_removal(noise)
        signal.truncate_silence(12)

        cookedfilename = COOKED_FOLDER+'0'+'-'+split(wavfilename)[1]
        signal.write(cookedfilename)
        for i in range(1,len(gain)):
            signal.attenuate(gain[i]/gain[i-1])
            cookedfilename = COOKED_FOLDER+str(i)+'-'+split(wavfilename)[1]
            signal.write(cookedfilename)
        print 'Done'
    print '\a'

cook()