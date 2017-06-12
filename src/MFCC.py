from scipy.io.wavfile import read
from scipy.fftpack import fft,ifft,dct
from scipy.signal import get_window
import numpy as np
from pdb import set_trace

# linear scale to mel scale 
def mel_scale(f):
    return 1127.0*np.log(1+f/700.0)

# mel scale to linear scale
def linear_scale(m):
    return 700.0*(np.exp(m/1127.0)-1)

# Pre-emphasize the signal, strengthen its high frequency components
def pre_emph(x):
    return x - 0.95*np.hstack((x[len(x)-1:len(x)],x[0:len(x)-1]))

# Frame signal
def _frame(x,width=512):
    frames = []
    hann_win = get_window('hann',width)
    step = width/2
    pad_size = step-len(x)%step+1
    num_frames = (len(x)-width+pad_size)/step+1
    zeros = np.zeros(pad_size)
    data = np.hstack((x,zeros))

    zeros = np.zeros(step)
    data = np.hstack((zeros,data))
    num_frames += 1

    for i in range(num_frames):
        start = i*step
        frame = data[start:start+width]*hann_win
        frames.append(frame)
    return frames

def triang_win(width,center=0.5):
    win = []
    cpos = center*width
    for i in range(width+1):
        if i<=cpos:
            win.append(1.0/cpos*i)
        else:
            win.append(float(width-i)/(width-cpos))
    return np.array(win)[0:width]

def frame_mfcc(frame,rate):
    width = len(frame)
    spectrum = fft(frame)[0:width/2+1]
    linear_upperbound = 44100.0/2
    linear_lowerbound = 0.0
    mel_upperbound = mel_scale(linear_upperbound)
    mel_lowerbound = mel_scale(linear_lowerbound)
    # Number of MFCC entries
    N = 13
    mel_step = (mel_upperbound-mel_lowerbound)/N
    mel_center = map(lambda i:(i+0.5)*mel_step,range(N))
    
    linear_center = [linear_lowerbound]+map(linear_scale,mel_center)+[linear_upperbound]
    banks = []
    # frequency step of the FFT output
    freq_unit = float(rate)/(width+2)
    for i in range(N):
        length = linear_center[i+2]-linear_center[i]
        center = (linear_center[i+1]-linear_center[i])/length
        win_size = int(length/freq_unit)
        banks.append(triang_win(win_size,center))

    energy = []
    for i in range(N):
        start = int(linear_center[i]/freq_unit)
        energy.append(np.log(1e-25+sum(map(lambda x:np.power(np.abs(x),2),spectrum[start:start+len(banks[i])]*banks[i]))))
        # energy.append(sum(map(lambda x:np.power(np.abs(x),2),spectrum[start:start+len(banks[i])]*banks[i])))
        
    energy = np.array(energy)
    # obtain mfcc
    mfcc = dct(energy)
    # replace first cepstral coefficient with log of frame energy
    # frame_energy = np.log(1e-25+sum(map(lambda x:np.power(np.abs(x),2),frame)))
    # frame_energy = sum(map(lambda x:np.power(np.abs(x),2),frame))
    # set_trace()
    # mfcc[0] = frame_energy
    return mfcc

def diff(coefs):
    dim = len(coefs[0])
    cs = coefs[:]
    cs.append(np.zeros(dim))
    deltas = []
    for i in range(len(cs)-1):
        deltas.append(cs[i+1]-cs[i])
    return deltas

def mfcc(x,rate):
    x = pre_emph(x)
    frames = _frame(x)

    mfccs = []
    for frame in frames:
        mfccs.append(frame_mfcc(frame,rate))
    
    # Compute velocity
    vel = diff(mfccs)
    # Compute acceleration
    acc = diff(vel)

    for i in range(len(mfccs)):
        # mfcc + velocity + acceleration + time
        # mfccs[i] = np.hstack((mfccs[i],vel[i],acc[i],np.array([float(i)])))
        mfccs[i] = np.hstack((mfccs[i],vel[i],acc[i]))


    return mfccs
