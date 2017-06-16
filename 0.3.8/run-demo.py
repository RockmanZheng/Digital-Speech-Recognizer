import src
from src.Utility import RunCmd

print('Welcome to Digital Speech Recognizer (v0.3.8)')
print('Simple demonstration follows these steps:')
print('1. Collect your sound prints. This might take up to 15 minutes (for 11 words).')
print('2. Cook your recordings (data preprocessing and augmentation).')
print('3. Extract MFCC features to feed training procedure.')
print('4. Invoke Viterbi algorithm to train the HMMs.')
print('5. Run forward algorithm to find the word matches the utterance the most. And calculate error rate on training set.')
print('6. Try real-time recognizer.')
print('\n')

WAV_DIR = 'wav/'
TRAIN_DIR = WAV_DIR+'train/'
PYTHON_EXE = 'python'
SCRIPT_DIR = 'src/'
DICT_FILENAME = 'dictionary'
CONFIG_FILENAME = 'project'

train_folder = input('Folder name to place your sound prints.\n')

# train_path = TRAIN_DIR+train_folder

print('Now start recording.')
# RunCmd(PYTHON_EXE+' '+SCRIPT_DIR+'SoundPrintCollector.py '+DICT_FILENAME+' '+CONFIG_FILENAME+' '+train_folder)

print('Now start preprocessing.')
RunCmd(PYTHON_EXE+' '+SCRIPT_DIR+'SpeechCooker.py '+train_folder)

print('Now cook mfcc.')
RunCmd(PYTHON_EXE+' '+SCRIPT_DIR+'mfccCooker.py '+train_folder)

print('Now invoke Viterbi training.')
RunCmd(PYTHON_EXE+' '+SCRIPT_DIR+'ModelInitializer.py '+DICT_FILENAME+' '+CONFIG_FILENAME+' '+train_folder)

print('Now test the models.')
RunCmd(PYTHON_EXE+' '+SCRIPT_DIR+'Tester.py '+DICT_FILENAME+' '+train_folder)

print('Run real-time recognizer.')
RunCmd(PYTHON_EXE+' '+SCRIPT_DIR+'Recognizer.py '+DICT_FILENAME)





