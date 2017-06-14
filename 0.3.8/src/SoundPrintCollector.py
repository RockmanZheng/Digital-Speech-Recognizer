# Prompt the user to utter scripts
# Record sound print into wave file
import pyaudio
import wave
import sys
import Utility

def Collect(output_file,instruction=''):
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 44100
	RECORD_SECONDS = 5
	
	if instruction != '':
		raw_input(instruction)
	
	p = pyaudio.PyAudio()
	stream = p.open(format=FORMAT,
				channels=CHANNELS,
				rate=RATE,
				input=True,
				frames_per_buffer=CHUNK)
	print("* recording. Press <crtl>+<c> to complete the recording.")
	frames = []
	while True:
		try:
			data = stream.read(CHUNK)
			frames.append(data)
		except KeyboardInterrupt:
			break
	print("* done recording.")

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(output_file, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()

MAIN_DIR = '../'
WAVE_FOLDER = MAIN_DIR + 'wav/'
SINGLE_FOLDER = WAVE_FOLDER + 'single/'
CONFIG_DIR = MAIN_DIR + 'config/'
DICT_DIR = MAIN_DIR + 'dict/'
# EMBEDDED_FOLDER = WAVE_FOLDER+'Embedded/'

# if len(sys.argv)<3:
if len(sys.argv) < 2:
	# sys.exit("Usage: SoundPrintCollector.py <dict> <transcipts> <config>")
	sys.exit("Usage: SoundPrintCollector.py <dict> <config>")

num_repeat_key = 'NUMREPEAT'	
# Get configuration
# num_repeat = Utility.ParseConfig(sys.argv[3],num_repeat_key)
# num_repeat = Utility.ParseConfig(sys.argv[2],num_repeat_key)
num_repeat = Utility.ParseConfig(CONFIG_DIR + sys.argv[2] + '.conf',num_repeat_key)

if num_repeat != '':
        num_repeat = int(num_repeat)
else:
        num_repeat = 1

# Get word list
# dict_file = open(sys.argv[1])
# model_id = []
# words = []
# for line in dict_file:
# 	tokens = line.strip().split()
# 	words.append(tokens[1])
# 	model_id.append(tokens[0])
# dict_file.close()

words,model_id = Utility.GetDictionary(DICT_DIR + sys.argv[1] + '.txt')

# # Get scripts
# scripts_file = open(sys.argv[2])
# script_ids = []
# scripts = []
# for line in scripts_file:
# 	tokens = line.strip().split()
# 	script_ids.append(tokens[0])
# 	script = ''
# 	for i in range(1,len(tokens)):
# 		script += tokens[i]+' '
# 	scripts.append(script)
# num_scripts = len(scripts)



# total_num = len(words)*num_repeat+num_scripts
total_num = len(words) * num_repeat


# Collect sound print for single model
for i in range(len(words)):
	for k in range(num_repeat):
		total_num -= 1
		print(str(total_num) + ' transcript(s) remaining.')
		if words[i].find('!') > -1:
			instruction = 'Press <Enter> to record background noise.\n' + words[i]
		else:
			instruction = 'Get ready to speek the following script and press <Enter> to start record.\n Remember to leave 3 seconds of blank before and after the utterance.' + words[i]
		Collect(SINGLE_FOLDER + model_id[i] + '-' + str(k) + '.wav',instruction)

# # Collect sound print for embedded model
# for i in range(num_scripts):
# 	total_num -= 1
# 	print str(total_num)+'transcript(s) remaining.'
# 	instruction = 'Get ready to speek the following script and press <Enter> to start record.\n'+scripts[i]
# 	Collect(EMBEDDED_FOLDER+script_ids[i]+'.wav',instruction)
	
	
	
	
	
	
	
	
	