# Utility model
import sys
import subprocess
import pyaudio
import wave
import sys
from tkinter import *
import threading
import time
 
class myThread (threading.Thread):
	def __init__(self, threadID, name, counter):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.counter = counter

	def run(self):
		input('Press <Enter> to stop recording.\n')
 

# Run command from python
def RunCmd(cmd):
	try:
		retcode = subprocess.call(cmd, shell=True,timeout=None)
		if retcode < 0:
			print >> sys.stderr, "Child was terminated by signal", -retcode
	except OSError as e:
		print >> sys.stderr, "Execution failed:", e
	return

# Find configuration value by key
def ParseConfig(file_name, key):
	config = open(file_name)
	for line in config:
		tokens = line.replace(' ','').split('=')
		if tokens[0] == key:
			config.close()
			return tokens[1]
	config.close()
	print >> sys.stderr, key + " is not found in " + file_name
	return ''

# MAIN_DIR = '../'
# DICT_DIR = MAIN_DIR+'dict/'
def GetDictionary(filename):
        # dict_file = open(DICT_DIR+filename+'.txt')
        dict_file = open(filename)
        words = []
        ID = []
        for line in dict_file:
                tokens = line.strip().split()
                ID.append(tokens[0])
                words.append(tokens[1])
        dict_file.close()
        return words, ID

def Collect(output_file,instruction=None):
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 44100
	RECORD_SECONDS = 5
	
	if instruction != None:
		input(instruction)
	
	p = pyaudio.PyAudio()
	stream = p.open(format=FORMAT,
				channels=CHANNELS,
				rate=RATE,
				input=True,
				frames_per_buffer=CHUNK)
	print("* recording.")
	frames = []

	wait_thread = myThread(1, "wait_thread", 1)
	wait_thread.start()

	while True:
		data = stream.read(CHUNK)
		frames.append(data)
		if not wait_thread.is_alive():
			break
	print("* done recording.\n")

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(output_file, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()

# def my_button(root,label_text,button_text,button_func):    
#     '''''''function of creat label and button'''    
#     #label details    
#     label = Label(root)    
#     label['text'] = label_text    
#     label.pack()    
#     #label details    
#     button = Button(root)    
#     button['text'] = button_text    
#     button['command'] = button_func    
#     button.pack() 