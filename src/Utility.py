# Utility model
import sys
import subprocess

# Run command from python
def RunCmd(cmd):
	try:
		retcode = subprocess.call(cmd, shell=True)
		if retcode < 0:
			print >>sys.stderr, "Child was terminated by signal", -retcode
	except OSError as e:
		print >>sys.stderr, "Execution failed:", e
	return

# Find configuration value by key
def ParseConfig(file_name, key):
	config = open(file_name)
	for line in config:
		tokens = line.replace(' ','').split('=')
		if tokens[0]==key:
			config.close()
			return tokens[1]
	config.close()
	print >>sys.stderr, key+" is not found in "+file_name
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
