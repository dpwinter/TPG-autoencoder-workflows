import os
from subprocess import Popen, PIPE, call, check_output

#helper function to source the cmssw code
def source(script, update=1):
	pipe=Popen(". %s; env" % script, stdout=PIPE, shell=True)
	data=pipe.communicate()[0]
	
	env=dict([line.split("=", 1) for line in data.splitlines() if len(line.split("=", 1))==2])
	if update:
		os.environ.update(env)
	return env