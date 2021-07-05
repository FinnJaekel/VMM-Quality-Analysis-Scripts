import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.optimize import curve_fit 

def parabola(x,a,b,c):
	return a*pow((x-b),2)+c

def parabola2(x,a,b,c):
	return a*pow(x,2)+b*x+c
	
def getlineup(leng,arr,up,factor):
	line = np.empty(leng)
	if(up==0):
		line.fill(np.average(arr)+factor*np.std(arr))
	else:
		line.fill(np.average(arr)-factor*np.std(arr))
	return line

fname = 'hybridOverview210615.txt'
allbChans = np.empty(0)
binss = np.linspace(-0.5,127.5,129)
channelList = []
with open(fname) as f_in:
  	lines = f_in.readlines()
#	for line in lines:
#		line = line.strip('{')
#		line = line.strip('}')
#		newlines.append(line)
	data = np.genfromtxt(lines,dtype=str,delimiter = '\t')
	for lin in data:
		newlin = []
		for x in lin:
			x = x.replace('{','')
			x = x.replace('}','')
			newlin = np.append(newlin,[x])
		VMMClass = newlin[5]
		VMMNo = int(newlin[4])
		print(newlin[0],newlin[6])
		if(VMMClass != 'E' and VMMClass !='A' and VMMClass !='D' and VMMClass != 'E-' and VMMClass !='A-' and VMMClass !='D-'):
			brokenChans = np.fromstring(newlin[6],sep=',')
			brokenChans = brokenChans + VMMNo*64
			allbChans = np.append(allbChans,brokenChans)
			channelList.append(brokenChans)
			# ~ plt.figure(0)
			# ~ plt.hist(brokenChans,bins=binss,stacked = True)

print(channelList)
unique, counts = np.unique(allbChans, return_counts=True)
print(dict(zip(unique, counts)))
plt.figure(0)
plt.hist(channelList,bins=binss,stacked=True)
	


plt.figure(1)		
plt.hist(allbChans,bins=binss)
plt.show()
