import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.optimize import curve_fit 
import pandas as pd
import csv

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
	
fname = 'pedestals.txt'
with open(fname) as f_in:
	lines = f_in.readlines()
#	for line in lines:
#		line = line.strip('{')
#		line = line.strip('}')
#		newlines.append(line)
	data = np.genfromtxt(lines,dtype=str)
	ally = np.array(0)
	for lin in data:
		newlin = []
		for x in lin:
			x = x.replace('{','')
			x = x.replace('}','')
			newlin = np.append(newlin,[x])
		x1=np.fromstring(newlin[6],sep=',')
		y1=np.fromstring(newlin[7],sep=',')
		y2=np.fromstring(newlin[8],sep=',')
		ally = np.append(ally,y1)
		ally = np.append(ally,y2)
		plt.figure(0)
		plt.plot(x1,y1,linewidth=0,marker ='x',label=lin[1]+' VMM 0')
		plt.plot(x1,y2,linewidth=0,marker ='x',label=lin[1]+' VMM 1')
		# ~ plt.figure(2)
		# ~ plt.hist(y1,stacked = True,bins=256)
		# ~ plt.hist(y2,stacked = True,bins=256)
		# ~ print(ally)
plt.figure(0)
#plt.legend()

plt.figure(1)
plt.hist(ally,bins=1024)		
plt.grid(linestyle = ':')
plt.xlim(0,1024)
plt.savefig('pedestalhisto_all.pdf')
plt.xlim(140,210)
plt.savefig('pedestalhisto_zoom.pdf')
plt.show()
