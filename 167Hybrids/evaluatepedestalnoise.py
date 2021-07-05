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
	
def singlegauss(x,a1,mu1,s1):
	val = a1*np.exp(-pow((x-mu1),2)/(2*pow((s1),2)))
	return val	
	
def getlineup(leng,arr,up,factor):
	line = np.empty(leng)
	if(up==0):
		line.fill(np.average(arr)+factor*np.std(arr))
	else:
		line.fill(np.average(arr)-factor*np.std(arr))
	return line
	
fname = 'pedestalnoises.txt'
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
		x1=np.fromstring(newlin[7],sep=',')
		y1=np.fromstring(newlin[8],sep=',')
		y2=np.fromstring(newlin[9],sep=',')
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

ymax = max(ally)
ymin = min(ally)
print(ymin,ymax)
nbins = (ymax-ymin)*7
counts,bins = np.histogram(ally, bins=int(nbins))
# ~ p0=(430,170,5)
# ~ popt,pcov = curve_fit(singlegauss,bins[:-1],counts,p0)
# ~ print(pcov)
# ~ print('Range: ' + str(popt[1]+3*popt[2]) + ' to ' + str(popt[1]-3*popt[2]) + ' good')
# ~ print('Range: ' + str(popt[1]+5*popt[2]) + ' to ' + str(popt[1]-5*popt[2]) + ' ok')
# ~ perr = np.sqrt(np.diag(pcov))
# ~ xfit = np.linspace(140,210,200)
# ~ yfit = singlegauss(xfit,*popt)
# ~ print(popt,perr)
plt.figure(1)
plt.grid(linestyle = ':')
plt.hist(ally,bins=int(nbins),label = 'Data')		
plt.xlim(0,4)
plt.xlabel(r'Pedestal RMS Noise$\,$/$\,$ADC Counts')
plt.ylabel('Count N')
#plt.savefig('pedestalhisto_all.pdf')
#plt.plot(xfit,yfit,label='Gaussian Fit')
#plt.xlim(140,210)
plt.legend()
plt.savefig('pedestalnoisehisto.pdf')
plt.show()
