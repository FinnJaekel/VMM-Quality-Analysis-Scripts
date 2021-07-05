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
	
fname = 'pedestals.txt'
means = np.empty(0)
meanerrs = np.empty(0)
devs = np.empty(0)
deverrs = np.empty(0)
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
		hID = newlin[0]
		print(hID)
		x1=np.fromstring(newlin[6],sep=',')
		y1=np.fromstring(newlin[7],sep=',')
		y2=np.fromstring(newlin[8],sep=',')
		ally = np.append(ally,y1)
		ally = np.append(ally,y2)
		plt.figure(0)
		plt.plot(x1,y1,linewidth=0,marker ='x',label=lin[1]+' VMM 0')
		plt.plot(x1,y2,linewidth=0,marker ='x',label=lin[1]+' VMM 1')
		ymax1 = max(y1)
		ymin1 = min(y1)
		nbins1 = ymax1-ymin1
		ymax2 = max(y2)
		ymin2 = min(y2)
		nbins2 = ymax2-ymin2
		counts1,bins1 = np.histogram(y1, bins=int(nbins1))
		counts2,bins2 = np.histogram(y2, bins=int(nbins2))
		p0i=(10,170,5)
		xfit = np.linspace(140,210,200)
		try:
			popt1,pcov1 = curve_fit(singlegauss,bins1[:-1],counts1,p0=p0i)
			y1fit = singlegauss(xfit,*popt1)
			plt.figure(5)
			plt.hist(y1,bins=int(nbins1))
			plt.plot(xfit,y1fit)
		except RuntimeError:
			print("For Measurement "+ hID + ", VMM 1 No Fit was found (Pedestal Problematic)")
		try:
			popt2,pcov2 = curve_fit(singlegauss,bins2[:-1],counts2,p0=p0i)
			y2fit =singlegauss(xfit,*popt2)
			plt.figure(5)
			plt.hist(y2,bins=int(nbins2))
			plt.plot(xfit,y2fit)
		except RuntimeError:
			print("For Measurement "+ hID + ", VMM 2 No Fit was found (Pedestal Problematic)")
		#plt.show()
		# ~ plt.figure(2)
		# ~ plt.hist(y1,stacked = True,bins=256)
		# ~ plt.hist(y2,stacked = True,bins=256)
		# ~ print(ally)
plt.figure(0)
#plt.legend()
plt.show()

allyfilt = ally[ally<200]
allyfilt = allyfilt[allyfilt>140]

ymax = max(ally)
ymin = min(ally)
print(ymin,ymax)
nbins = ymax-ymin
counts1,bins1 = np.histogram(ally, bins=int(nbins))
binwidth = bins1[1]-bins1[0]
#nbins2 = int((max(allyfilt)-min(allyfilt))/binwidth)
#print(binwidth)
counts,bins = np.histogram(allyfilt,bins=bins1)
bins = bins + binwidth*0.5
p0=(430,170,5)
popt,pcov = curve_fit(singlegauss,bins[:-1],counts,p0)
print(pcov)
print('Range: ' + str(popt[1]-3*popt[2]) + ' to ' + str(popt[1]+3*popt[2]) + ' good')
print('Range: ' + str(popt[1]-5*popt[2]) + ' to ' + str(popt[1]+5*popt[2]) + ' ok')
perr = np.sqrt(np.diag(pcov))
xfit = np.linspace(140,210,200)
yfit = singlegauss(xfit,*popt)
plt.figure(1)
plt.grid(linestyle = ':')
plt.hist(ally,bins=int(nbins),label = 'Data')		

print(popt,perr)
plt.xlim(0,nbins)
plt.xlabel(r'Pedestal$\,$/$\,$ADC Counts')
plt.ylabel('Count N')
plt.yscale('log')
plt.savefig('pedestalhisto_all.pdf')
plt.plot(xfit,yfit,label='Gaussian Fit')
plt.xlim(140,210)
plt.legend()
plt.yscale('linear')
plt.savefig('pedestalhisto_zoom.pdf')
plt.show()
