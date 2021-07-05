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
	
def laplacian(x,a,mu,s):
	xprime = abs(x-mu)
	val = a*np.exp(-xprime/s)
	return val
	
def getlineup(leng,arr,up,factor):
	line = np.empty(leng)
	if(up==0):
		line.fill(np.average(arr)+factor*np.std(arr))
	else:
		line.fill(np.average(arr)-factor*np.std(arr))
	return line
	
fname = 'internalpulses_good.txt'
means = np.empty(0)
devs = np.empty(0)
allydevs = np.empty(0)
hnumber = 0
allyfilts = np.empty(0)
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
		x1=np.fromstring(newlin[5],sep=',')
		y1=np.fromstring(newlin[6],sep=',')
		y2=np.fromstring(newlin[7],sep=',')
		y1filt = y1[np.where(y1>80000)]
		y2filt = y2[np.where(y2>80000)]
		x1filt = x1[np.where(y1>80000)]
		x2filt = x1[np.where(y2>80000)]
		ally = np.append(ally,y1)
		ally = np.append(ally,y2)
		allyfilts = np.append(allyfilts,y1filt)
		allyfilts = np.append(allyfilts,y2filt)
		m1 = np.median(y1filt)
		d1 = np.std(y1filt)
		m2 = np.median(y2filt)
		d2 = np.std(y2filt)
		means = np.append(means,m1)
		ydev1 = y1filt-m1
		ydev2 = y2filt-m2
		allydevs = np.append(allydevs,ydev1)
		allydevs = np.append(allydevs,ydev2)
		devs = np.append(devs,d1)
		means = np.append(means,m2)
		devs = np.append(devs,d2)
		plt.figure(2)
		# ~ plt.plot(x1filt,y1filt,linewidth=0,marker ='x',label=lin[1]+' VMM 0')
		# ~ plt.plot(x2filt,y2filt,linewidth=0,marker ='x',label=lin[1]+' VMM 1')
		plt.plot(x1,y1,linewidth=0,marker ='x',label=lin[1]+' VMM 0')
		plt.plot(x1,y2,linewidth=0,marker ='x',label=lin[1]+' VMM 1')
		print(m1,d1)
		print(m2,d2)
		# ~ plt.show()
		# ~ plt.figure(0)
		# ~ plt.plot(x1,y1,linewidth=0,marker ='x',label=lin[1]+' VMM 0')
		# ~ plt.plot(x1,y2,linewidth=0,marker ='x',label=lin[1]+' VMM 1')
		# ~ ymax1 = max(y1)
		# ~ ymin1 = min(y1)
		# ~ nbins1 = ymax1-ymin1
		# ~ ymax2 = max(y2)
		# ~ ymin2 = min(y2)
		# ~ nbins2 = ymax2-ymin2
		# ~ if(nbins1>500):
			# ~ nbins1=500
		# ~ if(nbins1==0):
			# ~ nbins1=500
		# ~ if(nbins2>500):
			# ~ nbins2=500
		# ~ if(nbins2==0):
			# ~ nbins2=500
		# ~ counts1,bins1 = np.histogram(y1, bins=int(nbins1))
		# ~ counts2,bins2 = np.histogram(y2, bins=int(nbins2))
		# ~ p0i=(10,170,5)
		# ~ popt1,pcov1 = curve_fit(singlegauss,bins1[:-1],counts1,p0=p0i)
		# ~ popt2,pcov2 = curve_fit(singlegauss,bins2[:-1],counts2,p0=p0i)
		# ~ xfit = np.linspace(140,210,200)
		# ~ y1fit = singlegauss(xfit,*popt1)
		# ~ y2fit =singlegauss(xfit,*popt2)
		# ~ plt.hist(y1,bins=int(nbins1))
		# ~ plt.hist(y2,bins=int(nbins2))
		# ~ plt.plot(xfit,y1fit)
		# ~ plt.plot(xfit,y2fit)
		# ~ plt.show()
		# ~ plt.figure(2)
		# ~ plt.hist(y1,stacked = True,bins=256)
		# ~ plt.hist(y2,stacked = True,bins=256)
		# ~ print(ally)
		hnumber +=2
## Median Distribution
plt.figure(0)
plt.grid(linestyle=':')
plt.hist(means,bins=20)
plt.xlabel('Median number of received pulses')
plt.ylabel('Count N')
plt.savefig('internalmedians.pdf')
print(means,devs)

#CHANNEL DEVIATIONS
plt.figure(3)
plt.hist(devs,bins=10)
#plt.legend()
devsfilt =allydevs[np.where(abs(allydevs)<1000)]
ndevbins = int(max(devsfilt)-min(devsfilt))
plt.figure(4)
plt.grid(linestyle=':')
plt.hist((devsfilt),bins=ndevbins)
plt.xlabel('Channel deviation from median pulse number')
plt.ylabel('Count N')
plt.savefig('internalchanneldevhisto.pdf')
counts,bins = np.histogram(devsfilt,bins=ndevbins)
binwidth = bins[1]-bins[0]
print(binwidth)
bins = bins + binwidth*0.5
p0l=(500,0,10)
popt,pcov = curve_fit(laplacian,bins[:-1],counts,p0=p0l)
xfit = np.linspace(-80,80,500)
yfit = laplacian(xfit,*popt)
#plt.plot(xfit,yfit)
print(popt)

#ALL RECEIVED PULSES
plt.figure(2)
plt.grid(linestyle=':')
plt.xlabel('Channel Nr.')
plt.ylabel('Received Pulses')
plt.savefig('internalpulsesall.pdf')

#ALL PULSES HISTO
ymax = max(ally)
ymin = min(ally)
print(ymin,ymax)
nbins = (ymax-ymin)
counts,bins = np.histogram(ally, bins=int(nbins))
binwidth = bins[1]-bins[0]
print(binwidth)
bins = bins + binwidth*0.5
p0=(430,9100,50)
#popt,pcov = curve_fit(singlegauss,bins[:-1],counts,p0)
#print(pcov)
#print('Range: ' + str(popt[1]-3*popt[2]) + ' to ' + str(popt[1]+3*popt[2]) + ' good')
#print('Range: ' + str(popt[1]-5*popt[2]) + ' to ' + str(popt[1]+5*popt[2]) + ' ok')
#perr = np.sqrt(np.diag(pcov))
xfit = np.linspace(9000,9400,800)
#yfit = singlegauss(xfit,*popt)
plt.figure(1)
plt.grid(linestyle = ':')
#plt.hist(devsfilt,bins=100,label = 'Data')		
plt.hist(ally,bins=100,label='Data')
#print(popt,perr)
#plt.xlim(0,ymax)
#plt.ylim(0,5)
plt.xlabel(r'Received Pulses')
plt.ylabel('Count N')
#plt.savefig('pedestalhisto_all.pdf')
#plt.plot(xfit,yfit,label='Gaussian Fit')
plt.xlim(0,125000)
plt.legend()
plt.savefig('internalhisto.pdf')
plt.show()
