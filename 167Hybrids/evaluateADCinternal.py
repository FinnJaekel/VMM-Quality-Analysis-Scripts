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
	
fname = 'adccalibdata_int_new_abcd.txt'
ally = np.empty(0)
allx = np.empty(0)
alla = np.empty(0)
allb = np.empty(0)
allc = np.empty(0)
allcents = np.empty(0)
allmax = np.empty(0)
alldevs = np.empty(0)

plt.rcParams.update({'font.size': 12})

listofbadhIDs0 = ["a0000065","a000007d","68aba000a0000015"]
listofbadhIDs1 = ["b8aca000a000006c"]

with open(fname) as f_in:
	lines = f_in.readlines()
#	for line in lines:
#		line = line.strip('{')
#		line = line.strip('}')
#		newlines.append(line)
	data = np.genfromtxt(lines,dtype=str)
	for lin in data:
		newlin = []
		for x in lin:
			x = x.replace('{','')
			x = x.replace('}','')
			newlin = np.append(newlin,[x])
		hID = newlin[1]
		print(hID)
		x1=np.fromstring(newlin[6],sep=',')
		y1=np.fromstring(newlin[7],sep=',')
		y2=np.fromstring(newlin[8],sep=',')
		x2=np.fromstring(newlin[6],sep=',')
		x1 = x1[np.where((y1>0)&(y1<550))]
		y1 = y1[np.where((y1>0)&(y1<550))]
		x2 = x2[np.where((y2>0)&(y2<550))]
		y2 = y2[np.where((y2>0)&(y2<550))]
		print(newlin[0],len(x1),len(y1))
		print(newlin[0],len(x2),len(y2))
		p0=[-0.05,32,310]
		if(len(x1)>0 and hID not in listofbadhIDs0):
			popt,pcov = curve_fit(parabola2,x1,y1,p0)
			alla = np.append(alla,popt[0])
			allb = np.append(allb,popt[1])
			allc = np.append(allc,popt[2])
			center = popt[1]/(-2*popt[0])
			allcents = np.append(allcents,center)
			maxv =popt[2]-popt[0]*pow(popt[1],2)/(4*pow(popt[0],2))
			allmax = np.append(allmax,maxv)
			xfit=np.linspace(0,63,200)
			yfit1 = parabola2(xfit,*popt)
			yexp = parabola2(x1,*popt)
			dev = y1-yexp
			alldevs = np.append(alldevs,dev)
			ally = np.append(ally,y1)
			allx = np.append(allx,x1)
			print(center,maxv)
		if(len(x2)>0 and hID not in listofbadhIDs1):
			popt,pcov = curve_fit(parabola2,x2,y2,p0)
			alla = np.append(alla,popt[0])
			allb = np.append(allb,popt[1])
			allc = np.append(allc,popt[2])
			center = popt[1]/(-2*popt[0])
			allcents = np.append(allcents,center)
			maxv =popt[2]-popt[0]*pow(popt[1],2)/(4*pow(popt[0],2))
			allmax = np.append(allmax,maxv)
			yexp = parabola2(x2,*popt)
			dev = y2-yexp
			alldevs = np.append(alldevs,dev)
			ally = np.append(ally,y2)
			allx = np.append(allx,x2)
			print(center,maxv)
		plt.figure(0)
		plt.plot(x1,y1,linewidth=0,marker ='x',label=lin[1]+' VMM 0')
		plt.plot(x2,y2,linewidth=0,marker ='x',label=lin[1]+' VMM 1')
		yfit2 = parabola2(xfit,*popt)
		plt.plot(xfit,yfit1)
		plt.plot(xfit,yfit2)
 		#plt.figure(2)
		#plt.hist(y1,stacked = True,bins=256)
		#plt.hist(y2,stacked = True,bins=256)
		#print(ally)
#plt.figure(0)
#plt.legend()

plt.figure(3)
nbins = 20 #20
bwidth = 0.0015
nbins = int((max(alla)-min(alla))/bwidth)
# ~ hardbins= [-0.08463212, -0.08314301, -0.0816539 , -0.08016479, -0.07867567,
       # ~ -0.07718656, -0.07569745, -0.07420834, -0.07271923, -0.07123011,
       # ~ -0.069741  , -0.06825189, -0.06676278, -0.06527366, -0.06378455,
       # ~ -0.06229544, -0.06080633, -0.05931721, -0.0578281 , -0.05633899,
       # ~ -0.05484988]
plt.hist(alla,bins=nbins,label='Data')
counts,bins = np.histogram(alla,bins=nbins)
#print('binsCurve', bins)
bw = bins[1]-bins[0]
bins = bins+0.5*bw
p0a= (20,-0.07,0.01)
popt,pcov = curve_fit(singlegauss,bins[:-1],counts,p0=p0a)
print('Curvature:')
print('Range: ' + str(popt[1]-3*popt[2]) + ' to ' + str(popt[1]+3*popt[2]) + ' good')
print('Range: ' + str(popt[1]-5*popt[2]) + ' to ' + str(popt[1]+5*popt[2]) + ' ok')
fitx = np.linspace(min(alla),max(alla),nbins*4)
fity = singlegauss(fitx,*popt)
plt.plot(fitx,fity,label='Gaussian fit',color='black')
fita = popt[1]
print('Curvature: ',popt,np.sqrt(np.diag(pcov)))
plt.xlabel('Curvatures')
plt.ylabel('Count N')
plt.legend()
plt.grid(linestyle = ':')
plt.savefig('ADCCalIntCurve.pdf')

plt.figure(4)
nbins = 30 #30
bwidth = 0.4
nbins = int((max(allcents)-min(allcents))/bwidth)
# ~ hardbins = [28.36652919, 28.76392011, 29.16131103, 29.55870195, 29.95609287,
       # ~ 30.3534838 , 30.75087472, 31.14826564, 31.54565656, 31.94304748,
       # ~ 32.34043841, 32.73782933, 33.13522025, 33.53261117, 33.93000209,
       # ~ 34.32739302, 34.72478394, 35.12217486, 35.51956578, 35.9169567 ,
       # ~ 36.31434762, 36.71173855, 37.10912947, 37.50652039, 37.90391131,
       # ~ 38.30130223, 38.69869316, 39.09608408, 39.493475  , 39.89086592,
       # ~ 40.28825684]
plt.hist(allcents,bins=nbins,color = 'red',label='Data')
counts,bins = np.histogram(allcents,bins=nbins)
#print('binsX_apex', bins)
bw = bins[1]-bins[0]
bins = bins+0.5*bw
p0b= (20,32,2)
popt,pcov = curve_fit(singlegauss,bins[:-1],counts,p0=p0b)
print('Centers:')
print('Range: ' + str(popt[1]-3*popt[2]) + ' to ' + str(popt[1]+3*popt[2]) + ' good')
print('Range: ' + str(popt[1]-5*popt[2]) + ' to ' + str(popt[1]+5*popt[2]) + ' ok')
print('Range: ' + str(popt[1]-6*popt[2]) + ' to ' + str(popt[1]+6*popt[2]) + ' 6sigma')
print('Range: ' + str(popt[1]-7*popt[2]) + ' to ' + str(popt[1]+7*popt[2]) + ' 7sigma')

print('Centers: ',popt,np.sqrt(np.diag(pcov)))
fitx = np.linspace(min(allcents),max(allcents),nbins*4)
fity = singlegauss(fitx,*popt)
fitb = popt[1]
plt.plot(fitx,fity,label='Gaussian fit',color='black')
plt.xlabel(r'$x_\mathrm{Apex}\,/\,[\mathrm{Ch}]$')
plt.ylabel('Count N')
plt.legend()
plt.grid(linestyle = ':')
plt.savefig('ADCCalIntCenter.pdf')

plt.figure(5)
nbins = 10 #10
bwidth = 6.05
nbins = int((max(allmax)-min(allmax))/bwidth)
# ~ hardbins = [423.9468931 , 430.00028085, 436.0536686 , 442.10705634,
       # ~ 448.16044409, 454.21383184, 460.26721958, 466.32060733,
       # ~ 472.37399508, 478.42738283, 484.48077057]
plt.hist(allmax,bins=nbins,color= 'green',label='Data')
counts,bins = np.histogram(allmax,bins=nbins)
#print('binsY_apex', bins)
bw = bins[1]-bins[0]
bins = bins+0.5*bw
p0c= (20,350,10)
popt,pcov = curve_fit(singlegauss,bins[:-1],counts,p0=p0c)
print('Heights:')
print('Range: ' + str(popt[1]-3*popt[2]) + ' to ' + str(popt[1]+3*popt[2]) + ' good')
print('Range: ' + str(popt[1]-5*popt[2]) + ' to ' + str(popt[1]+5*popt[2]) + ' ok')
print('Range: ' + str(popt[1]-6*popt[2]) + ' to ' + str(popt[1]+6*popt[2]) + ' 6sigma')
print('Range: ' + str(popt[1]-7*popt[2]) + ' to ' + str(popt[1]+7*popt[2]) + ' 7sigma')

print('Heights: ',popt,np.sqrt(np.diag(pcov)))
fitx = np.linspace(min(allmax),max(allmax),nbins*4)
fity = singlegauss(fitx,*popt)
fitc = popt[1]
plt.plot(fitx,fity,label='Gaussian fit',color='black')
plt.xlabel(r'$y_\mathrm{Apex}\,/\,[\mathrm{ADC counts}]$')
plt.ylabel('Count N')
plt.legend()
plt.grid(linestyle = ':')
plt.savefig('ADCCalIntHeight.pdf')

opta = np.mean(alla)
optcent = np.mean(allcents)
optmax = np.mean(allmax)
print(opta,optcent,optmax)

plt.figure(2)
plt.plot(allx,ally,linewidth=0,marker ='x',label='Data')
popt,pcov = curve_fit(parabola2,x1,y1,p0)
xf = np.linspace(0,63,200)
yf = parabola(xf,opta,optcent,optmax)
yfh = parabola(xf,fita,fitb,fitc)
curvature = popt[0]
center = popt[1]/(-2*popt[0])
maxv =popt[2]-popt[0]*pow(popt[1],2)/(4*pow(popt[0],2))
#plt.plot(xf,yf,label = 'From Mean')
plt.plot(xf,yfh,linewidth = 2,label='Optimum Parabola',color ='black')
plt.legend()
plt.xlabel('Channel Nr')
plt.ylabel('Average ADC')
plt.grid(linestyle = ':')
plt.savefig('ADCCalIntOptParabola.pdf')

expected = parabola(allx,fita,fitb,fitc)
deviations = ally-expected
nbins = int(max(deviations)-min(deviations))

plt.figure(7)
#plt.hist(deviations,bins=nbins)
counts,bins =np.histogram(deviations,nbins)
binw = bins[1]-bins[0]
bins= bins+0.5*binw
popt,pcov = curve_fit(singlegauss,bins[:-1],counts)
xfit = np.linspace(-100,100,200)
yfit = singlegauss(xfit,*popt)
print('Deviations from Standard parabola:')
print('Range: ' + str(popt[1]-3*popt[2]) + ' to ' + str(popt[1]+3*popt[2]) + ' good')
print('Range: ' + str(popt[1]-5*popt[2]) + ' to ' + str(popt[1]+5*popt[2]) + ' ok')
print('Range: ' + str(popt[1]-6*popt[2]) + ' to ' + str(popt[1]+6*popt[2]) + ' 6sigma')
print('Range: ' + str(popt[1]-7*popt[2]) + ' to ' + str(popt[1]+7*popt[2]) + ' 7sigma')
#plt.plot(xfit,yfit)
nbins = int(max(alldevs)-min(alldevs))
plt.hist(alldevs,bins=nbins,label='Deviations')
counts,bins =np.histogram(alldevs,nbins)
binw = bins[1]-bins[0]
bins= bins+0.5*binw
popt,pcov = curve_fit(singlegauss,bins[:-1],counts)
print(popt,np.sqrt(np.diag(pcov)))
xfit = np.linspace(-100,100,200)
yfit = singlegauss(xfit,*popt)
print('Deviation to respective fit parabola:')
print('Range: ' + str(popt[1]-3*popt[2]) + ' to ' + str(popt[1]+3*popt[2]) + ' good')
print('Range: ' + str(popt[1]-5*popt[2]) + ' to ' + str(popt[1]+5*popt[2]) + ' ok')
print('Range: ' + str(popt[1]-6*popt[2]) + ' to ' + str(popt[1]+6*popt[2]) + ' 6sigma')
print('Range: ' + str(popt[1]-7*popt[2]) + ' to ' + str(popt[1]+7*popt[2]) + ' 7sigma')
plt.plot(xfit,yfit,label='Gaussian Fit')
plt.grid(linestyle = ':')
plt.xlabel(r'Channel deviation from parabola$\,/\,$[ADC Counts]')
plt.ylabel('Count N')
plt.legend()
plt.savefig('ADCCalIntDeviations.pdf')
# ~ plt.figure(1)
# ~ plt.hist(ally,bins=1024)		
# ~ plt.show()

plt.show()
