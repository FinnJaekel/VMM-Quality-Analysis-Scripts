import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.optimize import curve_fit 
import sys

# ~ col = sys.argv[1]
# ~ bins = sys.argv[2]
# ~ xlabel = sys.argv[3]
# ~ ylabel = sys.argv[4]

def singlegauss(x,a1,mu1,s1):
	val = a1*np.exp(-pow((x-mu1),2)/(2*pow((s1),2)))
	return val	

fname = 'hybrid_overview.txt'
plt.rcParams.update({'font.size': 12})
slopes = np.genfromtxt(fname,usecols=(8))
intercepts = np.genfromtxt(fname,usecols=(9))
rlgs = np.empty(0)
rlos = np.empty(0)
rhgs = np.empty(0)
rhos = np.empty(0)
rwgs = np.empty(0)
rwos = np.empty(0)
nbin = np.empty(0)

slopemin = min(slopes)
slopemax = max(slopes)
sloperange = slopemax-slopemin
intercmin = min(intercepts)
intercmax = max(intercepts)
intercrange = intercmax-intercmin

for i in range(34,35):
	counts1,bins1 = np.histogram(slopes, bins=i)
	popt1,pcov1 = curve_fit(singlegauss,bins1[:-1],counts1)
	rlg = popt1[1]+3*popt1[2]
	rlo = popt1[1]+5*popt1[2]
	rhg = popt1[1]-3*popt1[2]
	rho = popt1[1]-5*popt1[2]
	rwg = abs(rhg-rlg)
	rwo = abs(rho-rlo)
	rlgs = np.append(rlgs,rlg)
	rlos = np.append(rlos,rlo)
	rhgs = np.append(rhgs,rhg)
	rhos = np.append(rhos,rho)
	rwgs = np.append(rwgs,rwg)
	rwos = np.append(rwos,rwo)
	nbin = np.append(nbin,i)
	print('Range: ' + str(popt1[1]+3*popt1[2]) + ' to ' + str(popt1[1]-3*popt1[2]) + ' good')
	print('Range: ' + str(popt1[1]+5*popt1[2]) + ' to ' + str(popt1[1]-5*popt1[2]) + ' ok')

counts3,bins3 = np.histogram(slopes, bins=34)
binwidth1 = bins3[1]-bins3[0]
slopesfilt = slopes[slopes<0.88]
counts1,bins1 = np.histogram(slopesfilt, bins=bins3)
binwidth = bins1[1]-bins1[0]
print(binwidth,binwidth1)
bins1 = bins1 + binwidth*0.5
popt1,pcov1 = curve_fit(singlegauss,bins1[:-1],counts1)
x1fit = np.linspace(0.7,0.9,100)
y1fit = singlegauss(x1fit,*popt1)
print(popt1,np.sqrt(np.diag(pcov1)))
counts2,bins2 = np.histogram(intercepts, bins=154)
p02 = (11,20,20)
popt2,pcov2 = curve_fit(singlegauss,bins2[:-1],counts2)
x2fit = np.linspace(0,160,200)
y2fit = singlegauss(x2fit,*popt2)
print(popt2)
print('Range: ' + str(popt1[1]+3*popt1[2]) + ' to ' + str(popt1[1]-3*popt1[2]) + ' good')
print('Range: ' + str(popt1[1]+5*popt1[2]) + ' to ' + str(popt1[1]-5*popt1[2]) + ' ok')
plt.figure(0)
plt.grid(linestyle=':')
plt.hist(slopes,bins=34,label = 'Slopes')
plt.plot(x1fit,y1fit,label = 'Gaussian Fit')
plt.xlabel(r'Monitoring ADC slope')
plt.ylabel('N')
plt.legend()
#plt.xlim(0,1)
plt.xticks((0.70,0.72,0.74,0.76,0.78,0.8,0.82,0.84,0.86,0.88,0.90))
plt.savefig('monitoringADC_slope.pdf')
plt.figure(1)
plt.grid(linestyle=':')	
plt.hist(intercepts,bins=50,label='Data')
plt.xlabel(r'Monitoring ADC y-intercept/ADC Counts')
plt.ylabel('N')
plt.xlim(0,160)
plt.legend()
plt.savefig('monitoringADC_intercept.pdf')
plt.show()

plt.figure(1)
plt.hist2d(slopes,intercepts)
plt.show()

#plt.figure(1)
#plt.plot(nbin,rlgs)
#plt.plot(nbin,rlos)
#plt.plot(nbin,rhgs)
#plt.plot(nbin,rhos)
#plt.plot(nbin,rwgs)
#plt.plot(nbin,rwos)
#plt.show()
