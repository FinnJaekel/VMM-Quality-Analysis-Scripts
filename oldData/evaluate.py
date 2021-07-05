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
fname = 'adccalibdata_latest.txt'
with open(fname) as f_in:
  lines = f_in.readlines()
  length = len(lines)
  entries = int(length/66)
  print(length)
  ally = np.empty(0)
  allx = np.empty(0)
  alla = np.empty(0)
  allb = np.empty(0)
  allc = np.empty(0)
  allcents = np.empty(0)
  allmax = np.empty(0)
  entrs = np.arange(0,2*entries)
  for i in range(0,entries):
	  start = i*66+1
	  end = i*66+65
	  print(i,start,end)
	  inp = str(lines[start:end])
	  footskip = length-end-(entries-i)
	  headskip = start
	  print(headskip,footskip)
	  x1 = np.genfromtxt(fname,usecols=(0),skip_header = headskip,skip_footer = footskip)
	  x2 = np.genfromtxt(fname,usecols=(0),skip_header = headskip,skip_footer = footskip)
	  y1 = np.genfromtxt(fname,usecols=(1),skip_header = headskip,skip_footer = footskip)
	  y2 = np.genfromtxt(fname,usecols=(2),skip_header = headskip,skip_footer = footskip)
	  deleted1 =0
	  deleted2 =0
	  for i in range(0,63):
		  if y1[i-deleted1] == 0 or y1[i-deleted1]>800:
			  x1 = np.delete(x1,i-deleted1)
			  y1 = np.delete(y1,i-deleted1)
			  deleted1+=1
		  if y2[i-deleted2]==0 or y2[i-deleted2]>800:
			  x2 = np.delete(x2,i-deleted2)
			  y2 = np.delete(y2,i-deleted2)
			  deleted2+=1
	  vals = np.arange(0,64,0.25)
	  parab = parabola(-0.05,32,310,vals)
	  p0=[-0.05,32,310]
	  popt,pcov = curve_fit(parabola2,x1,y1,p0)
	  alla = np.append(alla,popt[0])
	  allb = np.append(allb,popt[1])
	  allc = np.append(allc,popt[2])
	  center = popt[1]/(-2*popt[0])
	  allcents = np.append(allcents,center)
	  maxv =popt[2]-popt[0]*pow(popt[1],2)/(4*pow(popt[0],2))
	  allmax = np.append(allmax,maxv)
	  print(popt[0],popt[1],popt[2])
	  ally = np.append(ally,y1)
	  ally = np.append(ally,y2)
	  allx = np.append(allx,x1)
	  allx = np.append(allx,x2)
	  popt,pcov = curve_fit(parabola2,x2,y2,p0)
	  fittedparabel = parabola(vals,*popt)
	  #plt.plot(vals,fittedparabel)
	  alla = np.append(alla,popt[0])
	  allb = np.append(allb,popt[1])
	  allc = np.append(allc,popt[2])
	  center = popt[1]/(-2*popt[0])
	  allcents = np.append(allcents,center)
	  maxv =popt[2]-popt[0]*pow(popt[1],2)/(4*pow(popt[0],2))
	  allmax = np.append(allmax,maxv)
	  print(popt[0],popt[1],popt[2])
	  ally = np.append(ally,y1)
	  ally = np.append(ally,y2)
	  allx = np.append(allx,x1)
	  allx = np.append(allx,x2)
	  fittedparabel = parabola2(vals,*popt)
	  #plt.plot(vals,fittedparabel)
	  plt.plot(x1,y1,linewidth=0,marker ='x')
	  plt.plot(x2,y2,linewidth=0,marker ='x')
	  

vals = np.arange(0,63,0.25)
parab = parabola2(vals,-0.05,32,310)
#p0=[-0.05,32,310]
popt,pcov = curve_fit(parabola2,allx,ally)
print(popt[0],popt[1],popt[2])
#plt.plot(allx,ally,linewidth =0,marker = 'x')
fitp = parabola2(vals,*popt)
fitpup = parabola2(vals, popt[0]+np.std(alla),popt[1]-np.std(allb),popt[2]+3*np.std(allc))
fitpdown = parabola2(vals, popt[0]-np.std(alla),popt[1]+np.std(allb),popt[2]-3*np.std(allc))
qtfit = parabola2(vals,-0.0977594,7.33164,239.616)
print("expected diff")
print(qtfit[4*31]-qtfit[4*63-1])
averageparabola = parabola2(vals,np.average(alla),np.average(allb),np.average(allc))
plt.plot(vals,fitp,linewidth=5)
plt.xlabel("Channel")
plt.ylabel("ADC Value")
plt.savefig("idealparabola.pdf")
#plt.plot(vals,fitpup,linewidth=5)
#plt.plot(vals,fitpdown,linewidth=5)
#plt.plot(vals,qtfit,linewidth=10)
#plt.plot(vals,averageparabola,linewidth=3)
#plt.ylim(200,360)
plt.show()
print(np.average(alla),np.average(allb),np.average(allc),np.average(allcents),np.average(allmax))
print(np.std(alla),np.std(allb),np.std(allc),np.std(allcents),np.std(allmax))
plt.plot(entrs,alla)
line1 = getlineup(len(entrs),alla,0,2.5)
line2 = getlineup(len(entrs),alla,1,1.5)
plt.plot(entrs,line1)
plt.plot(entrs,line2)
#plt.plot(entrs,allb)
#plt.plot(entrs,allc)
#plt.plot(entrs,allcents)
#plt.plot(entrs,allmax)
plt.show()
plt.plot(entrs,allcents)
line1 = getlineup(len(entrs),allcents,0,2)
line2 = getlineup(len(entrs),allcents,1,2)
plt.plot(entrs,line1)
plt.plot(entrs,line2)
plt.show()
plt.plot(entrs,allmax)
line1 = getlineup(len(entrs),allmax,0,3)
line2 = getlineup(len(entrs),allmax,1,2)
plt.plot(entrs,line1)
plt.plot(entrs,line2)
plt.show()
