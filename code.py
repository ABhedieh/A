

from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file_list = ["18-19.txt","19-20.txt","20-21.txt","21-22.txt"]

def gaussian(x, mean, amplitude, standard_deviation):
  return amplitude * np.exp( - ((x -mean ) / standard_deviation) ** 2)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.set_figheight(16)
fig.set_figwidth(20)



# 1 0
# 0 0
i=0
data = []
with open("/home/hedieh/2/" + file_list[i], 'r') as file:
  line = file.readline().split()
  while line:
    data.append([float(line[0]), float(line[1]) ])
    line  = file.readline().split()




data = np.array(data)
counts, bins = np.histogram(data[:,1],bins=35)
ax1.hist(bins[:-1], bins[:-1], weights=counts,color='b',histtype='step')
index_max = np.argmax(counts)

new_bins = np.concatenate((bins[:index_max+2],np.flip(2*bins[index_max+1]-bins[:index_max+1])),axis=0)
new_counts = np.concatenate((counts[:index_max+1],np.flip(counts[:index_max+1])),axis=0)


ax1.hist(new_bins[:-1],new_bins, weights=new_counts,color='w',histtype='bar',alpha=0.5,edgecolor='g')

bin_centers = new_bins[:-1] + np.diff(new_bins) / 2
popt, _ = curve_fit(gaussian, bin_centers, new_counts, p0=[1., 0., 1.])
print(popt)
x_interval_for_fit = np.linspace(new_bins[0]-2, new_bins[-1]+5, 10000)
#ax1.set_xlabel('Variability index')
ax1.set_ylabel('Number of stars',fontsize='16')
#ax1.set_title('i=18-19',fontsize='10')
ax1.text(10, 70.0, 'i=18-19',color='r',fontsize='16')
ax1.text(10, 65.0, 'L=2.6',color='r',fontsize='16')
ax1.text(10, 58.0, 'Totall=77,non=7',color='b',fontsize='16')
ax1.axvline(x=2.6, color='r',linestyle='--',lw='0.5')
ax1.axhline(y=77, color='r',linestyle='--',lw='0.5')
ax1.axhline(y=7, color='r',linestyle='--',lw='0.5')
ax1.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), color='r')
ax1.minorticks_on()
ax1.tick_params(axis='both',which='major', length=7,pad=10,direction='in',labelsize='16')
ax1.tick_params(axis='y',which='both',left= True,direction='in')
ax1.tick_params(axis='x',which='both',bottom= True,direction='in')
ax1.tick_params(axis='x',which='both',top= True,direction='in')
ax1.tick_params(axis='y',which='both',right= True,direction='in')



# 0 1
# 0 0
i=1
data = []
with open("/home/hedieh/2/" + file_list[i], 'r') as file:
  line = file.readline().split()
  while line:
    data.append([float(line[0]), float(line[1]) ])
    line  = file.readline().split()


data = np.array(data)
counts, bins = np.histogram(data[:,1],bins=32)
ax2.hist(bins[:-1], bins[:-1], weights=counts,color='b',histtype='step')
index_max = np.argmax(counts)

new_bins = np.concatenate((bins[:index_max+2],np.flip(2*bins[index_max+1]-bins[:index_max+1])),axis=0)
new_counts = np.concatenate((counts[:index_max+1],np.flip(counts[:index_max+1])),axis=0)


ax2.hist(new_bins[:-1],new_bins, weights=new_counts,color='w',histtype='bar',alpha=0.5,edgecolor='g')
bin_centers = new_bins[:-1] + np.diff(new_bins) / 2
popt, _ = curve_fit(gaussian, bin_centers, new_counts, p0=[1., 0., 1.])
print(popt)
x_interval_for_fit = np.linspace(new_bins[0]-2, new_bins[-1]+5, 10000)
#ax2.set_xlabel('Variability index')
#ax2.set_ylabel('Number of stars')
#ax2.set_title('i-19-20',fontsize='10')
ax2.text(6, 108.0, 'i=19-20',color='r',fontsize='16')
ax2.text(6, 101.0, 'L=2.8',color='r',fontsize='16')
ax2.text(6, 90.0, 'Totall=87,non=8',color='b',fontsize='16')
ax2.axvline(x=2.8, color='r',linestyle='--',lw='0.5')
ax2.axhline(y=87, color='r',linestyle='--',lw='0.5')
ax2.axhline(y=8, color='r',linestyle='--',lw='0.5')
ax2.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), color='r')
ax2.minorticks_on()
ax2.tick_params(axis='both',which='major', length=7,pad=10,direction='in',labelsize='16')
ax2.tick_params(axis='y',which='both',left= True,direction='in')
ax2.tick_params(axis='x',which='both',bottom= True,direction='in')
ax2.tick_params(axis='x',which='both',top= True,direction='in')
ax2.tick_params(axis='y',which='both',right= True,direction='in')


# 0 0
# 1 0
i=2
data = []
with open("/home/hedieh/2/" + file_list[i], 'r') as file:
  line = file.readline().split()
  while line:
    data.append([float(line[0]), float(line[1]) ])
    line  = file.readline().split()

data = np.array(data)
counts, bins = np.histogram(data[:,1],bins=27)
ax3.hist(bins[:-1], bins[:-1], weights=counts,color='b',histtype='step')
index_max = np.argmax(counts)

new_bins = np.concatenate((bins[:index_max+2],np.flip(2*bins[index_max+1]-bins[:index_max+1])),axis=0)
new_counts = np.concatenate((counts[:index_max+1],np.flip(counts[:index_max+1])),axis=0)


ax3.hist(new_bins[:-1],new_bins, weights=new_counts,color='w',histtype='bar',alpha=0.5,edgecolor='g')
bin_centers = new_bins[:-1] + np.diff(new_bins) / 2
popt, _ = curve_fit(gaussian, bin_centers, new_counts, p0=[1., 0., 1.])
print(popt)
x_interval_for_fit = np.linspace(new_bins[0]-2, new_bins[-1]+5, 10000)
ax3.set_xlabel('Variability index',fontsize='16')
ax3.set_ylabel('Number of stars',fontsize='16')
#ax3.set_title('i=20-21',fontsize='10')
ax3.text(5, 163.0, 'i=20-21',color='r',fontsize='16')
ax3.text(5, 145.0, 'L=3.0',color='r',fontsize='16')
ax3.text(5, 123.0, 'Totall=60,non=6',color='b',fontsize='16')
ax3.axvline(x=3, color='r',linestyle='--',lw='0.5')
ax3.axhline(y=6, color='r',linestyle='--',lw='0.5')
ax3.axhline(y=60, color='r',linestyle='--',lw='0.5')

ax3.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), color='r')
ax3.minorticks_on()
ax3.tick_params(axis='both',which='major', length=7,pad=10,direction='in',labelsize='16')
ax3.tick_params(axis='y',which='both',left= True,direction='in')
ax3.tick_params(axis='x',which='both',bottom= True,direction='in')
ax3.tick_params(axis='x',which='both',top= True,direction='in')
ax3.tick_params(axis='y',which='both',right= True,direction='in')




# 0 0
# 0 1
i=3
data = []
with open("/home/hedieh/2/" + file_list[i], 'r') as file:
  line = file.readline().split()
  while line:
    data.append([float(line[0]), float(line[1])])
    line  = file.readline().split()


data = np.array(data)
counts, bins = np.histogram(data[:,1],bins=30)
ax4.hist(bins[:-1], bins[:-1], weights=counts,color='b',histtype='step')
index_max = np.argmax(counts)

new_bins = np.concatenate((bins[:index_max+2],np.flip(2*bins[index_max+1]-bins[:index_max+1])),axis=0)
new_counts = np.concatenate((counts[:index_max+1],np.flip(counts[:index_max+1])),axis=0)


ax4.hist(new_bins[:-1],new_bins, weights=new_counts,color='w',histtype='bar',alpha=0.5,edgecolor='g')
bin_centers = new_bins[:-1] + np.diff(new_bins) / 2
popt, _ = curve_fit(gaussian, bin_centers, new_counts, p0=[1., 0., 1.])
print(popt)
x_interval_for_fit = np.linspace(new_bins[0]-2, new_bins[-1]+5, 10000)
ax4.set_xlabel('Variability index',fontsize='16')
#ax4.set_ylabel('Number of stars')
#ax4.set_title('i=21-22',fontsize='10')
ax4.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), color='r')
ax4.minorticks_on()
ax4.tick_params(axis='both',which='major', length=7,pad=10,direction='in',labelsize='16')
ax4.tick_params(axis='y',which='both',left= True,direction='in')
ax4.tick_params(axis='x',which='both',bottom= True,direction='in')
ax4.tick_params(axis='x',which='both',top= True,direction='in')
ax4.tick_params(axis='y',which='both',right= True,direction='in')
ax4.text(6, 2.0, 'i=21-22',color='r',fontsize='16')
ax4.text(6, 101.0, 'L=3.6',color='r',fontsize='16')
ax4.text(5.8, 85.0, 'Totall=49,non=3',color='b',fontsize='16')
ax4.axvline(x=3.6, color='r',linestyle='--',lw='0.5')
ax4.axhline(y=3.5, color='r',linestyle='--',lw='0.5')
ax4.axhline(y=49, color='r',linestyle='--',lw='0.5')

#   plt.show()

plt.show()