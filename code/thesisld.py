import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import(MultipleLocator,FormatStrFormatter,AutoMinorLocator)

a=np.loadtxt('/run/media/hedieh/0403-FAFA/1111/2/histo.srt')
b=np.loadtxt('/run/media/hedieh/0403-FAFA/1111/2/add.srt')
x=a[:,2]
y=a[:,1]
x1=b[:,2]
y1=b[:,1]
figure, ax = plt.subplots(figsize = (6.6,8))
ax.plot(x,y,'k.', markersize=2,label='INT sources')
#ax.plot(x1,y1,'lime',marker='.',linestyle='', markersize=3,label='non-variable artificial stars')
plt.ylim(-1,16)


plt.xticks(np.arange(12,24.01,2))
plt.yticks(np.arange(0,15.01,5))
plt.xlim(12,24.1)
#plt.axhline(y=1.8, color='r', linestyle='--', lw=1)
# Customize the tick marks and turn the grid on

plt.minorticks_on()
plt.tick_params(axis='both',which='major', length=7,pad=10,direction='in',labelsize=9)
plt.tick_params(axis='y',which='both',left= True,direction='in')
plt.tick_params(axis='x',which='both',bottom= True,direction='in')
plt.tick_params(axis='x',which='both',top= True,direction='in')
plt.tick_params(axis='y',which='both',right= True,direction='in')


points = np.array([[18,2.6],[19,2.7],[20,2.8],[21,3.0],[10,2.6]])
plt.plot(18,2.6,'r', markersize=4,label='$L$ index threshold',marker='o',linestyle='', zorder=7,markerfacecolor='r')
plt.plot(19,2.7,'r', markersize=4,marker='o', zorder=7,markerfacecolor='r')
plt.plot(20,2.8,'r', markersize=4,marker='o',zorder=7,markerfacecolor='r')
plt.plot(21,3,'r', markersize=4,marker='o', zorder=7,markerfacecolor='r')
plt.plot(12,2.6,'w.', markersize=0.02)
plt.plot(9,2.6,'w.', markersize=0.02)


plt.plot(points[:,0],points[:,1],'m.', markersize=3)

model = np.poly1d(np.polyfit(points[:,0],points[:,1],2))
xp = np.linspace(10,25,100)##	تولید عدد برای رسم منحنی
plt.plot(xp,model(xp), c='r',  linewidth=0.75, linestyle='--')##	رسم منحنی
plt.axvline(x=21.2, color='k', linestyle='--', lw=0.75,dashes=[10,10])
plt.axvline(x=17.08, color='k', linestyle=':', lw=0.75)

plt.xlabel('$i$ (mag)',size='9')
plt.ylabel('variability index $L$',size='9')
print(model)
plt.legend(loc=2)
plt.show()
