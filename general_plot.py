# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Script to plot from 'general_1res.py'
# [Implementation of converse KAM result (MacKay'18) to an integrable
# magnetic field example: axisymmetrical magnetic field perturbed with
# 1 helical mode (Kallinikos'14).]
# 
# # # # # # # # # # # # # # # # # # # # 
# Inputs:  'ini.txt' - Parameter file
#          'gRN_{orbits}_{tf}_{epx}_{m}_{n}.txt'
#
# # # # # # # # # # # # # # # # # # # #
# Authors: Nikos Kallinikos (U.Warwick, U. W.Macedonia, Int Hellenic U.)
#          David Martinez del Rio (U.Warwick)
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import simplejson
import os
from matplotlib.colors import LinearSegmentedColormap


## SET RUN PARAMETERS ##
def read_data():
    file1 = open("ini.txt")
    data = file1.readlines()
    r = []
    for line in data:
        r.append(float(line))
    file1.close()
    return r

data = read_data()


tf = int(data[0])
Pts= int(data[1])
points = Pts**2 #


## IMPORTS DATA ##
ep=data[2]
m = data[3]
n = data[4]
w1 = data[5]
w2 = data[6]
R0 = data[7]

if ep < 0.1 and ep >= 0.01:
    ep1 = '0'+str(int(ep*1000))
elif ep >= 0.001:
    ep1 = '00'+str(int(ep*10000))
else:
    ep1 = '000'+str(int(ep*100000))
    
## OPENS FILE ##
file = open('gRN_%s_%s_%s_%s_%s.txt' % (points, tf,ep1, int(m),int(n)), 'r')
result = simplejson.load(file)
file.close()




x1p = []
x2p = []

x1u = []
x2u = []

x1 = []
x2 = []
q  = []

for i in range(len(result)) :
    x10 = result[i][0] - R0 
    x20 = result[i][1]
    ie  = result[i][2]
    te  = result[i][3]
    if ie == 1 :
        x1p.append(x10) 
        x2p.append(x20)
    else :
        x1u.append(x10)
        x2u.append(x20)
    x1.append(x10)
    x2.append(x20)
    q.append(te/tf)

# - = - = - = - = - = - = - = - = - = # - = - = - = - = - = - = - = - = - = 
deltaR = 0.005
deltaQ= 0.005
eps = ep
fi=0.0
# - = - = - = - = - = - = - = - = - = 
r = np.arange(0, R0, deltaR)
th = np.arange(-np.pi, np.pi, deltaQ)
ψ, Q = np.meshgrid(r, th)
# Hamiltonian - - - - - - - - - - - - 
H = w1*ψ + w2*ψ**2 + eps*(ψ**(m/2))*(ψ-R0**2)*np.cos(m*Q-n*fi)
Ht=m*H-n*ψ

if m==2 and n==1:    
    if eps ==0.007:            #                               m n eps
        ilvl=[-0.0248,-0.01,0.0,0.05 ,0.1,0.15,0.2,0.3, 0.6] # 2 1 0.007
        ilvl2=[-0.035,-0.03,-0.026]
    elif eps == 0.004: 
        ilvl=[-0.02746,-0.026,-0.01,0.0,0.05 ,0.1,0.15,0.2,0.3, 0.6]
        ilvl2=[-0.033,-0.03,-0.028]                          # 2 1 0.004
    else: 
        ilvl=[-0.029339,-0.022,0.0,0.05,0.1,0.2,0.3, 0.6]    # 2 1 0.002
        ilvl2=[-0.033,-0.031,-0.03]
else:
    ilvl=[-0.122825,-0.12,-0.09,-0.02,0.05,0.1,0.2,0.4]      # 3 2 0.007
    ilvl2=[-0.135,-0.132,-0.127]                             # 3 2

fig= plt.figure(figsize=(8,8))
fig2= plt.figure(figsize=(10,8))
ax= fig.gca()
ax2= fig2.gca()





## PLOTTING ##

# standard Converse KAM plot

ax.plot(x1p, x2p, 'ro', markersize=3)
ax.plot(x1u, x2u, 'bo', markersize=3)
ax.set_xlabel('$\~y$', fontsize = 16)
ax.set_ylabel('$\~z$', fontsize = 16, rotation=0)

ax.set_xlim(- 1.05  , 1.05 )     
ax.set_ylim(-1.05,1.05)

CS = ax.contour(np.sqrt(2*ψ)*np.cos(Q) , np.sqrt(2*ψ)*np.sin(Q), Ht,levels = ilvl,colors=[(0.9,0.9,1)],
                alpha=0.9 , linestyles=['solid'])
CS2 = ax.contour(np.sqrt(2*ψ)*np.cos(Q), np.sqrt(2*ψ)*np.sin(Q), Ht,levels = ilvl2, colors=['yellow'],
                 alpha=0.9, linestyles=['solid'])



# Converse-KAM-speed plot - - - - - - - - - - - - 
## Adapted color-map  - - - - - - - ##
col_list0 = [(0, 0, 1),
   ( 0.2752 ,   0.5037 ,   0.9677),
(    0.2287  ,  0.6360  ,  0.9896),
 (   0.1382  ,  0.7637   , 0.8945),
  (  0.0929 ,   0.8690    ,0.7571),
   ( 0.1732 ,   0.9403,    0.6200),
    (0.3561 ,   0.9856 ,   0.4452),
(    0.5574  ,  0.9988  ,  0.2882),
 (   0.7077,    0.9718   , 0.2099),
  (  0.8380 ,   0.9022    ,0.2088),
   ( 0.9394  ,  0.8040,    0.2275),
(    0.9907   , 0.6892 ,   0.2088),
 (   0.9917,    0.5416  ,  0.1496),
  (  0.9527 ,   0.3849   , 0.0822),
    (1,0,0)]
myorder = [14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
col_list = [col_list0[i] for i in myorder]
cmap_name = 'my_list'
cm1 = LinearSegmentedColormap.from_list(cmap_name, col_list, N=90)
## - - - - - - - - - - - - - - - - - ##

sc = ax2.scatter(x1, x2, c=q, vmin=0, vmax=1, s=10, cmap = cm1) #
fig2.cbar = plt.colorbar(sc)
fig2.cbar.set_label('$q$', rotation=0, fontsize = 16, labelpad=10, y=0.5) # 

ax2.set_xlabel('$\~y$', fontsize = 16)
ax2.set_ylabel('$\~z$', fontsize = 16 , rotation=0)


ax2.set_xlim(- 1.05 , 1.05 )     
ax2.set_ylim(-1.05,1.05)
CS3 = ax2.contour(np.sqrt(2*ψ)*np.cos(Q), np.sqrt(2*ψ)*np.sin(Q), Ht,levels = ilvl,colors=[(0.9,0.9,1)], # 'white'
                       alpha=0.9 , linestyles=['solid'])
CS4 = ax2.contour(np.sqrt(2*ψ)*np.cos(Q), np.sqrt(2*ψ)*np.sin(Q), Ht,levels = ilvl2,colors=['yellow'],
                                 alpha=0.9 , linestyles=['solid'])


plt.show()

