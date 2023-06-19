# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Plot from 'general_2r.py'
# [Implementation of converse KAM result (MacKay'18) to an integrable
# magnetic field example: axisymmetrical magnetic field perturbed with
# 2 helical mode (Kallinikos'14).]
# 
# # # # # # # # # # # # # # # # # # # # 
# Inputs:  'ini_2r.txt' - Parameter file
#          "gRN2r_{orbits}_{tf}_{e1}_{m1}_{n1}_{e2}_{m2}_{n2}.txt"
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import simplejson
import os
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams['text.usetex'] = True

## SET RUN PARAMETERS ##
def read_data():
    file1 = open("ini_RN5_D.txt")
    data = file1.readlines()
    r = []
    for line in data:
        r.append(float(line))
    file1.close()
    return r

data = read_data()

# String edition for files - - - 
def ep2str(ep):
    if ep < 0.1 and ep >= 0.01:
        eps = '0'+str(int(ep*1000))
    elif ep >= 0.001:
        eps = '00'+str(int(ep*10000))
    else:
        eps = '000'+str(int(ep*100000))
    return eps


tf = int(data[0])
Pts= int(data[1])
points = Pts**2 # orbits 


## IMPORTS DATA ##
e1=data[2]
e2=data[5]
m1 = data[3]
n1 = data[4]
m2 = data[6]
n2 = data[7]
w1 = data[8]
w2 = data[9]
R0 = data[10]

ep1 = ep2str(e1)
ep2 = ep2str(e2)
    
## OPENS FILE ##


file = open('gRN2r_%s_%s_%s_%s_%s_%s_%s_%s.txt' % ( points, tf,ep1, int(m1),int(n1),ep2, int(m2),int(n2)), 'r')
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


fig= plt.figure(figsize=(8,8))
fig2= plt.figure(figsize=(10,8))
ax= fig.gca()
ax2= fig2.gca()





## PLOTTING ##

# standard Converse KAM plot
ax.plot(x1p, x2p, 'ro', markersize=3)
ax.plot(x1u, x2u, 'bo', markersize=3)
ax.set_xlabel('$\\tilde{y}$', fontsize = 18)
ax.set_ylabel('$\\tilde{z}$', fontsize = 18, rotation=0)

ax.set_xlim(- 1.05 , 1.05 )     
ax.set_ylim(-1.05,1.05)


# Converse-KAM-speed plot = = = = = = = = = = = = = = 
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
ax2.set_xlabel('$\\tilde{y}$', fontsize = 18)
ax2.set_ylabel('$\\tilde{z}$', fontsize = 18, rotation=0)


ax2.set_xlim(- 1.05 , 1.05 )     
ax2.set_ylim(-1.05,1.05)


#plt.savefig('CM_160_200_001_2_1_001_3_2_V_v2.png', dpi=300)
plt.show()

