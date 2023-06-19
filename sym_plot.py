# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Script to plot the discriminated data from 'symmetrical.py'.
#
# [Implementation of converse KAM result (MacKay'18) using a symmetry
# to an near-integrable magnetic field example: axisymmetrical magnetic
# field perturbed with 2 helical mode (Kallinikos'14).]
#
# # # # # # # # # # # # # # # # # # # # 
# Inputs:  'ini_sym.txt' - Parameter file
#
#     Symm: "sym5_{orbits}_{tf}_{e1}_{m1}_{n1}_{e2}_{m2}_{n2}.txt"
#            (or "sym5_{orbits}_{tf}_{e1}_{m1}_{n1}.txt" if e2==0)
#
#     Psec: "sym5S_{orbits}_{tf}_{e1}_{m1}_{n1}.txt"   - (if e2==0)
#         "sym5S_{orbits}_{tf}_{e1}_{m1}_{n1}_{e2}_{m2}_{n2}.txt"
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import simplejson
import os
from matplotlib.colors import LinearSegmentedColormap

## SET RUN PARAMETERS ##
def read_data():
    file1 = open("ini_sym.txt")
    data = file1.readlines()
    r = []
    for line in data:
        r.append(float(line))
    file1.close()
    return r

data= read_data()


## Epsilon string for files # # 
def epstr2(e):
    if e < 0.1 and e >= 0.01:
        ep = '0'+str(int(e*1000))
    elif e >= 0.001:
        ep = '00'+str(int(e*10000))
    else:
        ep = '000'+str(int(e*100000))
    return ep

# # PARAMETERS # # # # # # # # # # 
tf = int(data[0])
points = int(data[1])
hr = 1
m1 = int(data[3])
n1 = int(data[4])

m2 = int(data[6])
n2 = int(data[7])

w1 = data[8]
w2 = data[9]
R0 = data[10]

eps1 = data[2]
eps2 = data[5]
ep1 = epstr2(eps1) # epsilon 1 string name for file
ep2 = epstr2(eps2) # epsilon 2 string name for file
# # # # # # # # # # # # # # # # # #


## OPENS FILE - - - - - - - - - - - - - - - - - - - ##
### CHOOSE FILE TO PLOT ###
#method = 'sym5' # Plot exist file from 'sym.py'
method = 'sym5S' # Plot Poincare Section from 'sym_Psec.py'

if eps2 == 0 :
    file = open('%s_%s_%s_%s_%s_%s.txt' % (method,points, tf, m1,n1,ep1), 'r')
else     :
    file = open('%s_%s_%s_%s_%s_%s_%s_%s_%s.txt' % (method,points, tf, m1,n1,ep1,m2,n2,ep2), 'r')
## - - - - - - - - - - - - - - - - - - - - - - - - ##



result = simplejson.load(file)
file.close()


## IMPORTS DATA ##

x1p = []
x2p = []

x1u = []
x2u = []

x1 = []
x2 = []
q  = []
q2  = []

for i in range(len(result)) :
    x10 = result[i][0] - R0 
    x20 = result[i][1]
    ie  = result[i][2]
    te  = result[i][3]
    if ie == 1 :
        x1p.append(x10) 
        x2p.append(x20) # Discrminated point
        q2.append(te/tf)
    else :
        x1u.append(x10) # Undetermined point
        x2u.append(x20)       
    x1.append(x10)
    x2.append(x20)
    q.append(te/tf)


# # # # # # # # Level set of K for 1-resonance cases # # # # # # # #
deltaR = 0.005
deltaQ= 0.005
fi=0.0

# - = - = - = - = - = - = - = - = - = 
r = np.arange(0, R0, deltaR)
th = np.arange(-np.pi, np.pi, deltaQ)
ψ, Q = np.meshgrid(r, th)
# Hamiltonian - - - - - - - - - - - - 
H = w1*ψ + w2*ψ**2 + eps1*(ψ**(m1/2))*(ψ-R0**2)*np.cos(m1*Q-n1*fi)
Ht=m1*H-n1*ψ

if m1==2 and n1==1:    
    if eps1 ==0.007:            #                               m n eps
        ilvl=[-0.0248,-0.01,0.0,0.05 ,0.1,0.15,0.2,0.3, 0.6] # 2 1 0.007
        ilvl2=[-0.035,-0.03,-0.026]
    elif eps1 == 0.004: 
        ilvl=[-0.027476,-0.026,-0.019,-0.01,-0.002,0.0,0.05 ,0.1,0.15,0.2,0.3, 0.6]
        ilvl2=[-0.033,-0.03,-0.028]                          # 2 1 0.004
    else: 
        ilvl=[-0.029339,-0.022,0.0,0.05,0.1,0.2,0.3, 0.6]    # 2 1 0.002
        ilvl2=[-0.033,-0.031,-0.03]
else:
    ilvl=[-0.122825,-0.12,-0.09,-0.02,0.05,0.1,0.2,0.4]      # 3 2 0.007
    ilvl2=[-0.135,-0.132,-0.127]                             # 3 2
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# # FIGURES # # # #
fig= plt.figure(figsize=(8,8))
fig2= plt.figure(figsize=(10,8))

ax= fig.gca()
ax2= fig2.gca()



## PLOTTING ##

# standard Converse KAM plot
ax.plot(x1p , x2p, 'ro', markersize=3)
ax.plot(x1u , x2u, 'bo', markersize=3) # Comment to not display undetermined points

ax.set_xlabel('$\~{y}$',fontsize = 16)
ax.set_ylabel('$\~{z}$',fontsize = 16) #, rotation=0)

ax.set_xlim( - hr - .05 ,  hr+ .05 )  
ax.set_ylim(- hr - .05 , hr+ .05)

# - = - = - = - = - = - = - = - = - = # - = - = - = - = - = - = - = - = - = 

if eps2 == 0 :
    CS = ax.contour(np.sqrt(2*ψ)*np.cos(Q), np.sqrt(2*ψ)*np.sin(Q), Ht,levels = ilvl,colors=[(0.9,0.9,1)],
                alpha=0.9 , linestyles=['solid'])
    CS2 = ax.contour(np.sqrt(2*ψ)*np.cos(Q), np.sqrt(2*ψ)*np.sin(Q), Ht,levels = ilvl2, colors=['yellow'],
                alpha=0.9, linestyles=['solid'])


# Converse-KAM-speed plot
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
sc = ax2.scatter(x1 , x2, c=q, vmin=0, vmax=1, s=10, cmap = cm1) 


fig2.cbar = plt.colorbar(sc)
fig2.cbar.set_label('$q$', rotation=0, fontsize = 16, labelpad=10, y=0.5) # labelpad=30

ax2.set_xlabel('$\~{y}$',fontsize = 16)
ax2.set_ylabel('$\~{z}$',fontsize = 16, rotation=0) #, rotation=0)

ax2.set_xlim(- hr - .05 ,  hr+ .05 )   
ax2.set_ylim(- hr - .05 , hr+ .05)

if eps2 == 0 :
    CS3 = ax2.contour(np.sqrt(2*ψ)*np.cos(Q) , np.sqrt(2*ψ)*np.sin(Q),
                  Ht,levels = ilvl,colors=[(0.0,0.0,0.6)],
                        alpha=0.9 , linestyles=['solid'],linewidths=[2])
    CS4 = ax2.contour(np.sqrt(2*ψ)*np.cos(Q) , np.sqrt(2*ψ)*np.sin(Q),
                  Ht,levels = ilvl2,colors=[(1.0,1.0,0.1)], #'yellow'
                        alpha=0.9 , linestyles=['solid'],linewidths=2)#

plt.show()




