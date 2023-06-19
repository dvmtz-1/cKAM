# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Script to compute the Poincare Section from the discriminated data
# from 'symmetrical.py'.
#
# [Implementation of converse KAM result (MacKay'18) using a symmetry
# to an near-integrable magnetic field example: axisymmetrical magnetic
# field perturbed with 2 helical mode (Kallinikos'14).]
#
# # # # # # # # # # # # # # # # # # # # 
# Inputs:  'ini_sym.txt' - Parameter file
#
#          "sym5_{orbits}_{tf}_{e1}_{m1}_{n1}.txt"   --(if e2==0)
#      or  "sym5_{orbits}_{tf}_{e1}_{m1}_{n1}_{e2}_{m2}_{n2}.txt"
#
# Output: "sym5S_{orbits}_{tf}_{e1}_{m1}_{n1}.txt"   --(if e2==0)
#      or "sym5S_{orbits}_{tf}_{e1}_{m1}_{n1}_{e2}_{m2}_{n2}.txt"
#
# # # # # # # # # # # # # # # # # # # #
# Authors: Nikos Kallinikos (U.Warwick, U. W.Macedonia, Int Hellenic U.)
#          David Martinez del Rio (U.Warwick)
#
# MODULES used  # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np
from scipy.integrate import odeint
import simplejson
import json
import re
#import stringify

import time
import datetime
start = time.time()
# # # # # # # # # # #  # # # # # # # # # #  # # # # # # # # # #

# Global varialbes and constant
#a1 = 5
#a2 = 0.5
#R0 = 4.0


# Reading data from file inidata5.txt- - - - - - - - - 
def read_data():
    file1 = open("ini_sym.txt")
    data = file1.readlines()
    r = []
    for line in data:
        r.append(float(line))
    file1.close()
    return r


## Epsilon string for files # #
def epstr2(e):
    if e < 0.1 and e >= 0.01:
        ep = '0'+str(int(e*1000))
    elif e >= 0.001:
        ep = '00'+str(int(e*10000))
    else:
        ep = '000'+str(int(e*100000))
    return ep


# Mod funtion for angle variables - - - -
def mod2pi(angle):
    Nlap=0
    while angle >= 2*np.pi:
        angle -= 2*np.pi
        Nlap+=1
    while angle < 0:
        angle += 2*np.pi
        Nlap=+1
    return angle, Nlap

# For printing ~ ~ 3
def espacio3(j):
    esp= ' '
    while len(esp) < len(str(j)):
        esp = " " + esp
    return esp

# Looking for Poincare section points on plane φ = 0
def findcrossings(orb):
    prb = list()
    L=2*np.pi
    for ii in range(len(orb)-1):
        if (orb[ii][2] < L) and (orb[ii+1][2] > L):
            prb.append(ii)
            L += 2*np.pi
            #print('   [q]',orb[ii][2], '  [q+1]',orb[ii+1][2])
    #print('   [-1]',orb[-1][2])
    return np.array(prb)

# Refine crossing on the Poincare Section
def refine_crossing(a,k,Dt):
    ttf = Dt
    #k=1
    b = a
    it=0
    #print('= = = = = = = = = = = = = = = = = = = = ')
    #print('[',it,']  ph(t0)-2π',abs(b[2]-2*np.pi*k))
    #print('= = = = = = = = = = = = = = = = = = = = ')
    while abs(b[2]-2*np.pi*k)>1e-5 and it<10:
        #b = odeint(pqdot, a, [0,tf], atol=1e-8, rtol=1e-6)[-1];
        ## Newton step using that b[0]=x(tf) and b[2]=x'(tf)
        #tf -= b[0]/b[2]
        b = odeint(B, a, [0,ttf],
               args=(m1,n1,e1,m2,n2,e2,R0),atol=1e-8, rtol=1e-6)[-1]
        ttf = ttf*0.7
        #tf -= (b[2]*tf)/(b[2]-a[2])
        it += 1
    #print('= = = = = = = = = = = = = = = = = = = = ')
    #print('[',it,']  ph(t0)-2π = ',(b[2]-2*np.pi*k))
    #print('= = = = = = = = = = = = = = = = = = = = ')
    return b


# Coordinates coversion
def cart2tor(a):
    y0=a[0]-R0
    z0=a[1]
    x0=a[2]
    ph0=0
    #r0= np.sqrt(y0**2 + z0**2)
    r0= (y0**2 + z0**2)/2
    if y0 != 0:
        th0 = np.arctan(z0/y0)
        if y0 < 0:
            th0 += np.pi
    elif z0> 0:
        th0=np.pi/2
    else:
        th0=-np.pi/2    
    return [r0, th0,ph0]

def tor2car(a):
    rho= R0 + a[0]*np.cos(a[1]) 
    xx= rho * np.sin(a[2])
    yy= rho * np.cos(a[2]) 
    zz= a[0] * np.sin(a[1])
    #return [yy, zz, xx]
    return [yy, zz]




# magnetic field #
def B(x,t,m1,n1,e1,m2,n2,e2,R0) : # for odeint
#def B(t,x,m1,n1,e1,m2,n2,e2,R0) : # for ivp_solver
    r  = x[0]
    th = x[1]
    ph = x[2]
    a = (1 - r/R0)**2    # r = ψ
    rho = 1 / (1/a -  1) # 
    R = R0 * a / (1 - (1-a)**(0.5) * np.cos(th))
    #ge = r - 4
    Br  = (  m1*e1*(r**(m1/2))*(r-4)*np.sin(m1*th-n1*ph)
             + m2*e2*(r**(m2/2))*(r-4)*np.sin(m2*th-n2*ph)) #/ (r*R)
    Bth = (w1 + 2*w2*r + e1*(r**(m1/2))*(((r-4)*m1/(2*r)) + 1)*np.cos(m1*th-n1*ph)
             + e2*(r**(m2/2))*(((r-4)*m2/(2*r)) + 1)*np.cos(m2*th-n2*ph))#/ (r*R)
    Bph = 1 #/(r*R)
    return [Br, Bth, Bph]

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# = = = Main Code = = = = = = = = = = = = = = = = = =
# = = = = = = = = = = = = = = = = 
#
# - = Parameters reading - = - = - = - = - = - = - = - =  
data= read_data()

#tf = int(data[0])
#points = int(data[1])
hr = 1
m1 = int(data[3])
n1 = int(data[4])

m2 = int(data[6])
n2 = int(data[7])

#R0 = int(data[9])
w1 = data[8]
w2 = data[9]
R0 = data[10]

e1 = data[2]
e2 = data[5]
ep1 = epstr2(e1) # epsilon 1 string name for file
ep2 = epstr2(e2) # epsilon 2 string name for file

# - = Integration Parameters - = - = - = - = - = - = - =
#delta = data[10]  # integration step size
#M   = int(data[11]) #Number of spins through z axis if \dot{\phi} = 1
#opt = int(data[12])   # Integrator: 1=Euler, 2= RK2, else=RK4
#N=int(M*(2*np.pi)//delta) # number of steps for M spins on \phi
#mult=20
#mult=10



## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
# Reading processed initial conditions from 'symmetrical5_1r.py'



#tf = 40         # timeout
tf = int(data[0])
#points = int(data[2])**2
points = int(data[1])
#print(points)

## OPENS FILE ##
if e2 == 0 :
    file = open('sym5_%s_%s_%s_%s_%s.txt' % (points, tf, m1,n1,ep1), 'r')
else     :
    file = open('sym5_%s_%s_%s_%s_%s_%s_%s_%s.txt' % (points, tf, m1,n1,ep1,m2,n2,ep2), 'r')

resul = simplejson.load(file)
file.close()
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 


# = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = #
# ƤƧ - ƤƧ -  Poincare Section ƤƧ - ƤƧ - ƤƧ - ƤƧ -
# = # = # = # = # = # = # = # = # = # = #
Num=3000   # Partition on time
tf1 = tf  # To vary the time out input
t = np.linspace(0, tf1,Num)

NPnt=50  #Number of points in the Poincare Section



for i in range(len(resul)):
#for i in range(50): # TEST # # #
    X=[]
    Xcart=[]
    y0, z0, f0, f1, f2 =  resul[i][0], resul[i][1], resul[i][2], resul[i][3], resul[i][4]
    x0=0.0
    #[r0,th0,ph0] = cart2tor([y0 , z0 , x0])
    X0 = cart2tor([y0 , z0 , x0])
    
    Xc = odeint(B, X0, t, args=(m1,n1,e1,m2,n2,e2,R0),atol=1e-8, rtol=1e-6)

    M1 = mod2pi(Xc[-1][1])[1]
    N1 = mod2pi(Xc[-1][2])[1]
    #print(' ||||  M1 =',M1,' N1 = ',N1)
    
    while N1< NPnt:
        Xc2 = odeint(B, Xc[-1], t,
               args=(m1,n1,e1,m2,n2,e2,R0),atol=1e-8, rtol=1e-6)
        #Xc2 = odeint(Bfield2, Xc[-1], t,
        #            args=(m1,n1,eps1,m2,n2,eps2,R0),atol=1e-8, rtol=1e-6)
        M1 = mod2pi(Xc2[-1][1])[1]
        N1 = mod2pi(Xc2[-1][2])[1]
        #print(' ||||  M1 =',M1,' N1 = ',N1)
        
        Xc= np.vstack([Xc,Xc2])
        
    
    #X.append([r0, th0 , ph0])
    X.append(X0)
    #Xcart.append([y0, z0 , x0])
    Xcart.append([y0, z0])
    X=np.array(X)
    Xcart=np.array(Xcart)
    k=1
    
    for j in findcrossings(Xc):
        #Yc = refine_crossing(Xc[j],k,mult*2*np.pi/N2)
        #Yc = refine_crossing(Xc[j],k,Num/tf)
        Yc = refine_crossing(Xc[j],k,tf1/Num)
        #print('len(Yc) = ',len(Yc))
        
        
        #X= np.vstack([X,Xc[j]])
        #Xcart= np.vstack([Xcart,tor2car(Xc[j])])
        X= np.vstack([X,Yc])  # X is in (psi,varthera, phi) coords
        #print('Yc = ',Yc)
        Yc[0]= np.sqrt(2*Yc[0]) # Yc[0] is now: psi --> r = sqrt(2 psi)
        #print('Yc = ',Yc)
        Xcart= np.vstack([Xcart,tor2car(Yc)])
        k += 1
    #print('k=',k)
    
    esp=espacio3(i+1)
    print("   | Poincare | [",i+1, "] | Laps: θM = ",M1, "  φN =",N1)

    print('   |  Section |  '+esp+'    | Points in PS: ',len(X),'   |  Color:',f2)
    # # X = X[0], mod2pi(X[1])
    
    Cv0 = f0*np.ones((len(X),1),dtype=int)
    Cv1 = f1*np.ones((len(X),1),dtype=int)
    Cv2 = f2*np.ones((len(X),1),dtype=int)
    Xcart= np.append(Xcart, Cv0,axis=1)
    Xcart= np.append(Xcart, Cv1,axis=1)
    Xcart= np.append(Xcart, Cv2,axis=1) 
    
    if i==0:
        X3 =Xcart
    else:
        X3 = np.concatenate((X3,Xcart),axis=0)         
    

#np.savetxt('sal016_ext.txt', X3)

X4 = X3.tolist()
#X4 = simplejson.dumps(lists)
##X4 = re.sub('\"', '', X4_uc)
##fp.write(json.dumps(','.join(X4)).replace('"', ''))
##X4 = X4.replace("'", '"')

#n2  = 2
#file = open('%s_%s_%s_2.txt' % (method, points, tf), 'w')
if e2 == 0 :
    file = open('sym5S_%s_%s_%s_%s_%s.txt' % (points, tf, m1,n1,ep1), 'w')
else     :
    file = open('sym5S_%s_%s_%s_%s_%s_%s_%s_%s.txt' % (points, tf, m1,n1,ep1,m2,n2,ep2), 'x')


simplejson.dump(X4, file)
#print(X4)
file.close()

finish = time.time()
hours, rem = divmod(finish-start, 3600)
minutes, seconds = divmod(rem, 60)
print('time = {:0>2}:{:0>2}:{:02.0f}'.format(int(hours),int(minutes),seconds))
# = # = # = # = # = # = # = # = # = # = # 
# ƤƧ - ƤƧ -  Poincare Section ƤƧ - ƤƧ - ƤƧ - ƤƧ -
# = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = #
