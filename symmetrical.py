# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Implementation of converse KAM result (MacKay'18) using a symmetry
# to an near-integrable magnetic field example: axisymmetrical magnetic
# field perturbed with 2 helical mode (Kallinikos'14).
# 
# # # # # # # # # # # # # # # # # # # # 
# Input:  'ini_sym.txt' - Parameter file
# Output: "sym5_{orbits}_{tf}_{e1}_{m1}_{n1}_{e2}_{m2}_{n2}.txt"
#         or "sym5_{orbits}_{tf}_{e1}_{m1}_{n1}.txt" if e2==0.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import matplotlib.pyplot as plt
import numpy as np

from sympy import symbols, sqrt, sin, cos, derive_by_array, Matrix
from sympy import tensorproduct, tensorcontraction
from sympy.utilities.lambdify import lambdify
from scipy.integrate import solve_ivp

from joblib import Parallel, delayed

import time
import datetime
import simplejson

start = time.time()




## PRELIMINARIES ##
def read_data():
    file1 = open("ini_sym.txt")
    data = file1.readlines()
    r = []
    for line in data:
        r.append(float(line))
    file1.close()
    return r

data= read_data()

π  = np.pi


## Epsilon string for files # 
def epstr2(e):
    if e < 0.1 and e >= 0.01:
        ep = '0'+str(int(e*1000))
    elif e >= 0.001:
        ep = '00'+str(int(e*10000))
    else:
        ep = '000'+str(int(e*100000))
    return ep


# Cartesian to toroidal coordinates (on a poloidal cross section)
def  rf(x,y,z) : return np.sqrt((np.sqrt(x**2+y**2)-R0)**2 + z**2)
def  ψf(x,y,z) : return (np.sqrt(x**2+y**2)-R0)**2 + z**2
def thf(x,y,z) : return np.arctan2(z,np.sqrt(x**2+y**2)-R0)

# operations
def dotproduct(A,B) : return tensorcontraction(tensorproduct(A,B),(0,1)) # dot vectors A, B
def multiply(A,B)   : return tensorcontraction(tensorproduct(A,B),(1,2)) # multiply matrix A with vector B
def contract(A,B)   : return tensorcontraction(tensorproduct(A,B),(2,3)) # contract 3-form A with vector B
# 3D Levi-Civita symbol (for volume form)
def LC(i,j,k) : return(j-i)*(k-i)*(k-j)/2






## SYSTEM ##

r, th, ph = symbols('r, th, ph')
Vr, Vth, Vph = symbols('Vr, Vth, Vph')



m1 = data[3]
n1 = data[4]
m2 = data[6]
n2 = data[7]
w1 = data[8]
w2 = data[9]
R0 = data[10]


e1 = data[2]
e2 = data[5]
ep1 = epstr2(e1) # epsilon 1 string name for file
ep2 = epstr2(e2) # epsilon 2 string name for file


a = (1 - r/R0)**2    # r = ψ
rho = 1 / (1/a -  1) # 
R = R0 * a / (1 - (1-a)**(0.5) * cos(th))

# magnetic field #
def B(r,th,ph) :
    ge = r - 4
    # B field - # Different scalings
    Br  = (  m1*e1*(r**(m1/2))*(r-4)*sin(m1*th-n1*ph)
             + m2*e2*(r**(m2/2))*(r-4)*sin(m2*th-n2*ph)) #/ (r*R)
    Bth = (w1 + 2*w2*r + e1*(r**(m1/2))*(((r-4)*m1/(2*r)) + 1)*cos(m1*th-n1*ph)
             + e2*(r**(m2/2))*(((r-4)*m2/(2*r)) + 1)*cos(m2*th-n2*ph))#/ (r*R)
    Bph = 1 #/(r*R)
    return [Br, Bth, Bph]

# flow #
def f(r,th,ph) : return B(r,th,ph)

# Jacobian #
def Df(r,th,ph) :
    k = len(f(r,th,ph))
    return [derive_by_array(f(r,th,ph)[i],[r,th,ph]) for i in range(0,k)]

# tangent flow #
def Tf(r,th,ph,Vr,Vth,Vph): return multiply(Df(r,th,ph), [Vr,Vth,Vph])

# whole system #
def F(r,th,ph,Vr,Vth,Vph) : return [*f(r,th,ph), *Tf(r,th,ph,Vr,Vth,Vph)]

Fnum = lambdify((r,th,ph,Vr,Vth,Vph), F(r,th,ph,Vr,Vth,Vph), 'numpy')

def system(t, X):
    x1,x2,x3,v1,v2,v3 = X
    dXdt = Fnum(x1,x2,x3,v1,v2,v3)
    return dXdt






## DIRECTION FIELD ξ ##

# Spherical coordinates - - - -- 
#metric = Matrix([[1, 0, 0],      
#                 [0, r**2, 0],
#                 [0, 0, R**2]])


metric = Matrix([[1/(2*r), 0, 0],   # Adapted cartesian 
                 [0, 2*r, 0],       # coordinates
                 [0, 0, R0**2]])


def  J(r,th,ph) : return r
def dJ(r,th,ph) : return derive_by_array(J(r,th,ph),[r,th,ph])

def xi(r,th,ph) :
    gradJ = multiply(metric.inv(), dJ(r,th,ph))
    #gradH = multiply(metric.inv(), dH(x,y,px,py))
    #dH2  = dotproduct(gradH, dH(x,y,px,py))
    #dHdJ = dotproduct(gradJ, dH(x,y,px,py))
    #a = dHdJ/dH2
    return gradJ #- a*gradH

ξ = lambdify((r,th,ph), xi(r,th,ph), 'numpy')

def mod2ξ(r,th,ph) : return np.dot(ξ(r,th,ph), ξ(r,th,ph))






## R-ANTISYMMETRIC η ##

def eta(r,th,ph) :
    ff  = multiply(metric, f(r,th,ph))
    return [0, ff[2], -ff[1]] # [0,1,0]
η = lambdify((r,th,ph), eta(r,th,ph), 'numpy')






## CONVERSE KAM CONDITION ##

# volume form #
def vol(r,th,ph) :
    ρ = R0**2  #  
    v = ρ*np.fromfunction(LC, (3,3,3))
    return v.tolist()

# magnetic flux form #
def beta(r,th,ph) : return contract(vol(r,th,ph), B(r,th,ph))
β = lambdify((r,th,ph), beta(r,th,ph), 'numpy')

def con(t, X) :
    x1,x2,x3,v1,v2,v3 = X
    if mod2ξ(x1,x2,x3) > 1e-12 :
        ω  = np.dot(ξ(x1,x2,x3), np.matmul(β(x1,x2,x3), [v1,v2,v3]))
        return ω
con.terminal = True







## INTEGRATION & RESULTS ##

# time parameters #
tf = int(data[0])
#tf = 50
t_int = [0, tf]
t = np.linspace(0, tf, 2001)



# space sampling #
o_ax = int(data[1])
orbits = o_ax # 
hr = 1.0 # axis halfrange

y0_list = np.linspace(R0-hr, R0+hr, o_ax)

ph0 = 0


# Loop to divede processes - - -
def loop(i) :
    
    s = time.time()
    
    y0 = y0_list[i]
    z0=0

    r0 = ψf(0,y0,z0)/2
    th0 = thf(0,y0,z0)
        
    v0 = η(r0, th0, ph0)
    X0 =  [r0, th0, ph0, *v0]
    
    SOL = solve_ivp(system, t_int, X0, method='RK45', events=con, rtol=1e-8, atol=1e-11)
    
    f = time.time()
    tr = f-s
    
    ie = SOL.status
    if ie==1 : te = SOL.t_events[0][0]
    else : te = tf
    
    #return y0, z0, ie, te, tr
    return y0, z0, ie, te, tr, SOL.y[2][-1] # Extra: phi[-1]


if __name__ == "__main__":
    result = Parallel(n_jobs=-1)(delayed(loop)(i) for i in range(o_ax) )
                           # -1=all cores, 1=serial



finish = time.time()
hours, rem = divmod(finish-start, 3600)
minutes, seconds = divmod(rem, 60)
print('time = {:0>2}:{:0>2}:{:02.0f}'.format(int(hours),int(minutes),seconds))


if e2 == 0 :
    file = open('sym5_%s_%s_%s_%s_%s.txt' % (orbits, tf, int(m1),int(n1),ep1), 'w')
else:
    file = open('sym5_%s_%s_%s_%s_%s_%s_%s_%s.txt' %
                (orbits,tf, int(m1),int(n1),ep1,int(m2),int(n2),ep2), 'w')
simplejson.dump(result, file)
file.close()


