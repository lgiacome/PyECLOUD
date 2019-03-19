# Lorenzo Giacomel - CERN BE-ABP-HSC 2019

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e as qe
from scipy.constants import m_e as me
import argparse

BIN = os.path.expanduser("../../../../")
sys.path.append(BIN)
import PyECLOUD.myloadmat_to_obj as mlo

# Load the results of the PyECLOUD simulation
ob = mlo.myloadmat_to_obj('Pyecltest.mat')

# Chamber radius
R = 2.3e-2


N = 200		# Number of time intervals per bunch passage
indhit = 65		# Index of the time step of emission of the MP (read from MP state)
R0 = 0.02289895		# Initial position of the MP (read from MP state)
V0 = -984277.04156396		# Initial velovity of the MP (read from MP state)

Nt = np.linspace(0,N-1,N)		# Vector of time step indices

# Preallocation of position and velocity vectors for explicit-implicit Euler
# method with with parabolic potential
r2 = np.zeros(N-indhit)
v2 = np.zeros(N-indhit)
r2[0] = R0
v2[0] = V0

# Preallocation of position and velocity vectors for "Runge-Kutta"-style method
# with with parabolic potential
r3 = np.zeros(N-indhit)
v3 = np.zeros(N-indhit)
r3[0] = R0
v3[0] = V0

# Preallocation of position and velocity vectors for explicit Euler method with
# potential read from the simulation
r4 = np.zeros(N-indhit)
v4 = np.zeros(N-indhit)
r4[0] = R0
v4[0] = V0

# Read and store the times from the PyECLOUD simulation
t2 = ob.t[N*16+indhit:N*17]
# Read and store from the PyECLOUD simulation Vmax at every time step
Vmax = -ob.phi_min[N*16+indhit:N*17]
alpha = 2*Vmax/(R*R)*qe/me

dt = t2[1]-t2[0]	# Time step for Euler and Runge-Kutta
dx = 0.0003;	# Mesh width for approximation of second order derivative (same)
				# as dx in PyECLOUD space charge computations

# Explicit-Implicit Euler method with parabolic potential
for i in np.arange(1,len(r3)):
	v3[i] = v3[i-1] + dt*alpha[(i-1)]*r3[i-1]
	r3[i] = r3[i-1] + dt*v3[i]

dt = 2*dt

# Runge-Kutta method with parabolic potential
for i in np.arange(2,len(r2),2):
	k0 = alpha[i-2]*r2[i-2]*dt**2
	k1 = alpha[i-1]*(r2[i-2]+1./2.*dt*v2[i-2]+1./8.*k0)*dt**2
	k2 = alpha[i]*(r2[i-2]+dt*v2[i-2]+1./2.*k1)*dt**2
	r2[i-1] = r2[i-2]+(dt*v2[i-2]+1./6.*(k0+2*k1))
	r2[i] = r2[i-2]+(dt*v2[i-2]+1./6.*(k0+2*k1))
	v2[i] = v2[i-2]+(1./(6.*dt)*(k0+4*k1+k2))

dt = dt/2.

# Explicit Euler method with potential read from the PyECLOUD simulation
norm_fac = 200./48.
pass_ind = 16
filename_phi = 'phi_video/phi_pass%d.mat'%pass_ind
obout_phi = mlo.myloadmat_to_obj(filename_phi)
rx = obout_phi.xg_sc
phi_video = obout_phi.phi_video

def bounds(rx, rr):
	for ir in range(len(rx)):
		lb = 0
		ub = 0
		if(rx[ir]>rr):
			lb = ir-1
			ub = ir
			break
	return lb, ub

for i in np.arange(1,len(r4)):
	ii = np.ceil((i+indhit)/norm_fac)
	if(ii > 47):
		ii = 47
	phi = np.squeeze(phi_video[int(ii-1), :, :])
	V = phi[:,80]

	dxrr = r4[i-1] - dx
	ub, lb = bounds(rx, dxrr)
	m = (V[ub]-V[lb])/(rx[ub]-rx[lb])
	Vdxrr =  m*(dxrr-rx[lb]) + V[lb]

	rrdx = r4[i-1] + dx
	ub, lb = bounds(rx, rrdx)

	m = (V[ub]-V[lb])/(rx[ub]-rx[lb])
	Vrrdx =  m*(rrdx-rx[lb]) + V[lb]

	v4[i] = v4[i-1] + dt*qe/me*(Vrrdx - Vdxrr)/(2*dx)
	r4[i] = r4[i-1] + dt*v4[i]

	if r4[i]>23e-3:
		r4 = np.resize(r4,i+1)
		break

plt.plot(t2,r[indhit:])
plt.plot(t2,r3)
plt.plot(t2,r2)
plt.plot(t2[0:i+1],r4)
plt.show()

'''
thit2 = t[(r2>0.02292322)&(r2<=2.3e-02)]
ts2 = thit2[1]-thit2[0]
'''
