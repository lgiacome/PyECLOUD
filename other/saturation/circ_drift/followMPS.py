# Lorenzo Giacomel - CERN BE-ABP-HSC 2019

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import PyECLOUD.myloadmat_to_obj as mlo

BIN = os.path.expanduser("../../../../")
sys.path.append(BIN)

N = 200		#Number of time intervals per bunch passage
Nt = np.linspace(0,N-1,N)	# Vector of time step indices
parts = [54564]	#IDs of the particles to be analyzed (list!)
parts.sort()
Np = len(parts) # Total number of particles

#Preallocations
r = np.zeros((N,Np))
v = np.zeros((N,Np))
q = np.zeros((N,Np))

for ind, i in enumerate(Nt):
	# Read and store the MP state at this time step
	filename = 'MP_state_%d.mat'%i
	MPob = mlo.myloadmat_to_obj(filename)
	# For each of the selected particles store the desired information
	for j in range(Np):
		r[ind,j] = np.sqrt(MPob.x_mp[int(parts[j])]**2 + MPob.y_mp[int(parts[j])]**2)
		v[ind,j] = np.sqrt(MPob.vx_mp[int(parts[j])]**2 + MPob.vy_mp[int(parts[j])]**2)
		q[ind,j] = MPob.nel_mp[int(parts[j])]

# Read and store the times from the PyECLOUD simulation
ob = mlo.myloadmat_to_obj('Pyecltest.mat')
t = ob.t[N*16:N*17]

# Plot trajectories, velocities and charges of the particles and the potential
# at the centre of the chamber
ax1 = plt.subplot(4,1,1)
ax2 = ax1.twiny()

for j in range(Np):
	ax1.plot(r[:,j],label=repr(parts[j]))
	ax2.plot(t,r[:,j],alpha=0)

ax1.plot(23e-3*np.ones(N))

ax1.legend()
ax1.set_ylabel('r')
ax1.set_xlabel('Time Step')
ax2.set_xlabel('t')

ax1 = plt.subplot(4,1,2)
ax2 = ax1.twiny()

for j in range(Np):
	ax1.plot(v[:,j],label=repr(parts[j]))
	ax2.plot(t,v[:,j],alpha=0)

ax1.plot(0*np.ones(N))

ax1.legend()
ax1.set_ylabel('v_r')
ax1.set_xlabel('Time Step')
ax2.set_xlabel('t')

ax3 = plt.subplot(4,1,3)

for j in range(Np):

	ax3.plot(t,q[:,j],label=repr(parts[j]))

ax3.legend()

ax3.set_ylabel('Nel')
ax3.set_xlabel('t')

ax4 = plt.subplot(4,1,4)

Vmax = -ob.phi_min[200*16:200*17]
ax4.plot(Vmax)
ax4.set_xlabel('Time Step')
ax4.set_ylabel('Vmax')

plt.show(block=False)
