import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, os
BIN = os.path.expanduser("../../../")
sys.path.append(BIN)
sim_input_folder = '../../testing/tests_buildup/LHC_ArcDipReal_450GeV_sey1.70_2.5e11ppb_bl_1.00ns'
from PyECLOUD.buildup_simulation import BuildupSimulation
from scipy.constants import e as qe
import PyECLOUD.hist_for as histf
import PyECLOUD.myloadmat_to_obj as mlo

ob = mlo.myloadmat_to_obj('Pyecltest')

Np = range(280,291)
cmap = mpl.cm.cool
Ne = 100

ind = range(0,Ne)

Z = [[0,0],[0,0]]
levels = Np#+0.5*np.ones(len(Np))
CS = plt.contourf(Z, levels, cmap=cmap)
plt.clf()
i = 0.
for n in Np:
    nnz_ind = list(np.where(ob.En_hist[n,:] !=0)[0])
    y = ob.forces_ratio_hist[n,:]/sum(ob.forces_ratio_hist[n,:])
    win = 15
    y_smooth = np.convolve(y, np.ones((win,))/win, mode='same')
    plt.semilogx(y_smooth,color=cmap(i / float(len(Np))))
    i = i + 1

cbar = plt.colorbar(CS)
ax = cbar.ax
ax.text(3,0.6,'Bunch Passage',rotation=270)
plt.xlabel(r'$||Fe|/||Fb||$')
plt.title('Fe vs Fb histogram')
plt.show()
