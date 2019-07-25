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

def _forces_ratio_hist_init(sim):
    # Energy histogram init
    sim.cloud_list[0].pyeclsaver.Dratio_hist = 1000
    sim.cloud_list[0].pyeclsaver.max_ratio_hist =  1.e7
    sim.cloud_list[0].pyeclsaver.min_ratio_hist = 0
    sim.cloud_list[0].pyeclsaver.Dt_forces_ratio_hist = 2.5e-9
    sim.cloud_list[0].pyeclsaver.forces_ratio_hist = []
    sim.cloud_list[0].pyeclsaver.t_last_forces_ratio_hist = -1
    sim.cloud_list[0].pyeclsaver.t_forces_ratio_hist = []
    return 0


def _forces_ratio_hist_save(sim):
    if sim.beamtim.tt_curr >= sim.cloud_list[0].pyeclsaver.t_last_forces_ratio_hist + sim.cloud_list[0].pyeclsaver.Dt_forces_ratio_hist or np.isclose(sim.beamtim.tt_curr, sim.cloud_list[0].pyeclsaver.t_last_forces_ratio_hist + sim.cloud_list[0].pyeclsaver.Dt_forces_ratio_hist, rtol=1.e-10, atol=0.0):
            MP_e = sim.cloud_list[0].MP_e
            N_mp = sim.cloud_list[0].MP_e.N_mp
            efx, efy, bfx, bfy, bfz = sim.spacech_ele.get_sc_em_field(MP_e)
            #compute the forces (on the maforces_ratios_hist_linecro-particles)
            Fefx_mp = qe*np.multiply(MP_e.nel_mp[0:N_mp],efx)
            Fefy_mp = qe*np.multiply(MP_e.nel_mp[0:N_mp],efy)
            Fbfx_mp = qe*np.multiply(MP_e.nel_mp[0:N_mp],np.multiply(MP_e.vy_mp[0:N_mp],bfz)-qe*np.multiply(MP_e.vz_mp[0:N_mp],bfy))
            Fbfy_mp = qe*np.multiply(MP_e.nel_mp[0:N_mp],np.multiply(MP_e.vz_mp[0:N_mp],bfx)-qe*np.multiply(MP_e.vx_mp[0:N_mp],bfz))
            Fbfz_mp = qe*np.multiply(MP_e.nel_mp[0:N_mp],np.multiply(MP_e.vx_mp[0:N_mp],bfy)-qe*np.multiply(MP_e.vy_mp[0:N_mp],bfx))
            Fe_norm_mp = np.sqrt(np.multiply(Fefx_mp,Fefx_mp)+np.multiply(Fefy_mp,Fefy_mp))
            Fb_norm_mp = np.sqrt(np.multiply(Fbfx_mp,Fbfx_mp)+np.multiply(Fbfy_mp,Fbfy_mp))
            mask = Fe_norm_mp!=0
            forces_ratios = np.divide(Fe_norm_mp[mask],Fb_norm_mp[mask])
            forces_ratios_hist_line = np.zeros(int(np.ceil((sim.cloud_list[0].pyeclsaver.max_ratio_hist-sim.cloud_list[0].pyeclsaver.min_ratio_hist)/sim.cloud_list[0].pyeclsaver.Dratio_hist)))
            nel_mp = MP_e.nel_mp[0:MP_e.N_mp]
            if len(nel_mp[mask])>0:
                histf.compute_hist(forces_ratios, nel_mp[mask], 0., sim.cloud_list[0].pyeclsaver.Dratio_hist, forces_ratios_hist_line)
            sim.cloud_list[0].pyeclsaver.forces_ratio_hist.append(forces_ratios_hist_line.copy())
            sim.cloud_list[0].pyeclsaver.t_forces_ratio_hist.append(sim.beamtim.tt_curr)
            sim.cloud_list[0].pyeclsaver.t_last_forces_ratio_hist = sim.beamtim.tt_curr
    return 0


step_by_step_custom_observables = {
        'dummy2': lambda sim : _forces_ratio_hist_save(sim)
        }


sim = BuildupSimulation(pyecl_input_folder=sim_input_folder,
        filen_main_outp='./Pyecltest.mat',
        extract_sey=False,
        step_by_step_custom_observables = step_by_step_custom_observables)

_forces_ratio_hist_init(sim)

sim.run()

ob = mlo.myloadmat_to_obj(sim_input_folder+'/Pyecltest_angle3D.mat')

Np = range(280,290)
cmap = mpl.cm.cool
Ne = 100

ind = range(0,Ne)

Z = [[0,0],[0,0]]
levels = Np#+0.5*np.ones(len(Np))
CS = plt.contourf(Z, levels, cmap=cmap)
plt.clf()
i = 0.
for n in Np:
 	plt.loglog(ob.forces_ratio_hist[n,:]/sum(ob.forces_ratio_hist[n,:]),color=cmap(i / float(len(Np))))
	i = i + 1

cbar = plt.colorbar(CS)
ax = cbar.ax
ax.text(3,0.6,'Bunch passage',rotation=270)
plt.xlabel(r'$E$')
plt.ylabel(r'$\phi(E)$')
plt.show()
