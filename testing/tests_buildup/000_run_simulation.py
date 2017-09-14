import sys, os
import time
BIN = os.path.expanduser("../../../") #folder containing PyECLOUD, PyPIC, PyKLU
if BIN not in sys.path:
    sys.path.append(BIN)
import argparse

from PyECLOUD.buildup_simulation import BuildupSimulation

sim_folder = 'LHC_ArcDipReal_450GeV_sey1.70_2.5e11ppb_bl_1.00ns'
#sim_folder = 'LHC_ArcDipReal_450GeV_sey1.00_2.5e11ppb_bl_1.00ns_gas_ionization'
#sim_folder = 'LHC_ArcDipReal_450GeV_sey1.70_2.5e11ppb_bl_1.00ns_change_s_and_E0'
#sim_folder = 'LHC_ArcDipReal_450GeV_sey1.70_2.5e11ppb_bl_1.00ns_multigrid'
#sim_folder = 'LHC_ArcQuadReal_450GeV_sey1.65_2.5e11ppb_bl_1.00ns'
#sim_folder = 'LHC_ArcQuadReal_450GeV_sey1.65_2.5e11ppb_bl_1.00ns_circular'
#sim_folder = 'LHC_ArcQuadReal_450GeV_sey1.65_2.5e11ppb_bl_1.00ns_skew'
#sim_folder = 'LHC_ArcQuadReal_450GeV_sey1.65_2.5e11ppb_bl_1.00ns_skew_circular'
#sim_folder = 'LHC_Sextupole_450GeV_sey1.65_2.5e11ppb_bl_1.00ns'
#sim_folder = 'LHC_Sextupole_450GeV_sey1.65_2.5e11ppb_bl_1.00ns_skew'
#sim_folder = './LHC_Octupole_6500GeV_sey1.65_2.5e11ppb_b1_1.00ns'
#sim_folder = './LHC_Octupole_6500GeV_sey1.65_2.5e11ppb_b1_1.00ns_skew'


# check if user provided folder as command line argument
parser = argparse.ArgumentParser()
parser.add_argument('--folder', help='Simulation_folder')
parser.add_argument('--angle-dist-func', help='Angular distribution of new MPs relative to surface normal. Introduced in July 2017.', choices=('2D', '3D'), default='3D')
args = parser.parse_args()
if args.folder:
    sim_folder = args.folder

angle_distribution = 'cosine_%s' % args.angle_dist_func
filen_main_outp = sim_folder+'/Pyecltest_angle%s.mat' % args.angle_dist_func


time_0 = time.time()
sim = BuildupSimulation(pyecl_input_folder = sim_folder, filen_main_outp=filen_main_outp,
                        secondary_angle_distribution=angle_distribution, photoelectron_angle_distribution=angle_distribution)
sim.run()

time_needed = time.time() - time_0


print('')
print('Test simulation done in %.2f s!' % time_needed)
print('To inspect the results you can run:')
print('001_comparison_against_reference.py')
print('')
