import os, sys

BIN = os.path.expanduser("../../../") #folder containing PyECLOUD, PyPIC, PyKLU
if BIN not in sys.path:
    sys.path.append(BIN)

import PyECLOUD.myloadmat_to_obj as mlm

import matplotlib.pyplot as plt

obn = mlm.myloadmat_to_obj('LHC_ArcDipReal_6500GeV_sey_1.70_1.1e11ppb_b1_1.00ns/Pyecltest_angle3D_nunif_x4.mat')
ob4 = mlm.myloadmat_to_obj('LHC_ArcDipReal_6500GeV_sey_1.70_1.1e11ppb_b1_1.00ns/Pyecltest_angle3D_unif_x4.mat')
ob9 = mlm.myloadmat_to_obj('LHC_ArcDipReal_6500GeV_sey_1.70_1.1e11ppb_b1_1.00ns/Pyecltest_angle3D_unif_x9.mat')
ob20 = mlm.myloadmat_to_obj('LHC_ArcDipReal_6500GeV_sey_1.70_1.1e11ppb_b1_1.00ns/Pyecltest_angle3D_unif_x20.mat')

# obr = mlm.myloadmat_to_obj('LHC_ArcDipReal_6500GeV_sey_1.70_1.1e11ppb_b1_1.00ns/Pyecltest_angle3D_ref.mat')

# obr = mlm.myloadmat_to_obj('LHC_ArcDipReal_450GeV_sey1.00_2.5e11ppb_bl_1.00ns_gas_ionization/Pyecltest_angle3D_ref.mat')
# ob4 = mlm.myloadmat_to_obj('LHC_ArcDipReal_450GeV_sey1.00_2.5e11ppb_bl_1.00ns_gas_ionization/Pyecltest_angle3D_x4.mat')
# ob9 = mlm.myloadmat_to_obj('LHC_ArcDipReal_450GeV_sey1.00_2.5e11ppb_bl_1.00ns_gas_ionization/Pyecltest_angle3D_x9.mat')
# obn = mlm.myloadmat_to_obj('LHC_ArcDipReal_450GeV_sey1.00_2.5e11ppb_bl_1.00ns_gas_ionization/Pyecltest_angle3D.mat')


plt.close('all')
plt.figure(1)
# plt.plot(obr.t, obr.Nel_timep, '.-')
plt.plot(ob4.t, ob4.Nel_timep, 'r.-')
plt.plot(ob9.t, ob9.Nel_timep, 'g.-')
plt.plot(obn.t, obn.Nel_timep, 'k.-')
plt.plot(ob20.t, ob20.Nel_timep, 'm.-')

plt.figure(2)
plt.plot(ob4.t, ob4.En_imp_eV_time, 'r.-')
plt.plot(ob9.t, ob9.En_imp_eV_time, 'g.-')
plt.plot(obn.t, obn.En_imp_eV_time, 'k.-')
plt.plot(ob20.t, ob20.En_imp_eV_time, 'm.-')



# plt.plot(obn.t, obn.Nel_timep, '.r')
plt.show()
