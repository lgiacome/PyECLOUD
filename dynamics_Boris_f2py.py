# -Begin-preamble-------------------------------------------------------
#
#                           CERN
#
#     European Organization for Nuclear Research
#
#
#     This file is part of the code:
#
#                   PyECLOUD Version 8.5.0
#
#
#     Main author:          Giovanni IADAROLA
#                           BE-ABP Group
#                           CERN
#                           CH-1211 GENEVA 23
#                           SWITZERLAND
#                           giovanni.iadarola@cern.ch
#
#     Contributors:         Eleonora Belli
#                           Philipp Dijkstal
#                           Lorenzo Giacomel
#                           Lotta Mether
#                           Annalisa Romano
#                           Giovanni Rumolo
#                           Eric Wulff
#
#
#     Copyright  CERN,  Geneva  2011  -  Copyright  and  any   other
#     appropriate  legal  protection  of  this  computer program and
#     associated documentation reserved  in  all  countries  of  the
#     world.
#
#     Organizations collaborating with CERN may receive this program
#     and documentation freely and without charge.
#
#     CERN undertakes no obligation  for  the  maintenance  of  this
#     program,  nor responsibility for its correctness,  and accepts
#     no liability whatsoever resulting from its use.
#
#     Program  and documentation are provided solely for the use  of
#     the organization to which they are distributed.
#
#     This program  may  not  be  copied  or  otherwise  distributed
#     without  permission. This message must be retained on this and
#     any other authorized copies.
#
#     The material cannot be sold. CERN should be  given  credit  in
#     all references.
#
# -End-preamble---------------------------------------------------------

from numpy import array, cross, sum, squeeze, min
import scipy.io as sio
from . import int_field_for as iff
from .boris_step import boris_step


class pusher_Boris():

    def __init__(self, Dt, lattice_elems_B=None, lattice_elems_E=None,
                 N_sub_steps=1):

        if lattice_elems_E is None:
            lattice_elems_E = []
        if lattice_elems_B is None:
            lattice_elems_B = []

        print("Tracker: Boris")

        self.Dt = Dt
        self.N_sub_steps = N_sub_steps
        self.Dtt = Dt / float(N_sub_steps)
        self.time = 0.

        self.lattice_elems_B = lattice_elems_B
        self.lattice_elems_E = lattice_elems_E

        print("N_subst_init=%d" % self.N_sub_steps)

    # @profile
    def step(self, MP_e, Ex_n, Ey_n, Ez_n=0., Bx_n=0., By_n=0., Bz_n=0.):

        if MP_e.N_mp > 0:

            xn1 = MP_e.x_mp[0:MP_e.N_mp]
            yn1 = MP_e.y_mp[0:MP_e.N_mp]
            zn1 = MP_e.z_mp[0:MP_e.N_mp]
            vxn1 = MP_e.vx_mp[0:MP_e.N_mp]
            vyn1 = MP_e.vy_mp[0:MP_e.N_mp]
            vzn1 = MP_e.vz_mp[0:MP_e.N_mp]

            mass = MP_e.mass
            charge = MP_e.charge

            # make sure field arrays are initialized
            if Ez_n == 0.:
                Ez_n = 0. * xn1
            if Bx_n == 0.:
                Bx_n = 0. * xn1
            if By_n == 0.:
                By_n = 0. * xn1
            if Bz_n == 0.:
                Bz_n = 0. * xn1

            for ii in range(self.N_sub_steps):
                # add external B field contributions
                for B_ob in self.lattice_elems_B:
                    Bx_map, By_map, Bz_n_map = B_ob.get_B(xn1, yn1)
                    time_fact = B_ob.B_time_func(self.time)
                    Bx_n += (Bx_map + B_ob.B0x) * time_fact
                    By_n += (By_map + B_ob.B0y) * time_fact
                    Bz_n += (Bz_map + B_ob.B0z) * time_fact

                # add external E field contributions
                for E_ob in self.lattice_elems_E:
                    Ex_map, Ey_map, Ez_n_map = self.E_ob.get_E(xn1, yn1)
                    time_fact = E_ob.E_time_func(self.time)
                    Ex_n += (Ex_map + E_ob.E0x) * time_fact
                    Ey_n += (Ey_map + E_ob.E0y) * time_fact
                    Ez_n += (Ez_map + E_ob.E0z) * time_fact

                boris_step(self.Dtt, xn1, yn1, zn1, vxn1, vyn1, vzn1,
                           Ex_n, Ey_n, Ez_n, Bx_n, By_n, Bz_n, mass, charge)

                # advance time
                self.time += self.Dtt

            MP_e.x_mp[0:MP_e.N_mp] = xn1
            MP_e.y_mp[0:MP_e.N_mp] = yn1
            MP_e.z_mp[0:MP_e.N_mp] = zn1
            MP_e.vx_mp[0:MP_e.N_mp] = vxn1
            MP_e.vy_mp[0:MP_e.N_mp] = vyn1
            MP_e.vz_mp[0:MP_e.N_mp] = vzn1


        return MP_e

    def stepcustomDt(self, MP_e, Ex_n, Ey_n, Ez_n=0., Bx_n=0., By_n=0., Bz_n=0.,
                     Dt_substep=None, N_sub_steps=None):

        if MP_e.N_mp > 0:

            xn1 = MP_e.x_mp[0:MP_e.N_mp]
            yn1 = MP_e.y_mp[0:MP_e.N_mp]
            zn1 = MP_e.z_mp[0:MP_e.N_mp]
            vxn1 = MP_e.vx_mp[0:MP_e.N_mp]
            vyn1 = MP_e.vy_mp[0:MP_e.N_mp]
            vzn1 = MP_e.vz_mp[0:MP_e.N_mp]

            mass = MP_e.mass
            charge = MP_e.charge

            if Ez_n == 0.:
                Ez_n = 0. * xn1

            for ii in range(N_sub_steps):
                # add external B field contributions
                for B_ob in self.lattice_elems_B:
                    Bx_map, By_map, Bz_n_map = B_ob.get_B(xn1, yn1)
                    time_fact = B_ob.B_time_func(self.time)
                    Bx_n += (Bx_map + B_ob.B0x) * time_fact
                    By_n += (By_map + B_ob.B0y) * time_fact
                    Bz_n += (Bz_map + B_ob.B0z) * time_fact

                # add external E field contributions
                for E_ob in self.lattice_elems_E:
                    Ex_map, Ey_map, Ez_n_map = self.E_ob.get_E(xn1, yn1)
                    time_fact = E_ob.E_time_func(self.time)
                    Ex_n += (Ex_map + E_ob.E0x) * time_fact
                    Ey_n += (Ey_map + E_ob.E0y) * time_fact
                    Ez_n += (Ez_map + E_ob.E0z) * time_fact

                boris_step(Dt_substep, xn1, yn1, zn1, vxn1, vyn1, vzn1,
                           Ex_n, Ey_n, Ez_n, Bx_n, By_n, Bz_n, mass, charge)

                # advance time
                self.time += self.Dt_substep

            MP_e.x_mp[0:MP_e.N_mp] = xn1
            MP_e.y_mp[0:MP_e.N_mp] = yn1
            MP_e.z_mp[0:MP_e.N_mp] = zn1
            MP_e.vx_mp[0:MP_e.N_mp] = vxn1
            MP_e.vy_mp[0:MP_e.N_mp] = vyn1
            MP_e.vz_mp[0:MP_e.N_mp] = vzn1

        return MP_e
