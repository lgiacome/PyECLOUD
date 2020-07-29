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

from numpy import array, cross, sum, squeeze
import scipy.io as sio
from . import int_field_for as iff
from .boris_step import boris_step


def crprod(bx, by, bz, cx, cy, cz):
    ax = by * cz - bz * cy
    ay = bz * cx - bx * cz
    az = bx * cy - by * cx

    return ax, ay, az


class B_none():

    def get_B(self, xn, yn):
        Bx_n = 0 * xn
        By_n = 0 * xn
        Bz_n = 0 * xn
        return Bx_n, By_n, Bz_n


class E_none():

    def get_E(self, xn, yn):
        Ex_n = 0 * xn
        Ey_n = 0 * xn
        Ez_n = 0 * xn
        return Ex_n, Ey_n, Ez_n


class B_quad():

    def __init__(self, fact_Bmap):
        self.B0x = B0x
        self.B0y = B0y
        self.B0z = B0z
        self.fact_Bmap = fact_Bmap

    def get_B(self, xn, yn):
        Bx_n = self.fact_Bmap * yn.copy()
        By_n = self.fact_Bmap * xn.copy()
        Bx_n = Bx_n
        By_n = By_n
        Bz_n = 0 * xn
        return Bx_n, By_n, Bz_n


class B_file():

    def __init__(self, fact_Bmap, B_map_file):
        print('Loading B map')
        dict_Bmap = sio.loadmat(B_map_file)

        self.Bmap_x = fact_Bmap * squeeze(dict_Bmap['Bx'].real)
        self.Bmap_y = fact_Bmap * squeeze(dict_Bmap['By'].real)
        self.xx = squeeze(dict_Bmap['xx'].T)
        self.yy = squeeze(dict_Bmap['yy'].T)

        self.xmin = min(self.xx)
        self.ymin = min(self.yy)
        self.dx = self.xx[1] - self.xx[0]
        self.dy = self.yy[1] - self.yy[0]

    def get_B(self, xn, yn):
        Bx_n, By_n = iff.int_field(xn, yn, self.xmin, self.ymin,
                                   self.dx, self.dy, self.Bmap_x, self.Bmap_y)
        # the rescaling factor has already been applied to the map
        Bx_n = Bx_n
        By_n = By_n
        Bz_n = 0 * xn
        return Bx_n, By_n, Bz_n


class E_file():

    def __init__(self, fact_Emap, E_map_file):
        print('Loading E map')
        dict_Emap = sio.loadmat(E_map_file)

        self.Emap_x = fact_Emap * squeeze(dict_Emap['Ex'].real)
        self.Emap_y = fact_Emap * squeeze(dict_Emap['Ey'].real)
        self.xx = squeeze(dict_Emap['xx'].T)
        self.yy = squeeze(dict_Emap['yy'].T)

        self.xmin = min(self.xx)
        self.ymin = min(self.yy)
        self.dx = self.xx[1] - self.xx[0]
        self.dy = self.yy[1] - self.yy[0]

    def get_E(self, xn, yn):
        Ex_n, Ey_n = iff.int_field(xn, yn, self.xmin, self.ymin,
                                   self.dx, self.dy, self.Emap_x,
                                   self.Emap_y)
        # the rescaling factor has already been applied to the map
        Ex_n = Ex_n
        Ey_n = Ey_n
        Ez_n = 0 * xn
        return Ex_n, Ey_n, Ez_n


def const_func(t):
    return 1.


def input_to_list(var):
    if not hasattr(var, "__len__"):
        return [var]

    else:
        return var


class pusher_Boris():

    def __init__(self, Dt, B0x, B0y, B0z, E0x, E0y, E0z,
                 B_map_file, fact_Bmap, Bz_map_file,
                 E_map_file, fact_Emap, Ez_map_file,
                 N_sub_steps=1, B_time_func=None, E_time_func=None):

        print("Tracker: Boris")

        self.Dt = Dt
        self.N_sub_steps = N_sub_steps
        self.Dtt = Dt / float(N_sub_steps)
        self.time = 0.
        self.B0x = B0x
        self.B0y = B0y
        self.B0z = B0z
        self.E0x = E0x
        self.E0y = E0y
        self.E0z = E0z

        # convert inputs into lists if needed
        self.B_map_file_list = input_to_list(B_map_file)
        self.E_map_file_list = input_to_list(E_map_file)
        self.fact_Bmap_list = input_to_list(fact_Bmap)
        self.fact_Emap_list = input_to_list(fact_Emap)

        # if no time dependence is specified, use constant
        if B_time_func is None:
            B_time_func = [const_func]*len(self.fact_Bmap_list)

        if E_time_func is None:
            E_time_func = [const_func]*len(self.fact_Emap_list)

        # convert inputs into lists if needed
        self.B_time_func_list = input_to_list(B_time_func)
        self.E_time_func_list = input_to_list(E_time_func)

        self.B_ob_list = []
        self.E_ob_list = []

        # raise an exception if the length of the lists do not correspond
        if not (len(self.B_map_file_list) == len(self.fact_Bmap_list)
                == len(self.B_time_func_list)):
            raise(ValueError('B_map_file, fact_Bmap, B_time_func must have '
                             'same length'))

        if not (len(self.E_map_file_list) == len(self.fact_Emap_list)
                == len(self.E_time_func_list)):
            raise(ValueError('E_map_file, fact_Emap, E_time_Eunc must have '
                             'same length'))

        # initialize external field objects
        for i, B_map_file in enumerate(B_map_file_list):
            if B_map_file is None:
                self.B_ob_list.append(B_none())

            elif B_map_file is 'analytic_qaudrupole_unit_grad':
                print("B map analytic quadrupole")
                self.B_ob_list.append(B_quad(fact_Bmap))

            else:
                self.B_ob_list.append(B_file(fact_Bmap_list[i],
                                             B_map_file_list[i]))

        for i, E_map_file in enumerate(E_map_file_list):
            if E_map_file is None:
                self.E_ob_list.append(E_none())

            else:
                self.E_ob_list.append(E_file(fact_Emap_list[i],
                                             E_map_file[i]))

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
                for i, B_ob in enumerate(self.B_ob_list):
                    Bx_map, By_map, Bz_n_map = B_ob.get_B(xn1, yn1)
                    time_fact = self.B_time_func_list[i](self.time)
                    Bx_n += (Bx_map + self.B0x) * time_fact
                    By_n += (By_map + self.B0y) * time_fact
                    Bz_n += (Bz_map + self.B0z) * time_fact

                # add external E field contributions
                for i, E_ob in enumerate(self.E_ob):
                    Ex_map, Ey_map, Ez_n_map = self.E_ob_i.get_E(xn1, yn1)
                    time_fact = self.E_time_func_list[i](self.time)
                    Ex_n += (Ex_map + self.E0x) * time_fact
                    Ey_n += (Ey_map + self.E0y) * time_fact
                    Ez_n += (Ez_map + self.E0z) * time_fact

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
                for i, B_ob in enumerate(self.B_ob_list):
                    Bx_map, By_map, Bz_n_map = B_ob.get_B(xn1, yn1)
                    time_fact = self.B_time_func_list[i](self.time)
                    Bx_n += (Bx_map + self.B0x) * time_fact
                    By_n += (By_map + self.B0y) * time_fact
                    Bz_n += (Bz_map + self.B0z) * time_fact

                # add external E field contributions
                for i, E_ob in enumerate(self.E_ob):
                    Ex_map, Ey_map, Ez_n_map = self.E_ob_i.get_E(xn1, yn1)
                    time_fact = self.E_time_func_list[i](self.time)
                    Ex_n += (Ex_map + self.E0x) * time_fact
                    Ey_n += (Ey_map + self.E0y) * time_fact
                    Ez_n += (Ez_map + self.E0z) * time_fact
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
