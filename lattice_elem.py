import numpy as np
from . import int_field_for as iff


def const_func(t):
    return 1.


class EGriddedElem():
    def __init__(self, E0x, E0y, E0z, E_map_file, fact_Emap,
                 E_time_func=const_func):
        self.E0x = E0x
        self.E0y = E0y
        self.E0z = E0z
        self.E_map_file = E_map_file
        self.E_time_func = E_time_func

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

    def get_E(self, xn, yn, t):
        Ex_n, Ey_n = iff.int_field(xn, yn, self.xmin, self.ymin,
                                   self.dx, self.dy, self.Emap_x,
                                   self.Emap_y)
        # the rescaling factor has already been applied to the map
        Ex_n = Ex_n*self.E_time_func(t)
        Ey_n = Ey_n*self.E_time_func(t)
        Ez_n = 0 * xn
        return Ex_n, Ey_n, Ez_n


class BGriddedElem():
    def __init__(self, B0x, B0y, B0z, B_map_file, fact_Bmap,
                 B_time_func=const_func):
        self.B0x = B0x
        self.B0y = B0y
        self.B0z = B0z
        self.B_map_file = B_map_file
        self.B_time_func = B_time_func

        print('Loading B map')
        dict_Bmap = sio.loadmat(B_map_file)

        self.Bmap_x = fact_Bmap * squeeze(dict_Bmap['Ex'].real)
        self.Bmap_y = fact_Bmap * squeeze(dict_Bmap['Ey'].real)
        self.xx = squeeze(dict_Bmap['xx'].T)
        self.yy = squeeze(dict_Bmap['yy'].T)

        self.xmin = min(self.xx)
        self.ymin = min(self.yy)
        self.dx = self.xx[1] - self.xx[0]
        self.dy = self.yy[1] - self.yy[0]

    def get_B(self, xn, yn, t):
        Bx_n, By_n = iff.int_field(xn, yn, self.xmin, self.ymin,
                                   self.dx, self.dy, self.Bmap_x,
                                   self.Bmap_y)
        # the rescaling factor has already been applied to the map
        Bx_n = Bx_n*self.B_time_func(t)
        By_n = By_n*self.B_time_func(t)
        Bz_n = 0 * xn
        return Bx_n, By_n, Bz_n


class B_quad():

    def __init__(self, fact_Bmap, B_time_func=const_func):
        self.B0x = B0x
        self.B0y = B0y
        self.B0z = B0z
        self.fact_Bmap = fact_Bmap
        self.B_time_func = B_time_func

    def get_B(self, xn, yn, t=0):
        Bx_n = self.fact_Bmap * yn.copy()
        By_n = self.fact_Bmap * xn.copy()
        Bx_n = Bx_n*self.B_time_func(t)
        By_n = By_n*self.B_time_func(t)
        Bz_n = 0 * xn
        return Bx_n, By_n, Bz_n


class B_multipole():
    def __init__(self, B_multip=None, B_skew=None, B0x=None, B0y=None,
                 B0z=None, B_time_func=const_func):

        self.B0x = B0x
        self.B0y = B0y
        self.B0z = B0z
        self.B_time_func = B_time_func

        if B_multip is None or len(B_multip) == 0:
            B_multip = np.array([0.], dtype=float)

        # B_multip are derivatives of B_field
        # B_field are field strengths at x=1 m, y=0
        factorial = np.array(
            [math.factorial(ii) for ii in range(len(B_multip))], dtype=float)
        self.B_field = np.array(B_multip, dtype=float) / factorial
        if B_skew is None:
            self.B_field_skew = np.zeros_like(self.B_field, dtype=float)
        else:
            self.B_field_skew = np.array(B_skew, dtype=float) / factorial

        self.N_multipoles = len(B_multip)

    def get_B(self, xn, yn, t=0):
        rexy = 1.
        imxy = 0.
        By_n = B_field[0]
        Bx_n = B_skew[0]
        Bz_n = 0.
        for order in range(1, self.N_multipoles):
            # rexy, imxy correspond to real, imaginary
            # part of(x + iy) ^ (n - 1)
            rexy_0 = rexy
            rexy = rexy_0 * xn - imxy * yn
            imxy = imxy * xn + rexy_0 * yn

            # Bx + iBy = sum[(k + ik')(x + iy)^(n-1) ]
            # where k, k' are the strengths and skew strengths of the magnet
            By_n += ((B_field[order] * rexy - B_skew[order] * imxy) *
                     self.B_time_func(t))
            Bx_n += ((B_field[order] * imxy + B_skew[order] * rexy) *
                     self.B_time_func(t))

        return Bx_n, By_n, Bz_n
