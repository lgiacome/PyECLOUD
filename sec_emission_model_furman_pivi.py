from __future__ import division, print_function
import numpy as np
import scipy.special
import numpy.random as random


class SEY_model_Furman_Pivi(object):
    def __init__(self, max_secondaries):
        self.max_secondaries = max_secondaries
        self.possible_electrons_vect = np.arange(0, max_secondaries+0.01, 1., dtype=float)

    def SEY_process(self, nel_impact, E_impact_eV, costheta_impact, i_impact):
        # must return delta_ref_frac

        # Furman-Pivi algorithm
        # (1): Compute emission angles and energy: already part of the impact_man class.

        # (2): Compute delta_e, delta_r, delta_ts
        delta_e = self._delta_e(E_impact_eV, costheta_impact)
        delta_r = self._delta_r(E_impact_eV, costheta_impact)
        delta_ts = self._delta_ts(E_impact_eV, costheta_impact)


        # (3): Generate probability of number electrons created

        # Emission probability per penetrated electron
        # (39)

        # Decide on type
        rand = random.rand(E_impact_eV.size)
        flag_rediffused = rand < delta_r
        flag_backscattered = np.logical_and(~flag_rediffused, rand < delta_r+delta_e)
        flag_truesec = np.logical_and(~flag_rediffused, ~flag_backscattered)

        # Reflected or backscattered electrons have yield 1 by definition.
        delta = np.ones_like(E_impact_eV, dtype=float)
        # True secondary part has to be adjusted accordingly.
        delta[flag_truesec] = delta_ts[flag_truesec] / (1.-delta_r[flag_truesec]-delta_e[flag_truesec])

        # (4): Generate number of secondaries for every impact
        # Since this is a macro-particle code, where every MP stands for many electrons,
        # this step is omitted.

        # (5): Delete if n = 0
        # Done automatically by the MP system.

        # (6): Generate energy:
        # In impact_management_system

        nel_emit = delta*nel_impact

        return nel_emit, flag_backscattered, flag_rediffused, flag_truesec


    def _delta_e(self, E_impact_eV, costheta_impact):
        """
        Backscattered electrons (elastically scattered).
        (25) in FP paper.
        """

        exp_factor = -(np.abs(E_impact_eV - self.eEHat)/self.w)**self.p / self.p
        delta_e0 = self.p1EInf + (self.p1Ehat-self.p1EInf)*np.exp(exp_factor)
        angular_factor = 1. + self.e1*(1.-np.cos(costheta_impact)**self.e2)

        return delta_e0 * angular_factor

    def _delta_r(self, E_impact_eV, costheta_impact):
        """
        Rediffused electrons (not in ECLOUD model).
        (28) in FP paper.
        """

        exp_factor = -(E_impact_eV/self.eR)**self.r
        delta_r0 = self.p1RInf * (1.-np.exp(exp_factor))
        angular_factor = 1. + self.r1*(1.-np.cos(costheta_impact)**self.r2)

        return delta_r0 * angular_factor

    def _delta_ts(self, E_impact_eV, costheta_impact):
        """
        True secondaries.
        (31) in FP paper.
        """

        eHat = self.eHat0 * (1. + self.t3*(1. - np.cos(costheta_impact)**self.t4))
        delta_ts0 = self.deltaTSHat * self._D(E_impact_eV/eHat)
        angular_factor = 1. + self.t1*(1.-np.cos(costheta_impact)**self.t2)

        return delta_ts0 * angular_factor

    def _D(self, x):
        s = self.s
        return s*x / (s-1+x**s)

    def energy_rediffused(self, E0):
        randn = random.rand(len(E0))
        return randn**(1/(self.q+1))*E0

    def energy_trueSecondary(self, E0):
        u = np.random.rand(len(E0))
        E_out = self.eHat0

        return E_out



class SEY_model_FP_Cu(SEY_model_Furman_Pivi):

    # Parameters for backscattered (elastically scattered) electrons
    # (25) in FP paper
    p1EInf      = 0.02      # Minimum probability of elastic scattering (at infinite energy)
    p1Ehat      = 0.496     # Peak probability
    eEHat       = 0.        # Peak energy
    w           = 60.86     # Exponential factor 1
    p           = 1.        # Exponential factor 2
    # (47a)                 # Angular factors
    e1          = 0.26
    e2          = 2.
    # (26)
    sigmaE      = 2.

    # Parameters for rediffused electrons
    # (28)
    p1RInf      = 0.2       # Minimum probability of rediffused scattering (at infinite energy)
    eR          = 0.041     # Peak energy
    r           = 0.104     # Exponential factor
    # (29)
    q           = 0.5
    # (47b)                 # Angular factors
    r1          = 0.26
    r2          = 2.

    # Parameters for true secondaries
    # (31)
    deltaTSHat  = 1.8848    # Maximum probability of secondaries
    eHat0       = 276.8     # Peak enery
    # (32)
    s           = 1.54      # Form factor of fitting curve
    # (48a)                 # Angular factors
    t1          = 0.66
    t2          = 0.8
    # (48b)
    t3          = 0.7
    t4          = 1.

