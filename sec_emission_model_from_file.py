from __future__ import division, print_function
import os

import numpy as np
import scipy.io as sio
from numpy.random import rand

class SEY_model_from_file(object):

    def __init__(self, sey_file, flag_factor_costheta):
        """
        - sey file is the path to a mat file of the correct format, either an absolute path or in the sey_files folder.
        - if flag_factor_costheta is True, the SEY is increased depending on the angle with which the electrons are hitting.
            Set to None or False to disable.
        """
        if type(sey_file) is dict:
            sey_properties = sey_file
        else:
            # Search for file with given name, either as an absolute path or in the dedicated folder
            # Absolute path has precedence.
            candidate_files = [
                os.path.expanduser(sey_file),
                os.path.abspath(os.path.dirname(__file__)) + '/sey_files/' + sey_file,
            ]
            existing_files = filter(os.path.isfile, candidate_files)
            if not existing_files:
                raise ValueError('SEY file %s is not found' % sey_file)
            sey_file_real = existing_files[0]
            print('Secondary emission from file %s' % sey_file_real)

            sey_properties = sio.loadmat(sey_file)
            
        
        if flag_factor_costheta not in [True, False]:
            raise ValueError('SEY_model_from_file:\nIn this mode flag_factor_costheta must be explicitely specified')
        
        energy_eV = sey_properties['energy_eV'].squeeze()
        sey_true = sey_properties['sey_true'].squeeze()
        sey_elast = sey_properties['sey_elast'].squeeze()
        extrapolate_grad_true = float(sey_properties['extrapolate_grad_true'].squeeze())
        extrapolate_const_true = float(sey_properties['extrapolate_const_true'].squeeze())
        extrapolate_grad_elast = float(sey_properties['extrapolate_grad_elast'].squeeze())
        extrapolate_const_elast = float(sey_properties['extrapolate_const_elast'].squeeze())

        diff_e = np.round(np.diff(energy_eV), 3)
        delta_e = diff_e[0]
        if np.any(diff_e != delta_e):
            raise ValueError('Energy in file %s is not equally spaced.' % sey_file_real)

        # sey_diff is needed by the interp function
        # A 0 is appended because this last element is never needed but the array must have the correct shape
        self.sey_true_diff = np.append(np.diff(sey_true), 0.)
        self.sey_elast_diff = np.append(np.diff(sey_elast), 0.)
        

        # This merely populates the object namespace
        self.energy_eV              = energy_eV
        self.sey_true               = sey_true
        self.sey_elast              = sey_elast
        self.sey_file               = sey_file
        self.flag_factor_costheta   = flag_factor_costheta
        self.energy_eV_min          = energy_eV.min()
        self.energy_eV_max          = energy_eV.max()
        self.delta_e                = delta_e
        self.extrapolate_grad_true  = extrapolate_grad_true
        self.extrapolate_const_true = extrapolate_const_true
        self.extrapolate_grad_elast  = extrapolate_grad_elast
        self.extrapolate_const_elast = extrapolate_const_elast
        
    def SEY_values(self, E_impact_eV, costheta_impact):
        delta_true = np.zeros_like(E_impact_eV, dtype=float)
        delta_elast= np.zeros_like(E_impact_eV, dtype=float)
        
        mask_fit = (E_impact_eV > self.energy_eV_max)
        mask_regular = ~mask_fit

        delta_true[mask_regular], delta_elast[mask_regular] = self.interp(E_impact_eV[mask_regular])
        
        delta_true[mask_fit] = self.extrapolate_const_true + self.extrapolate_grad_true * E_impact_eV[mask_fit]
        delta_elast[mask_fit] = self.extrapolate_const_elast + self.extrapolate_grad_elast * E_impact_eV[mask_fit]
        
        delta_true[delta_true<1e-10] = 0. # We get rid of negative values
        delta_elast[delta_elast<1e-10] = 0. # We get rid of negative values
        
        if self.flag_factor_costheta:
            factor_costheta = np.exp(0.5*(1.-costheta_impact))
        else:
            factor_costheta = 1.
            
        delta = delta_true*factor_costheta + delta_elast
        
        ref_frac=0.*delta
        mask_non_zero=(delta>0)
        ref_frac[mask_non_zero]=delta_elast[mask_non_zero]/delta[mask_non_zero]
        
        return delta, ref_frac
        

    def SEY_process(self, nel_impact,E_impact_eV, costheta_impact, i_impact):
        
        yiel, ref_frac = self.SEY_values(E_impact_eV, costheta_impact)
        flag_elast=(rand(len(ref_frac))<ref_frac);
        flag_truesec=~(flag_elast);
        nel_emit=nel_impact*yiel;

        return nel_emit, flag_elast, flag_truesec

    def interp(self, energy_eV):
        """
        Linear interpolation of the energy - SEY curve.
        """
        index_float = (energy_eV - self.energy_eV_min)/self.delta_e
        index_remainder, index_int = np.modf(index_float)
        index_int = index_int.astype(int)
        return self.sey_true[index_int] + index_remainder*self.sey_true_diff[index_int], self.sey_elast[index_int] + index_remainder*self.sey_elast_diff[index_int]

    def interp_regular(self, energy_eV):
        #This fails if the input is not in ascending order.
        #return np.interp(energy_eV, self.energy_eV, self.sey_parameter)
        raise ValueError('Warning! Do not use interp_regular!')
