import csv
import math
import os
import pathlib
import time

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import fmin

class TrapFieldProfile:
    def __init__(self, main_field, trap_current, penning_voltage = 0):

        # TODO: May want to protect these variables with underscores?
        # TODO: Add in trap radius as an attribute?

        self.trap_current = trap_current
        self.main_field = main_field
        self.penning_voltage = penning_voltage

        self.Bz = self.initialize_Bz_interp()
        self.voltage = self.initialize_voltage_interp()

    def initialize_Bz_interp(self):
        """Returns function object f(rho, z) which returns magnetic field (magnitudes?) as a function of position"""
        # TODO: HARDCODED WITH RESPECT TO CACHED FIELD MAP
        # Config decay_cell_radius changes waveguide propagation, not trap. Regenerate trap if you so dersire
        waveguide_radius = 0.578e-2  # (m)
        trap_zmax = 10e-2  # (m)

        grid_edge_length = 4e-4  # (m), it was found that grid_edge_length = 5e-4 results in 1ppm agreement between Bz and Bz_interp

        rho_array = np.arange(0, waveguide_radius, grid_edge_length)
        z_array = np.arange(-trap_zmax, trap_zmax, grid_edge_length)

        dir_path = pathlib.Path(__file__).parents[0]

        pkl_path = dir_path / "trap_field_profile_pkl/2021_trap_profile_mainfield_0T_trap_1A.csv"

        try:
            with open(pkl_path, "r") as pkl_file:
                map_array = np.loadtxt(pkl_file)

        except IOError as e:
            print("Do you have a field map here: {} ".format(pkl_path))
            raise e

        # Adjust the field values so they align with the given trap configuration.
        map_array = map_array * self.trap_current + self.main_field
        map_array = np.transpose(map_array)
        # Now use the map_array to do the interpolation.
        field_interp = RectBivariateSpline(rho_array, z_array, map_array)

        #return evaluation function for use
        return field_interp.ev

    def initialize_voltage_interp(self):
        """Returns function object f(rho, z) which returns voltage as a function of position"""
        #TODO: I assume the fld file has been stripped such that all x == 0.
        #Unsure how to use non-azimuthally symmetric potential in cyclotron+axial+magnetron framework
        # TODO: voltage of electrodes in the simulation
        hardcoded_reference_voltage = 100.

        dir_path = pathlib.Path(__file__).parents[0]
        pkl_path = dir_path / "trap_field_profile_pkl/2026_exb_electrodes.fld"

        try:
            with open(pkl_path, "r") as pkl_file:
                map_array = np.loadtxt(pkl_file, skiprows = 2 ) #skip header

        except IOError as e:
            print("Do you have a field map here: {} ".format(pkl_path))
            raise e

        x, y, z, V = map_array.T

        ###################### Interpolate #########################
        #Grid Output Min: [-5mm -5mm -5cm] Max: [5mm 5mm 5cm] Grid Size: [0.1mm 0.1mm 0.1cm]
        #non-zero delta includes the "final" point in this...
        delta = 1e-9
        rho_array = np.arange(-5e-3, 5e-3 + delta, 1e-4)
        z_array = np.arange(-5e-2, 5e-2 + delta, 1e-3)

        Nrho = len(rho_array)
        Nz = len(z_array)

        V_array = V.reshape(Nrho, Nz)

        # Adjust the field values so they align with the given trap configuration.
        V_array *= self.penning_voltage / hardcoded_reference_voltage

        # Now use the map_array to do the interpolation.
        volt_interp = RectBivariateSpline(rho_array, z_array, V_array)

        #return evaluation function for use
        return volt_interp.ev
