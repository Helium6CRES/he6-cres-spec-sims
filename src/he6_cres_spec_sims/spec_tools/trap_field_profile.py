import csv
import math
import os
import pathlib
import time

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import fmin

class TrapFieldProfile:
    def __init__(self, main_field, trap_current):

        # TODO: May want to protect these variables with underscores?
        # TODO: Would be nice to have an attribute be the relative trap depth.
        # TODO: Add in trap radius as an attribute?

        self.trap_current = trap_current
        self.inverted = trap_current*main_field < 0
        self.main_field = main_field

        self.field_strength = self.initialize_field_strength_interp()
        self.trap_center = self.find_trap_center() if self.inverted else 0 # may vary with rho, need to fully implement

        self.trap_width = self.trap_width_calc()
        
        # min and max fields of trap with rho dependence
        # rho dependence means calling fmin a lot, might be slow
        # TODO: find out if/how much trap_center depends on rho
        self.Bmin = lambda rho = 0: self.field_strength(rho, self.find_trap_center(rho)) # self.field_strength(rho, self.trap_center) 
        self.Bmax = lambda rho = 0: self.main_field if self.inverted else self.field_strength(rho,self.trap_width[1]) 

        # TODO: Actually test to be sure it is a trap.
        self.is_trap = True

        self.relative_depth = (main_field - self.field_strength(0, 0)) / main_field

    def initialize_field_strength_interp(self):
        """Returns function object f(rho, z) which returns magnetic field (magnitudes?) as a function of position"""
        # TODO: hmm I guess these need to be hardcoded for the moment.
        waveguide_radius = 0.578e-2  # (m)
        trap_zmax = 5.5e-2  # (m)

        grid_edge_length = 4e-4  # (m), it was found that grid_edge_length = 5e-4 results in 1ppm agreement between field_stength and field_strength_interp

        rho_array = np.arange(0, waveguide_radius, grid_edge_length)
        z_array = np.arange(-trap_zmax, trap_zmax, grid_edge_length)# + self.trap_center

        dir_path = pathlib.Path(__file__).parents[0]

        # pkl_path = dir_path / "trap_field_profile_pkl/2021_trap_profile_mainfield_0T_trap_1A.csv"
        pkl_path = dir_path / "trap_field_profile_pkl/trap_1A.csv"
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

    def find_trap_center(self, rho = 0):
        """finds the z-position of the center of an inverted trap"""
        func = lambda z: self.field_strength(rho, z)
        z_side_coil = 4.3e-2
        z_center = fmin(func, z_side_coil, xtol=1e-12)[0]
        return z_center

    def trap_width_calc(self):
        """
        Calculates the trap width of the object trap_profile.
        """

        field_func = self.field_strength
        
        func = lambda z: -1 * field_func(0, z)
        
        if self.inverted:
            trap_zmax = 5.5e-2  # (m)
            print("Inverted trap width: (0, {})".format(trap_zmax))
            print("Maximum Field (at z=0): {}".format(-1* func(0)))
            trap_width = (0,trap_zmax)
        else:
            maximum = fmin(func, 0, xtol=1e-12)[0]
            print("Trap width: ({},{})".format(-maximum, maximum))
            print("Maximum Field: {}".format(-1 * func(maximum)))
            trap_width = (-maximum, maximum)

        return trap_width #- self.trap_center
