import pathlib

import numpy as np
from scipy.interpolate import RectBivariateSpline, CubicSpline
from scipy.optimize import fmin

class TrapFieldProfile:
    def __init__(self, main_field, trap_current):

        # TODO: May want to protect these variables with underscores?
        # TODO: Would be nice to have an attribute be the relative trap depth.
        # TODO: Add in trap radius as an attribute?

        self.trap_current = trap_current
        self.inverted = trap_current*main_field < 0
        self.main_field = main_field

        # there is probably a more elegant way to do this
        test_asym = False
        
        if test_asym:
            self.field_strength = self.test_trap_asym
            self.trap_width = (-0.055,0.055)
        else:
            self.field_strength = self.initialize_field_strength_interp()
            # a = [z, x, y, xz, yz, z^2, xy, x^2-y^2]
            self.trap_width = self.trap_width_calc()

        self.trap_center = self.initialize_trap_center_interp()

        # min and max fields of trap with rho dependence
        self.Bmin = lambda rho = 0: self.field_strength(rho, self.trap_center(rho)) 
        self.Bmax = lambda rho = 0: self.field_strength(rho,self.trap_width[1]) 

        # TODO: Actually test to be sure it is a trap.
        self.is_trap = True

        # self.relative_depth = (main_field - self.Bmin()) / main_field # unused?

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

    def initialize_trap_center_interp(self):
        """Returns function f(rho) which returns the trap center as a function of rho"""
        waveguide_radius = 0.578e-2 # (m)
        grid_points = 20 # (m)
        rho_array = np.linspace(0, waveguide_radius, grid_points)
        trap_center_array = np.zeros(np.size(rho_array))
        
        for i in range(len(rho_array)):
            trap_center_array[i] = self.find_trap_center(rho_array[i]) 
        
        # is extrapolate ever useful? 
        trap_center_interp = CubicSpline(rho_array, trap_center_array, extrapolate=False)
        
        return trap_center_interp 

    def find_trap_center(self, rho = 0):
        """finds the z-position of the center of an inverted trap by minimizing"""

        if not self.inverted:
            return 0.

        waveguide_radius = 0.578e-2 # (m)

        func = lambda z: self.field_strength(rho, z)
        z_side_coil = 4.3e-2 # hardcoded for now
        z_center = fmin(func, z_side_coil, xtol=1e-12, disp=False)[0]
        # print(f"Trap center at rho={rho}: {z_center}")
        return z_center

    def trap_width_calc(self):
        """
        Calculates the trap width of the object trap_profile at rho = 0.
        """

        """ 
        TODO: rework this function to work more generally. My idea 
        is to have it search left and right from trap_center up to +/- 
        trap_zmax, not totally sure how to do that 
        """

        func = lambda z: -1 * self.field_strength(0, z)

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

        return trap_width

    def shim_field(self, rho, phi, z, a):
        """
        Returns field from shimming coils to be added to field_strength
        a = [a1, a2, a3, a4, a5, a6, a7, a8]
        Bshim = a1*z + a2*x+ a3*y + a4*xz + a5* yz + a6*z^2 + a7*xy + a8*(x^2+y^2)
        """
        # TODO: accept [a1, ..., a8] if passed 8 coefficients and insert a0 = main_field at start if passed 8

        if len(a) != 8:
            print(f"ERROR: shim_field requires 8 coefficients, but was passed {len(a)}.")
            return 0

        # easier indexing to match full field expansion
        a.insert(0, self.main_field)

        x = rho*np.cos(phi / RAD_TO_DEG)
        y = rho*np.sin(phi / RAD_TO_DEG)

        Bshim = a[1]*z + a[2]*x + a[3]*y + a[4]*x*z + a[5]*y*z + a[6]*z**2 + a[7]*x*y + a[8]*(x**2-y**2)

        return Bshim
     
    def test_trap_asym(self, rho, z):
        """
        Returns field strength for ideal asymmetrical harmonic trap
        """
        
        a = 10
        b = 6
        test_center = 2.5e-2 
        
        return ((z  > test_center) * ((z-test_center)**2/a**2 + self.main_field) + 
                (z <= test_center) * ((z-test_center)**2/b**2 + self.main_field) - 1e-4)

