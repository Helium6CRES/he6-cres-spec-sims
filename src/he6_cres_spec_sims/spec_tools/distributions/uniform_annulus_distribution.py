from .base_distribution import BaseDistribution
from numpy import sqrt

class UniformAnnulusDistribution(BaseDistribution):
    """ Generator for the radial distances over a 2D annulus (disk with a hole in center), given uniform spatial distribution
        Derived from inverse transform sampling 
        See also: https://stackoverflow.com/a/13065255
    """
    def __init__(self, rho_min=0, rho_max=1):
        self.rho_min = rho_min
        self.rho_max = rho_max

    def set_parameters(self, yaml_block):
        # if present, assign from config file
        if "rho_min" in yaml_block:
            self.rho_min = yaml_block["rho_min"]
        if "rho_max" in yaml_block:
            self.rho_max = yaml_block["rho_max"]

        # Define to avoid having to recalculate each generate() call)
        self.factors = [rho_min**2, rho_max**2 - rho_min**2]

    def generate(self, size=None):
        return sqrt(self.factors[0] + self.rng.uniform(size=size) * self.factors[1])
