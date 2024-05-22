from .base_distribution import BaseDistribution
from numpy import ones

class DiracDistribution(BaseDistribution):
    """ Generator for a dirac (fixed-value) probability distribution
    """
    def __init__(self, value=0.):
        # Set default values
        self.value = value

    def set_parameters(self, yaml_block):
        # if present, assign from config file
        if "value" in yaml_block:
            self.value = yaml_block["value"]

    def generate(self):
        return self.value

    def generate(self, size):
        return self.value * ones(size)
