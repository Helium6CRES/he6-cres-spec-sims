from .base_distribution import BaseDistribution

class UniformDistribution(BaseDistribution):
    """ Generator for a uniform probability distribution
    """
    def __init__(self, low=0., high=1.):
        self.low = low
        self.high = high

    def set_parameters(self, yaml_block):
        # if present, assign from config file
        if "low" in yaml_block:
            self.low = yaml_block["low"]
        if "high" in yaml_block:
            self.high = yaml_block["high"]

    def generate(self, size=None):
        return self.rng.uniform(self.low, self.high, size)
