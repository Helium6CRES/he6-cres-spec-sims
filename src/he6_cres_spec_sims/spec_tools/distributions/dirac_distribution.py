from base_distribution import BaseDistribution

class DiracDistribution(BaseDistribution):
    """ Generator for a dirac (fixed-value) probability distribution
    """
    def __init__(self, value)
        self.value = value

    def generate(self)
        return self.value

    def generate(self, size)
        return self.value * np.ones(size)
