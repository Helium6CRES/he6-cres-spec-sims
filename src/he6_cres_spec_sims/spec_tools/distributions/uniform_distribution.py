from base_distribution import BaseDistribution

class UniformDistribution(BaseDistribution):
    """ Generator for a uniform probability distribution
    """
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def generate(self, size=None):
        return self.rng.uniform(self.low, self.high, size)
