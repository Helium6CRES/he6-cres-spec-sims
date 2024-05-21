from base_distribution import BaseDistribution

class NormalDistribution(BaseDistribution):
    """ Generator for a normal (Gaussian) probability distribution
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def generate(self, size=None):
        return self.rng.normal(self.mu, self.sigma, size)
