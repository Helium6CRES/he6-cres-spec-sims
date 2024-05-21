from base_distribution import BaseDistribution

class ExponentialDistribution(BaseDistribution):
    """ Generator for a exponential probability distribution as e^{-t/tau}
    """
    def __init__(self, tau):
        self.tau = tau

    def generate(self, size=None):
        return self.rng.exponential(self.tau, size)
