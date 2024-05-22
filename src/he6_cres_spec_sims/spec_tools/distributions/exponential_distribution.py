from .base_distribution import BaseDistribution

class ExponentialDistribution(BaseDistribution):
    """ Generator for a exponential probability distribution as e^{-t/tau}
    """
    def __init__(self, tau=1.):
        # Set default values
        self.tau = tau

    def set_parameters(self, yaml_block):
        # if present, assign from config file
        if "tau" in yaml_block:
            self.tau = yaml_block["tau"]

    def generate(self, size=None):
        return self.rng.exponential(self.tau, size)
