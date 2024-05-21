from abc import ABC, abstractmethod

class BaseDistribution(ABC):
    """ Abstract Base Class for random distribution generator. Child classes need to have generate()
    """
    #def __init__(self, config):
    #    self.rng = config.rng

    def set_random_engine(self, rng):
       self.rng = rng 

    @abstractmethod
    def generate(self):
        raise NotImplementedError("Scalar generate() must be implemented")

    @abstractmethod
    def generate(self, size):
        raise NotImplementedError("Array generate() must be implemented")
