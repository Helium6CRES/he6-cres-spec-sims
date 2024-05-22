from abc import ABC, abstractmethod

class BaseDistribution(ABC):
    """ Abstract Base Class for random distribution generator. Child classes need to have generate()
    """
    def set_random_engine(self, rng):
       self.rng = rng 

    @abstractmethod
    def generate(self, size):
        raise NotImplementedError("generate() must be implemented")