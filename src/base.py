import numpy as np 
from abc import ABC, abstractmethod

class VectorQuantizer(ABC):
    def __init__(self, dim: int, bit_width: int):
        self.dim = dim
        self.bit_width = bit_width

    @abstractmethod
    def quantize(self, X: np.ndarray):
        pass

    @abstractmethod
    def dequantize(self, codes):
        pass