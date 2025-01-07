"""
This module defines a base abstract class for numerical algorithms.

It provides:
1. A fixed precision setup for computations using GMPY2.
2. Methods for tracking the accuracy of iterative processes.
3. A basic infrastructure to be extended by actual algorithm implementations.
"""

import matplotlib.pyplot as plt
import math
import gmpy2
import sympy
import numpy as np
from gmpy2 import mpz, mpfr
from abc import ABC, abstractmethod

# Example constants and function for demonstration.
# These can be overridden or replaced in actual usage.
example_number = mpfr('30')
example_x_0 = example_x_1 = mpfr('5')
intervals = [[mpfr('0.7'), mpfr('0.8')]]
function = lambda x: x - gmpy2.cos(x)

__all__ = [
    'Algorithm'
]

class Algorithm(ABC):
    """
    Base abstract class for numerical algorithms.

    Attributes:
        e (int): Number of decimal places to be used for calculations.
        precision (mpz): The actual precision used by GMPY2,
                         based on the user-specified decimal places.
        epsilon (mpfr): Smallest increment, derived from the precision,
                        used as a stopping criterion in algorithms.
        accuracy (np.array): Stores accuracy metrics (or residuals) over iterations.
        iteration_count (int): Tracks the number of iterations performed.
    """

    def __init__(self, e):
        """
        Initialize the algorithm with the specified number of decimal places (e).

        Args:
            e (int): Number of decimal places for precision.
        """
        self.e = e
        self.precision = pow(mpz(10), mpz(self.e))  # 10^e
        # Ensure that GMPY2 can handle the requested precision:
        assert gmpy2.get_max_precision() > mpz(gmpy2.log(self.precision) / gmpy2.log(2)), \
            "Requested precision is too large for GMPY2."
        # Set actual context precision in bits (approx. 10 times the number of decimal digits):
        gmpy2.get_context().precision = int(mpz(gmpy2.log(self.precision) / gmpy2.log(2)) * 10)

        self.epsilon = mpfr('1') / self.precision
        self.accuracy = np.array([])
        self.iteration_count = 0

    def plot_accuracy(self):
        """
        Plot the accuracy or residuals over the number of iterations.
        """
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, self.accuracy.size + 1), self.accuracy, linewidth=2.0)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy (number of decimal places)')
        plt.xticks(np.arange(1, self.accuracy.size + 1))
        plt.grid(True)
        plt.show()

    @abstractmethod
    def compute(self):
        """
        Abstract method to perform the numerical algorithm.
        Must be implemented by subclasses.
        """
        pass 