import matplotlib.pyplot as plt
import math, gmpy2, sympy
import numpy as np
from gmpy2 import mpz, mpfr
from abc import ABC, abstractmethod
import gmpy2
from algorithm import *  # Assuming the base class is named 'Algorithm' in 'algorithm.py'

__all__ = [
    'TrapezoidalMethod',
    'RectangularMethod'
]

class Integral(Algorithm):
    """
    Base class for numerical integration methods.
    Inherits from the generic 'Algorithm' class and extends it for integration tasks.

    Attributes:
        a (mpfr): Lower bound of the integral.
        b (mpfr): Upper bound of the integral.
        function (callable): The function to integrate.
        exact_result_available (bool): If set to True, indicates an exact result is computed.
        estimated_iteration_count (int): An estimate for how many iterations might be needed.
        result (mpfr): Final result of the integration.
    """

    def __init__(self, function, a, b, e):
        """
        Initialize the integral with the given function, bounds [a, b], and precision e.

        Args:
            function (callable): The function to integrate.
            a (str or mpfr): Lower bound of integration.
            b (str or mpfr): Upper bound of integration.
            e (int): Number of decimal places for calculations.
        """
        super().__init__(e)

        self.a = mpfr(a)
        self.b = mpfr(b)
        self.function = function

        self.exact_result_available = False
        self.estimated_iteration_count = 0
        self.result = mpfr('0')

    def print_result(self):
        """
        Print the integration result, along with the precision info and
        number of iterations used.
        """
        if self.exact_result_available:
            print("Exact result: {:.{precision}f}".format(self.result,
                                                          precision=self.e))
        else:
            print("Result: {:.{precision}f}\n ± {epsilon:.2E}".format(self.result,
                                                                      precision=self.e,
                                                                      epsilon=self.epsilon))
        if self.estimated_iteration_count != 0:
            print("Estimated number of iterations: ", self.estimated_iteration_count)
        print("Number of iterations: ", self.iteration_count)


class RectangularMethod(Integral):
    """
    Performs numerical integration using the rectangular (midpoint) rule.
    """

    def compute(self):
        """
        Computes the definite integral using the rectangular (midpoint) rule
        with adaptive refinement until the desired precision is reached.
        """
        m = 1
        prev_value = current_value = mpfr('0')
        finished = False

        while not finished:
            self.iteration_count += 1

            h = (self.b - self.a) / m
            current_value = h * sum(
                [self.function(self.a + i*h + h/2) for i in range(m)]
            )

            # Logarithm of difference (base 10) to track accuracy
            # Avoid error if (current_value - prev_value) is zero or extremely small
            diff = abs(current_value - prev_value)
            if diff > 0:
                precision_log = float(abs(gmpy2.log(diff) / gmpy2.log(10)))
            else:
                precision_log = float(self.e)

            self.accuracy = np.append(self.accuracy, [precision_log])

            if diff < self.epsilon:
                finished = True

            prev_value = current_value
            m *= 2

        self.result = current_value


class TrapezoidalMethod(Integral):
    """
    Performs numerical integration using the trapezoidal rule.
    """

    def compute(self):
        """
        Computes the definite integral using the trapezoidal rule
        with adaptive refinement until the desired precision is reached.
        """
        m = 2
        prev_value = current_value = mpfr('0')
        finished = False

        while not finished:
            self.iteration_count += 1

            h = (self.b - self.a) / m
            # Summation of trapezoids: [f(x_i) + f(x_(i+1))] for i in [0..m-1]
            current_value = h * sum(
                [self.function(self.a + i*h) + self.function(self.a + (i+1)*h)
                 for i in range(m)]
            ) / 2

            diff = abs(current_value - prev_value)
            if diff > 0:
                precision_log = float(abs(gmpy2.log(diff) / gmpy2.log(10)))
            else:
                precision_log = float(self.e)

            self.accuracy = np.append(self.accuracy, [precision_log])

            if diff < self.epsilon:
                finished = True

            prev_value = current_value
            m *= 2

        self.result = current_value