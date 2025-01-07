import matplotlib.pyplot as plt
import math, gmpy2, sympy
import numpy as np
from gmpy2 import mpz, mpfr
from abc import ABC, abstractmethod
import gmpy2
from algorithm import *

from sympy import *

__all__ = [
    'Bisection',
    'RegulaFalsi',
    'Newton'
]

class RootEquation(Algorithm):
    """
    Base class for root-finding methods.

    Attributes:
        function (callable): The function for which we are finding the root.
        intervals (list): List of intervals to search for the root.
        result (mpfr): The found root (or the most recent root approximation).
        x0 (mpfr): Helper variable used by specific methods (initial point).
        x1 (mpfr): Helper variable used by specific methods (next point).
        exact_result (bool): Indicates if the exact (precise) result was found (function value is zero).
        estimated_iteration_count (int): Estimated number of iterations (if computed).
    """

    def __init__(self, function, precision_digits, intervals):
        """
        Initialize the root equation solver with the function, precision, and intervals.

        Args:
            function (callable): The function whose root will be found.
            precision_digits (int): Number of decimal places for precision.
            intervals (list of lists): Each sub-list has two elements [start, end] representing one interval.
        """
        super().__init__(precision_digits)
        self.function = function
        self.intervals = []
        for interval in intervals:
            self.intervals.append([mpfr(str(interval[0])), mpfr(str(interval[1]))])
        self.result = self.x0 = self.x1 = mpfr('0')
        self.exact_result = False
        self.estimated_iteration_count = 0

    def print_result(self):
        """
        Prints out the final result of the computation, along with precision
        information and number of iterations.
        """
        if self.exact_result:
            print(f"Exact result: {self.result:.{self.e}f}")
        else:
            print(f"Result: {self.result:.{self.e}f}\n +- {self.epsilon:.2E}")
        if self.estimated_iteration_count != 0:
            print("Estimated number of iterations:", self.estimated_iteration_count)
        print("Number of iterations:", self.iteration_count)


class Bisection(RootEquation):
    """
    Implements the Bisection method for root finding.
    """

    def estimate_iterations(self, a, b, epsilon):
        """
        Estimates the number of iterations required for the Bisection method.

        Args:
            a (mpfr): Start of the interval.
            b (mpfr): End of the interval.
            epsilon (mpfr): Desired precision.

        Returns:
            int: Estimated number of iterations.
        """
        return math.ceil(gmpy2.log(abs(a - b) / epsilon) / gmpy2.log(2))

    def compute(self):
        """
        Performs the Bisection method on each interval until the root is found or intervals are exhausted.
        """
        interval_idx = 0
        finished = False

        self.x1 = mpfr('0')

        while not finished and interval_idx < len(self.intervals):
            a, b = self.intervals[interval_idx]

            # Check if the function changes sign on [a, b]
            if self.function(a) * self.function(b) < 0:

                while True:
                    self.iteration_count += 1

                    self.x1 = (a + b) / 2

                    if self.function(a) * self.function(self.x1) < 0:
                        b = self.x1
                    else:
                        a = self.x1

                    accuracy_log = float(abs(gmpy2.log(abs(b - a)) / gmpy2.log(10)))
                    self.accuracy = np.append(self.accuracy, [accuracy_log])

                    # Check if we found the exact root
                    if self.function(self.x1) == 0:
                        finished = True
                        self.exact_result = True
                        break

                    # Check if the desired precision is reached
                    if abs(b - a) < self.epsilon:
                        finished = True
                        self.estimated_iteration_count = self.estimate_iterations(
                            self.intervals[interval_idx][0],
                            self.intervals[interval_idx][1],
                            self.epsilon
                        )
                        break
            else:
                interval_idx += 1
        self.result = self.x1


class RegulaFalsi(RootEquation):
    """
    Implements the Regula Falsi (False Position) method for root finding.
    """

    def estimate_iterations(self, a, b, epsilon):
        """
        Estimates the number of iterations required for Regula Falsi.

        Args:
            a (mpfr): Start of the interval.
            b (mpfr): End of the interval.
            epsilon (mpfr): Desired precision.

        Returns:
            int: Estimated number of iterations (using a rough bisection-based approach).
        """
        return math.ceil(math.log(abs(a - b) / epsilon) / math.log(2))

    def compute(self):
        """
        Performs the Regula Falsi method on each interval until the root is found or intervals are exhausted.
        """
        interval_idx = 0
        finished = False

        self.x1 = mpfr('0')

        while not finished and interval_idx < len(self.intervals):
            a, b = self.intervals[interval_idx]

            # Check if the function changes sign on [a, b]
            if self.function(a) * self.function(b) < 0:

                while True:
                    self.iteration_count += 1

                    # Regula Falsi formula
                    self.x1 = a - (float(b - a) / (self.function(b) - self.function(a)) * self.function(a))

                    if self.function(a) * self.function(self.x1) < 0:
                        b = self.x1
                    else:
                        a = self.x1

                    accuracy_log = float(abs(gmpy2.log(abs(self.x0 - self.x1)) / gmpy2.log(10)))
                    self.accuracy = np.append(self.accuracy, [accuracy_log])

                    # Check if we found the exact root
                    if self.function(self.x1) == 0:
                        finished = True
                        self.exact_result = True
                        break

                    # Check if the desired precision is reached
                    if abs(self.x0 - self.x1) < self.epsilon:
                        finished = True
                        break

                    self.x0 = self.x1
            else:
                interval_idx += 1
        self.result = self.x1


class Newton(RootEquation):
    """
    Implements the Newton-Raphson method for root finding.
    """

    def estimate_iterations(self, a, b, epsilon):
        """
        Estimates the number of iterations required for the Newton method (using a bisection-based formula).

        Args:
            a (mpfr): Start of the interval.
            b (mpfr): End of the interval.
            epsilon (mpfr): Desired precision.

        Returns:
            int: Estimated number of iterations.
        """
        return math.ceil(math.log(abs(a - b) / epsilon) / math.log(2))

    def compute(self):
        """
        Performs the Newton-Raphson method on each interval until the root is found or intervals are exhausted.
        """
        x = sympy.symbols('x')
        dfun_str = str(self.function(x).diff(x))
        dfun = lambda x: eval(dfun_str)

        dfun2_str = str(dfun(x).diff(x))
        dfun2 = lambda x: eval(dfun2_str)

        interval_idx = 0
        finished = False

        self.x1 = mpfr('0')

        while not finished and interval_idx < len(self.intervals):

            a = self.intervals[interval_idx][0]
            b = self.intervals[interval_idx][1]

            if self.function(a) * self.function(b) < 0:
                # Initial guess x0
                self.x0 = a if self.function(a) * dfun2(a) > 0 else b
                while True:
                    self.x1 = self.x0 - (self.function(self.x0) / dfun(self.x0))

                    self.iteration_count += 1

                    accuracy_log = float(abs(gmpy2.log(abs(self.x0 - self.x1)) / gmpy2.log(10)))
                    self.accuracy = np.append(self.accuracy, [accuracy_log])

                    # Check if we found the exact root
                    if self.function(self.x1) == 0:
                        finished = True
                        self.exact_result = True
                        break

                    # Check if the desired precision is reached
                    if abs(self.x0 - self.x1) < self.epsilon:
                        finished = True
                        break

                    self.x0 = self.x1
            else:
                interval_idx += 1

        self.result = self.x1