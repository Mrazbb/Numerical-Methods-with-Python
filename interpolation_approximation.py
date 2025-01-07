"""
This module defines classes for interpolation and approximation:
- InterpolationApproximation (base class)
- LagrangePolynomial
- Approximation
- LinearInterpolationSpline

Each class extends the Algorithm base class and implements its own
method of 'compute' for polynomial interpolation, linear approximation,
and spline interpolation.
"""

import matplotlib.pyplot as plt
import math
import gmpy2
import sympy
import numpy as np
from gmpy2 import mpz, mpfr
from abc import ABC, abstractmethod
from algorithm import Algorithm

__all__ = [
    'LagrangePolynomial',
    'Approximation',
    'LinearInterpolationSpline'
]

class InterpolationApproximation(Algorithm):
    """
    Base class providing shared functionality for interpolation and approximation.

    Attributes:
        coords (np.array): A NumPy array of (x, y) coordinate pairs.
        result (mpfr): A stored result (if needed by subclasses).
        x_0 (mpfr): A placeholder for x_0 (optionally used by subclasses).
        x_1 (mpfr): A placeholder for x_1 (optionally used by subclasses).
    """

    def __init__(self, coords):
        """
        Initialize with the given coordinates.

        Args:
            coords (list of tuples): Each tuple is (x, y) representing a data point.
        """
        super().__init__(10)
        coords.sort(key=lambda x: x[0])
        self.coords = np.array(coords)
        self.result = self.x_0 = self.x_1 = mpfr('0')

    def print_result(self):
        """
        Prints out the result in some format.
        Subclasses can override this method for detailed output.
        """
        pass


class LagrangePolynomial(InterpolationApproximation):
    """
    Computes a polynomial using the Lagrange interpolation formula.
    """

    def _phi(self, x, index, coords):
        """
        Private helper function to compute the basis polynomial phi_i(x).

        Args:
            x (symbolic expression or numeric): The evaluation point.
            index (int): Index of the data point used for the basis polynomial.
            coords (array-like): The entire set of data points.

        Returns:
            float or symbolic: The value of the i-th basis polynomial at x.
        """
        numerator = math.prod([
            x - coords[j][0]
            for j in range(len(coords)) if j != index
        ])
        denominator = math.prod([
            coords[index][0] - coords[j][0]
            for j in range(len(coords)) if j != index
        ])
        return numerator / denominator

    def compute(self):
        """
        Computes the Lagrange polynomial for the given set of data points.

        The function is stored in:
            self.function_str (sympy expression as a string)
            self.function (a Python lambda)
        """
        x = sympy.symbols('x', positive=True)
        sum_expr = 0
        for i in range(len(self.coords)):
            sum_expr += self.coords[i][1] * self._phi(x, i, self.coords)
        self.function_str = sympy.simplify(sum_expr)
        # Create a lambda function using sympy's lambdify
        self.function = sympy.lambdify(sympy.symbols('x'), self.function_str, 'numpy')


class Approximation(InterpolationApproximation):
    """
    Computes a linear approximation (best fit line) for the given set of points
    using the method of least squares.
    """

    def compute(self):
        """
        Performs least squares approximation: y = c1 + c2 * x

        The resulting line is stored in:
            self.function_str (string format: "y = {c1} + {c2}x")
            self.function (a Python lambda)
        """
        a = len(self.coords)                    # number of points
        b = sum(self.coords[:, 0])             # sum of x-coordinates
        c = sum(self.coords[:, 1])             # sum of y-coordinates
        d = sum(self.coords[:, 0] ** 2)        # sum of x^2
        e = sum(self.coords[:, 0] * self.coords[:, 1])  # sum of x*y

        # Determinant of the matrix
        matrix_determinant = np.linalg.det(np.array([[a, b], [b, d]]))
        a1 = np.linalg.det(np.array([[c, b], [e, d]]))
        a2 = np.linalg.det(np.array([[a, c], [b, e]]))

        c1 = a1 / matrix_determinant  # intercept
        c2 = a2 / matrix_determinant  # slope

        self.function = lambda x_val: c1 + c2 * x_val
        self.function_str = "y = {:.4f} + {:.4f}x".format(c1, c2)

        # TODO: Compute mean square error or other metrics if needed.


class LinearInterpolationSpline(InterpolationApproximation):
    """
    Performs linear interpolation piecewise between the provided points.
    """

    def compute():
        """
        Placeholder for future implementation of the linear interpolation spline.
        """
        pass

    def print_result(self):
        """
        Prints a piecewise linear polynomial for each interval between consecutive points.
        For an interval [x0, x1], the interpolation is:
            y(t) = y0 * (1 - t) + y1 * t,  where t = (x - x0) / (x1 - x0).
        """
        x = sympy.symbols('x', positive=True)
        for point_start, point_end in zip(self.coords, self.coords[1:]):
            t = (x - point_start[0]) / (point_end[0] - point_start[0])
            result_expr = point_start[1] * (1 - t) + point_end[1] * t
            print(
                sympy.simplify(result_expr),
                "for x in <",
                point_start[0],
                ",",
                point_end[0],
                ">"
            ) 