"""
This module provides classes to solve systems of linear equations using various iterative methods.
It extends a base "Algorithm" (an abstract class) and provides:
1. A class (LinearSolver) for setting up a matrix A and vector b.
2. Different solver subclasses (JacobiMethod, GaussSeidelMethod) that implement compute().
"""

import itertools
import matplotlib.pyplot as plt
import math
import gmpy2
import sympy
import numpy as np
from gmpy2 import mpz, mpfr
from abc import ABC, abstractmethod

# This import assumes the base abstract class is found under the name "Algorithm" 
# within a file named "algorithm.py". Adjust as necessary.
from algorithm import Algorithm


__all__ = [
    'JacobiMethod',
    'GaussSeidelMethod'
]


class LinearSolver(Algorithm):
    """
    A base class that represents the structure for solving a system of linear equations.

    Inherits from:
        Algorithm (ABC): The abstract base class providing precision settings and methods to track accuracy.

    Attributes:
        A (list[list[mpfr]]): The coefficient matrix of the linear system.
        b (list[mpfr]): The right-hand side vector of the linear system.
        result (list[mpfr]): The computed result vector.
        iteration_count (int): Tracks how many iterations have been carried out.
        exact_solution (bool): Flag indicating if we consider the solution exact.
        estimated_iteration_result (int): Placeholder for storing an estimate, if used.
    """

    def __init__(self, A, b, decimal_places):
        """
        Initialize the linear solver with the given matrix A, vector b, and precision in decimal places.
        
        Args:
            A (list[list[str]]): 2D list of strings, each representing a matrix element.
            b (list[str]): 1D list of strings, representing the right-hand side vector.
            decimal_places (int): Number of decimal places for high-precision arithmetic.
        """
        super().__init__(decimal_places)
        # Convert input matrix A and vector b into mpfr for high-precision calculations
        self.A = []
        for row_in_A in A:
            row_converted = [mpfr(str(val)) for val in row_in_A]
            self.A.append(row_converted)

        self.b = [mpfr(str(val)) for val in b]

        # Basic placeholders and defaults
        self.exact_solution = False
        self.estimated_iteration_result = 0
        self.result = [mpfr('0') for _ in range(len(b))]

        # Optional permutation-based approach for ensuring convergence. 
        # Currently, we only check convergence and assert if it fails.
        # If needed, one can attempt permutations or transformations.
        assert self.is_convergent(self.A), "Please change matrix A, it does not converge."

    def print_solution(self):
        """
        Print the solution vector and iteration count in a formatted manner.
        """
        for idx in range(len(self.result)):
            print(f" x{idx}: {self.result[idx]:.{self.e}f}")
        # Print an indication of the error margin
        print(f" +- {self.epsilon:.2E}")
        print("Number of iterations:", self.iteration_count)

    def row_norm(self, matrix):
        """
        Compute the maximum row norm if 'matrix' is 2D, or the L-infinity norm if 1D.

        Args:
            matrix (list[list[mpfr]] or list[mpfr]): The matrix or vector.

        Returns:
            mpfr or float: The row norm (max sum of absolute row entries in 2D, or max absolute value in 1D).
        """
        # If it's effectively a vector
        if isinstance(matrix[0], mpfr):
            return max(abs(x) for x in matrix)
        else:
            # It's a 2D matrix
            max_row_val = sum(abs(x) for x in matrix[0])
            for i in range(1, len(matrix)):
                row_sum = sum(abs(x) for x in matrix[i])
                max_row_val = max(max_row_val, row_sum)
            return max_row_val

    def column_norm(self, matrix):
        """
        Compute the maximum column norm if 'matrix' is 2D, or the L-1 norm if 1D.

        Args:
            matrix (list[list[mpfr]] or list[mpfr]): The matrix or vector.

        Returns:
            mpfr or float: The column norm (max sum of absolute column entries in 2D, or sum of absolute values in 1D).
        """
        # If it's effectively a vector
        if isinstance(matrix[0], mpfr):
            return sum(abs(x) for x in matrix)
        else:
            # Transpose the matrix to work column by column
            transposed = list(zip(*matrix))
            max_col_val = sum(abs(x) for x in transposed[0])
            for i in range(1, len(transposed)):
                col_sum = sum(abs(x) for x in transposed[i])
                max_col_val = max(max_col_val, col_sum)
            return max_col_val

    def frobenius_norm(self, matrix):
        """
        Compute the Frobenius norm of the given 2D matrix.

        Args:
            matrix (list[list[mpfr]]): The matrix.

        Returns:
            float: The Frobenius norm of the matrix.
        """
        total = mpfr('0')
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                total += matrix[i][j] ** 2
        return math.sqrt(total)

    def is_convergent(self, matrix):
        """
        Check if the matrix is diagonally dominant, which is a basic 
        requirement for certain iterative methods to converge (Jacobi, Gauss-Seidel, etc.).

        Args:
            matrix (list[list[mpfr]]): The matrix to check.

        Returns:
            bool: True if the matrix is convergent (diagonally dominant), else False.
        """
        for i in range(len(matrix)):
            # Sum of off-diagonal elements
            non_diagonal_sum = sum(abs(matrix[i][j]) for j in range(len(matrix[i])) if j != i)
            if non_diagonal_sum > abs(matrix[i][i]):
                return False
        return True


class JacobiMethod(LinearSolver):
    """
    Implementation of the Jacobi iterative method for solving a system of linear equations.

    Inherits from:
        LinearSolver
    """

    def compute(self):
        """
        Execute the Jacobi method until the desired precision (epsilon) is reached.
        """
        x_old = np.array([mpfr('0') for _ in range(len(self.b))])
        x_new = np.copy(x_old)
        current_epsilon = mpfr('0')

        while True:
            self.iteration_count += 1
            for i in range(len(self.A)):
                # Summation of A[i][j]*x_old[j] for j != i
                sum_off_diag = sum(self.A[i][j] * x_old[j] for j in range(len(self.A[0])) if j != i)
                x_new[i] = (self.b[i] - sum_off_diag) / self.A[i][i]

            # Measure the difference between x_new and x_old
            difference = (x_new - x_old)
            current_epsilon = abs(self.row_norm(difference))

            # Log the precision in decimal digits
            precision_log = float(abs(gmpy2.log(current_epsilon) / gmpy2.log(10)))
            # Store the current precision into self.accuracy
            self.accuracy = np.append(self.accuracy, [precision_log])

            if current_epsilon < self.epsilon:
                break

            x_old = np.copy(x_new)

        self.result = np.array(x_new)


class GaussSeidelMethod(LinearSolver):
    """
    Implementation of the Gauss-Seidel iterative method for solving a system of linear equations.

    Inherits from:
        LinearSolver
    """

    def compute(self):
        """
        Execute the Gauss-Seidel method until the desired precision (epsilon) is reached.
        """
        x_current = np.array([mpfr('0') for _ in range(len(self.b))])
        x_previous = np.copy(x_current)
        current_epsilon = mpfr('0')

        while True:
            self.iteration_count += 1

            # Update each variable in-place, using already updated components
            for i in range(len(self.A)):
                sum_off_diag = sum(self.A[i][j] * x_current[j] for j in range(len(self.A[0])) if j != i)
                x_current[i] = (self.b[i] - sum_off_diag) / self.A[i][i]

            difference = (x_current - x_previous)
            current_epsilon = abs(self.row_norm(difference))

            precision_log = float(abs(gmpy2.log(current_epsilon) / gmpy2.log(10)))
            self.accuracy = np.append(self.accuracy, [precision_log])

            if current_epsilon < self.epsilon:
                break

            x_previous = np.copy(x_current)

        self.result = np.array(x_current)