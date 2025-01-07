# Numerical Methods with Python

This repository demonstrates a set of numerical methods implemented in Python, using high-precision arithmetic provided by GMPY2. It covers several mathematical problems including root-finding for equations, numerical integration, solving systems of linear equations, and polynomial interpolation/approximation.

Below is an overview of each file and its purpose, followed by instructions on how to install dependencies and run the examples.

---

## Table of Contents

1. [Project Structure](#project-structure)  
   1.1 [Core Modules](#core-modules)  
   1.2 [Notebooks](#notebooks)  
2. [Usage Instructions](#usage-instructions)  
3. [Dependencies](#dependencies)  

---

## 1. Project Structure

### 1.1 Core Modules

1. <strong>algorithm.py</strong>  
   • Defines the base abstract class <code>Algorithm</code>.  
   • Provides a precision setup (using GMPY2) and basic iteration tracking (accuracy, iteration counts).  
   • Acts as a backbone for all other numerical methods.

2. <strong>rootequation.py</strong>  
   • Implements root-finding classes, inheriting from <code>Algorithm</code>.  
     - <code>RootEquation</code>: Abstract extension specialized for root-finding tasks.  
     - <code>Bisection</code>: Standard Bisection method implementation.  
     - <code>RegulaFalsi</code>: False Position (Regula Falsi) method.  
     - <code>Newton</code>: Newton–Raphson method.  
   • Each class tracks precision, iterations, and provides a <code>print_result()</code> method for convenient output.

3. <strong>integral.py</strong>  
   • Implements numerical integration classes (extending <code>Algorithm</code>):  
     - <code>Integral</code>: Base class for integration tasks.  
     - <code>RectangularMethod</code>: Uses the midpoint (rectangular) rule with adaptive refinement.  
     - <code>TrapezoidalMethod</code>: Uses the trapezoidal rule with adaptive refinement.

4. <strong>interpolation_approximation.py</strong>  
   • Defines classes to solve interpolation and approximation problems (extending <code>Algorithm</code>):  
     - <code>InterpolationApproximation</code>: Base class providing coordinate sorting, result storage, etc.  
     - <code>LagrangePolynomial</code>: Implements Lagrange polynomials for interpolation.  
     - <code>Approximation</code>: Performs a least squares linear fit.  
     - <code>LinearInterpolationSpline</code>: Framework for piecewise-linear interpolation (placeholder example).

5. <strong>linearequations.py</strong>  
   • Extends <code>Algorithm</code> with classes that solve systems of linear equations:  
     - <code>LinearSolver</code>: Base class that reads matrix <code>A</code> and vector <code>b</code>, checks convergence, etc.  
     - <code>JacobiMethod</code> & <code>GaussSeidelMethod</code>: Iterative methods for solving systems of linear equations.

6. <strong>requirements.txt</strong>  
   • Lists all Python dependencies needed to run the project (numpy, gmpy2, sympy, matplotlib, scipy).

---

### 1.2 Notebooks

Several Jupyter notebooks showcase how to use these core modules. Each notebook demonstrates an example numerical problem and how to apply the corresponding method:

1. <strong>rootequation.ipynb</strong>  
   • Uses <code>Bisection</code>, <code>RegulaFalsi</code>, and <code>Newton</code> to find roots of a sample function (e.g., <code>x^(1/2) - 2.5</code>).

2. <strong>integer.ipynb</strong>  
   • Demonstrates numerical integration methods, calling <code>RectangularMethod</code> and <code>TrapezoidalMethod</code>, and showing how <code>Integral</code> classes refine the computation until desired precision is reached.

3. <strong>interpolation_aproximacion.ipynb</strong>  
   • Shows how <code>LagrangePolynomial</code>, <code>Approximation</code>, and <code>LinearInterpolationSpline</code> can be used for interpolating or approximating a set of points.

4. <strong>linearequations.ipynb</strong>  
   • Demonstrates solving linear systems with <code>JacobiMethod</code> and <code>GaussSeidelMethod</code>.

---

## 2. Usage Instructions

1. <strong>Clone the Repository</strong>  
   • Run:  
     <pre><code>git clone https://github.com/your_username/your_repository.git
cd your_repository
</code></pre>

2. <strong>Install Dependencies</strong>  
   • From the project’s root directory:  
     <pre><code>pip install -r requirements.txt
</code></pre>

3. <strong>Run a Notebook</strong>  
   • Launch JupyterLab or Jupyter Notebook:  
     <pre><code>jupyter notebook
</code></pre>
   • Open one of the supplied *.ipynb* files (e.g., <code>rootequation.ipynb</code>) to experiment with the numerical methods.

4. <strong>Using the Python Modules Directly</strong>  
   • You can also import the modules in your own Python scripts or notebooks.  
   • For example:
     <pre><code>from rootequation import Bisection, RegulaFalsi, Newton

my_function = lambda x: x**2 - 4
solver = Bisection(my_function, 10, [[1, 3]])
solver.compute()
solver.print_result()
</code></pre>

---

## 3. Dependencies

All external libraries are listed in <code>requirements.txt</code>:
- <strong>numpy</strong>  
- <strong>matplotlib</strong>  
- <strong>sympy</strong>  
- <strong>gmpy2</strong>  
- <strong>scipy</strong>

These handle numerical arrays (<code>numpy</code>), big-number arithmetic (<code>gmpy2</code>), symbolic operations (<code>sympy</code>), plotting (<code>matplotlib</code>), and advanced computations (<code>scipy</code>).

---
