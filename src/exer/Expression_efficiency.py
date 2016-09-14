from __future__ import print_function
from fenics import *
import time

def make_sine_Function(N, method):
    """Fill a Function with sin(x*y*z) values."""
    mesh = UnitCubeMesh(N, N, N)
    V = FunctionSpace(mesh, 'Lagrange', 2)

    if method.startswith('Python'):
        if method.endswith('fenics.sin'):
            # Need sin as local variable in this function
            from fenics import sin
        elif method.endswith('math.sin'):
            from math import sin
        elif method.endswith('numpy.sin'):
            from numpy import sin
        elif method.endswith('sympy.sin'):
            from sympy import sin
        else:
            raise NotImplementedError('method=%s' % method)
        print('sin:', sin, type(sin))

        class SineXYZ(Expression):
            def __init__(self, a, b):
                self.a, self.b = a, b

            def eval(self, value, x):
                value[0] = self.a*sin(self.b*x[0]*x[1]*x[2])

        expr = SineXYZ(a=1, b=2)

    elif method == 'C++':
        expr = Expression('a*sin(b*x[0]*x[1]*x[2])', a=1, b=2)

    t0 = time.clock()
    u = interpolate(expr, V)
    t1 = time.clock()
    return u, t1-t0

def main(N):
    u, cpu_py_fenics  = make_sine_Function(N, 'Python-fenics.sin')
    u, cpu_py_math    = make_sine_Function(N, 'Python-math.sin')
    u, cpu_py_numpy   = make_sine_Function(N, 'Python-numpy.sin')
    u, cpu_py_sympy   = make_sine_Function(N, 'Python-sympy.sin')
    u, cpu_cpp = make_sine_Function(N, 'C++')
    print("""DOFs: %d
Python:
fenics.sin: %.2f
math.sin:   %.2f
numpy.sin:  %.2f
sympy.sin:  %.2f
C++:        %.2f
Speed-up:   math: %.2f  sympy: %.2f""" %
          (u.function_space().dim(),
           cpu_py_fenics, cpu_py_math,
           cpu_py_numpy, cpu_py_sympy,
           cpu_cpp,
           cpu_py_math/float(cpu_cpp),
           cpu_py_sympy/float(cpu_cpp)))

def profile():
    import cProfile
    prof = cProfile.Profile()
    prof.runcall(main)
    prof.dump_stats("tmp.profile")
    # http://docs.python.org/2/library/profile.html

main(20)
#profile()
