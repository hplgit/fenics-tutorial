from dolfin import *
import numpy

mesh2D = UnitSquare(5, 4)
V2D = FunctionSpace(mesh2D, 'Lagrange', 1)

mesh1D = UnitInterval(4)
V1D = FunctionSpace(mesh1D, 'Lagrange', 1)

v_formula = Expression('4*x[0]*(1 - x[0])')

v1D = project(v_formula, V1D)
print 'v1D:', v1D.vector().array()

# How to let v be a function of y and populate a Function u in V2D?
# (channel flow)

class v_interpolate2D(Expression):
    def eval(self, value, x):
        assert len(x) == 2
        point = [x[1]]
        value[0] = self.v(point)

v2D_extension = v_interpolate2D()
v2D_extension.v = v1D

u = project(v2D_extension, V2D)
print 'u:', u.vector().array()

