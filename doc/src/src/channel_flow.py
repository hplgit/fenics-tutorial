"""
This program shows how to use FEniCS to solve dimension-reduced
differential equations and insert the solutions into higher-dimensional
mehes.

The particular examples concerns the analytical approach to computing
flow of a viscous fluid in a channel. The domain is either 2D or
3D. By assuming a uni-directional velocity field, the Navier-Stokes
equations can be reduced to two differential equations in one variable:

1. A problem for the velocity v along the channel, where v only
depends on the coordinate in the normal direction to the channel walls
(here named y):

   mu*v''(y) = p'(x)

where p is the pressure. If y=0 is the symmetry plane of the
channel, and the channel height is H, we have the boundary conditions

  v'(0) = 0  due to symmetry
  v(H) = 0   due to no-splip condition at the wall

2. A problem for pressure p, where p only varies with the coordinate
x along the channel:

  p''(x) = 0

The boundary conditions are p(0)=p0 and p(L)=0.

We will define two 1D problems for v and for p. Then we will make
a 2D or 3D mesh and insert the 1D solutions such that we get
a complete velocity vector field u and scalar field p defined
on the complete domain for the channel.
"""
from fenics import *

def compute_v(H, mu, p, degree=1):
    nx = 15
    mesh = IntervalMesh(nx, 0, H)
    V = FunctionSpace(mesh, 'P', degree)

    def boundary(x, on_boundary):
        tol = 1E-14
        if on_boundary and abs(x[0]-H) < tol:
            return True

    bc = DirichletBC(V, 0, boundary)

    v = TrialFunction(V)
    v_test = TestFunction(V)
    a = mu*dot(grad(v), grad(v_test))*dx
    L = -grad(p)*v_test*dx  # non-trivial, grad is _x for p
    v = Function(V)
    solve(a == L, v, bc)
    return v

def compute_p(L, p0, degree=1):
    nx = 13
    mesh = IntervalMesh(nx, 0, L)
    V = FunctionSpace(mesh, 'P', degree)

    def boundary_0(x, on_boundary):
        tol = 1E-14
        if on_boundary and abs(x[0]) < tol:
            return True
        if on_boundary and abs(x[0]) < tol or abs(x[0]-H) < tol:
            return True

    tol = 1E-14
    bc0 = DirichletBC(V, Constant(p0), lambda x, onb: onb and abs(x[0]) < tol)
    bcL = DirichletBC(V,  Constant(0), lambda x, onb: onb and abs(x[0]-L) < tol)
    bc = [bc0, bcL]

    p = TrialFunction(V)
    p_test = TestFunction(V)
    a = dot(grad(p), grad(p_test))*dx
    L = Constant(0)*p_test*dx
    p = Function(V)
    solve(a == L, p, bc)
    return p

mu = 1
L = 2
H = 1
p0 = 2
p = compute_p(L, p0)
v = compute_v(H, mu, p)

print 'v:', v.vector().array()
print 'p:', p.vector().array()

class p_interpolator(Expression):
    def eval(self, p, x):
        # x is some point in some mesh
        # p(x) is the scalar pressure value
        # self.p1D is 1D pressure p(x)
        point = [x[0]]
        p[0] = self.p1D(point)

p_expression = p_interpolator()
p_expression.p1D = p  # could also initialize this in constructor

class v_interpolator(Expression):
    def __init__(self, v1D, num_space_dim):
        self.v1D = v1D
        self.nsd = num_space_dim
        print 'Hi - in __init__', self.nsd, self.v1D


    def eval(self, u, x):
        # x is some point in some mesh
        # u(x) is the velocity vector
        # self.v1D is 1D velocity v(y) or v(z) normal to the flow dir.

        if len(x) == 2:
            point = [x[1]]
            u[1] = 0
        elif len(x) == 3:
            point = [x[2]]
            u[1] = u[2] = 0
        u[0] = self.v1D(point)

        print 'x=%s nsd=%d v1D-dim=%d' % (x, self.nsd, self.v1D.function_space().dim())

    def value_shape(self):
        """Return shape (tuple) of u (here vector of self.nsd dimensions)."""
        #return (self.nsd,)  # doesn't work!
        return (2,)  # length must be hardcoded

# Note that arguments to Expression constructor must use keyword
# arguments, and __init__ must not call Expression.__init__
# No: seems to work fine without keyword arguments to constructor:
u_expression = v_interpolator(v, 2)
#u_expression = v_interpolator(v1D=v, num_space_dim=2)
#u_expression.v1D = v

n_flow = 3
n_perp = 3
degree = 1
mesh2D = RectangleMesh(0, 0, L, H, n_flow, n_perp)
Vu2D = VectorFunctionSpace(mesh2D, 'P', degree)
Vp2D = FunctionSpace(mesh2D, 'P', degree)

p = interpolate(p_expression, Vp2D)
print '2D p:', p.vector().array()
plot(mesh2D)
plot(p)
u = interpolate(u_expression, Vu2D)
plot(u)
interactive()
import sys
sys.exit(0)

W = 3
nz = 3
mesh3D = BoxMesh(0, 0, 0, L, W, H, n_flow, n_perp, nz)
Vu3D = VectorFunctionSpace(mesh3D, 'P', degree)
Vp3D = FunctionSpace(mesh3D, 'P', degree)
u_expression = v_interpolator(v, 3)

p = interpolate(p_expression, Vp2D)
print '3D p:', p.vector().array()
plot(mesh3D)
plot(p)
u = interpolate(u_expression, Vu2D)
plot(u)


