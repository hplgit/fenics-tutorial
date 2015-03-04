"""Rough code from Mikael. Develop this to a general class
that can take a form (func or class) and run Newton or Picard
on it. A first step toward pdesys. The nonline class has all
necessary parameters in a parameters dict.
"""


import sys
from dolfin import *
mesh=UnitSquare(10, 10)
V = FunctionSpace(mesh, 'CG', 1)
u = TrialFunction(V)
v = TestFunction(V)
u_ = Function(V)
bc = DirichletBC(V, Constant(0.), DomainBoundary())
x = u_.vector()
bc.apply(x)
omega = 1.
error = 1.

method = sys.argv[-1]

def form(u, v, u_, **kwargs):
    return (1.-u_)*inner(grad(u), grad(v))*dx + Constant(1.)*v*dx
    
if method == 'Newton':
    u_old = u
    u = u_
    F_ = form(**vars())
    u = u_old
    J = derivative(F_, u_, u)
    a, L = J, -F_
    dx = Vector(x)
    i = 0
    while error > 1e-8:
        A = assemble(a)
        b = assemble(L)
        bc.apply(A, b, x)
        solve(A, dx, b)        
        x.axpy(omega, dx)
        error = norm(dx)
        i+=1
        print i, 'error = ', error 

elif method == 'Picard':
    F = form(**vars())
    a, L = lhs(F), rhs(F)
    x_star = Vector(x)
    i = 0
    while error > 1e-8:
        A = assemble(a)
        b = assemble(L)
        bc.apply(A, b)
        x_star[:] = x[:]
        solve(A, x_star, b)       
        # x = (1-omega)*x + omega*x_star = x + omega*(x_star-x):
        x_star.axpy(-1., x); x.axpy(omega, x_star)
        error = norm(x_star)
        i+=1
        print i, 'error = ', error 

plot(u_)
