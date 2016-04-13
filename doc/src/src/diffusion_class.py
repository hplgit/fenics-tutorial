from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

class Solver(object):
    def __init__(self, problem):
        self.problem = problem
        self.mesh, degree = problem.mesh_degree()
        self.V = V = FunctionSpace(self.mesh, 'P', degree)

        # Initial condition
        if hasattr(problem, 'I_project'):
            I_project = getattr(problem, 'I_project')
        else:
            I_project = False
        self.u_1 = project(problem.I(), V) if I_project \
                   else interpolate(problem.I(), V)
        self.u_1.rename('u', 'initial condition')
        problem.user_action(0, self.u_1)

        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        p = problem.p_coeff()
        self.p = p  # store for flux computations
        a_M = u*v*dx
        a_K = dot(p*grad(u), grad(v))*dx
        a_K += sum([r*u*v*ds_
                    for r, s, ds_ in problem.Robin_conditions()])
        self.M = assemble(a_M)
        self.K = assemble(a_K)

        self.f = problem.f_rhs()  # used in step()
        L = Constant(0)*v*dx # Must initialize L if next line has empty lists
        # f is handled in the time loop by interpolation and M matrix
        L -= sum([g*v*ds_
                  for g, ds_ in problem.Neumann_conditions()])
        L -= sum([r*s*v*ds_
                  for r, s, ds_ in problem.Robin_conditions()])
        # Note: updating of Expression objects (t attribute)
        # does not change these a_* and L expressions.
        self.b_surface_int = assemble(L)
        self.u = Function(V)   # the unknown at a new time level
        self.u.rename('u', 'solution')
        self.T = problem.end_time()

    def step(self, t, linear_solver='direct',
             abs_tol=1E-6, rel_tol=1E-5, max_iter=1000):
        """Advance solution one time step."""
        # Find new Dirichlet conditions at this time step
        Dirichlet_cond = self.problem.Dirichlet_conditions()
        if isinstance(Dirichlet_cond, Expression):
            # Just one Expression for Dirichlet conditions on
            # the entire boundary
            self.bcs = [DirichletBC(
                self.V, Dirichlet_cond,
                lambda x, on_boundary: on_boundary)]
        else:
            # Boundary SubDomain markers
            self.bcs = [
                DirichletBC(self.V, value, boundaries, index)
                for value, boundaries, index
                in Dirichlet_cond]

        # Update A
        self.dt = self.problem.time_step(t)
        A = self.M + self.dt*self.K

        # Update right-hand side
        f_k = interpolate(self.f, self.V)
        F_k = f_k.vector()
        b = self.M*self.u_1.vector() + self.dt*self.M*F_k + \
            self.dt*self.b_surface_int
        self.problem.update_boundary_conditions(t)

        # Solve linear system
        [bc.apply(A, b) for bc in self.bcs]
        if linear_solver == 'direct':
            solve(A, self.u.vector(), b)
        else:
            solver = KrylovSolver('gmres', 'ilu')
            solver.solve(A, self.u.vector(), b)

    def solve(self):
        """Run time loop."""
        self.dt = self.problem.time_step(0)
        t = self.dt
        while t <= self.T:
            self.step(t)
            self.problem.user_action(t, self.u)
            t += self.dt
            self.u_1.assign(self.u)

class Problem(object):
    """Abstract base class for specific diffusion applications."""

    def solve(self, solver_class=Solver, linear_solver='direct',
              abs_tol=1E-6, rel_tol=1E-5, max_iter=1000):
        """Solve the PDE problem for the primary unknown."""
        self.solver = solver_class(self)
        iterative_solver = KrylovSolver('gmres', 'ilu')
        prm = iterative_solver.parameters
        prm['absolute_tolerance'] = abs_tol
        prm['relative_tolerance'] = rel_tol
        prm['maximum_iterations'] = max_iter
        prm['nonzero_initial_guess'] = True  # Use u (last sol.)
        return self.solver.solve()

    def flux(self):
        """Compute and return flux -p*grad(u)."""
        mesh = self.mesh()
        degree = self.solution().ufl_element().degree()
        V_g = VectorFunctionSpace(mesh, 'P', degree)
        self.flux_u = project(-self.p*grad(self.u), V_g)
        self.flux_u.rename('flux(u)', 'continuous flux field')
        return self.flux_u

    def mesh_degree(self):
        """Return mesh, degree."""
        raise NotImplementedError('Must implement mesh')

    def I(self):
        """Return initial condition."""
        return Constant(0.0)

    def p_coeff(self):
        return Constant(1.0)

    def f_rhs(self):
        return Constant(0.0)

    def time_step(self, t):
        raise NotImplentedError('Must implement time_step')

    def end_step(self):
        raise NotImplentedError('Must implement end_time')

    def solution(self):
        return self.solver.u

    def user_action(self, t, u):
        """Post process solution u at time t."""
        pass

    def update_boundary_conditions(self, t):
        """Update t parameter in Expression objects in BCs."""
        pass

    def Dirichlet_conditions(self):
        """Return either an Expression (for the entire boundary) or
        a list of (value,boundary_parts,index) triplets."""
        return []

    def Neumann_conditions(self):
        """Return list of (g,ds(n)) pairs."""
        return []

    def Robin_conditions(self):
        """Return list of (r,s,ds(n)) triplets."""
        return []


class Problem1(Problem):
    """Evolving boundary layer, I=0, but u=1 at x=0."""
    def __init__(self, Nx, Ny, p_values):
        Problem.__init__(self)
        self.init_mesh(Nx, Ny, p_values)
        self.Dirichlet_bc = Expression('sin(t)', t=0)
        self.file = File('temp.pvd')

    def init_mesh(self, Nx, Ny, p_values=[1, 0.1]):
        """Initialize mesh, boundary parts, and p."""
        self.mesh = UnitSquareMesh(Nx, Ny)
        self.divisions = (Nx, Ny)

        tol = 1E-14

        class BoundaryX0(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and abs(x[0]) < tol

        class BoundaryX1(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and abs(x[0] - 1) < tol

        class BoundaryY0(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and abs(x[1]) < tol

        class BoundaryY1(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and abs(x[1] - 1) < tol

        # Mark boundaries
        #self.boundary_parts = FacetFunction('size_t', mesh)
        self.boundary_parts = FacetFunction('uint', self.mesh)
        self.boundary_parts.set_all(9999)
        self.bx0 = BoundaryX0()
        self.bx1 = BoundaryX1()
        self.by0 = BoundaryY0()
        self.by1 = BoundaryY1()
        self.bx0.mark(self.boundary_parts, 0)
        self.bx1.mark(self.boundary_parts, 1)
        self.by0.mark(self.boundary_parts, 2)
        self.by1.mark(self.boundary_parts, 3)
        self.ds =  Measure(
            'ds', domain=self.mesh,
            subdomain_data=self.boundary_parts)

        # The domain is the unit square with an embedded rectangle
        class Rectangle(SubDomain):
            def inside(self, x, on_boundary):
                return 0.3 <= x[0] <= 0.7 and 0.3 <= x[1] <= 0.7

        self.materials = CellFunction('size_t', self.mesh)
        self.materials.set_all(0)  # "the rest"
        subdomain = Rectangle()
        subdomain.mark(self.materials, 1)
        self.V0 = FunctionSpace(self.mesh, 'DG', 0)
        self.p = Function(self.V0)
        help = np.asarray(self.materials.array(), dtype=np.int32)
        self.p.vector()[:] = np.choose(help, p_values)

    def time_step(self, t):
        return 0.025

    def end_time(self):
        return 0.25

    def user_action(self, t, u):
        """Post process solution u at time t."""
        print('user_action: t=%g, umax=%g' % (t, u.vector().array().max()))
        plot(u, interactive=True)
        self.file << (u, float(t))

        if not hasattr(self, 'solver'):
            return
        if not hasattr(self.solver, 'u'):
            print('no u in self.solver', dir(self.solver))
            return
        u2 = self.solver.u if u.ufl_element().degree() == 1 else \
             interpolate(self.u, FunctionSpace(mesh, 'P', 1))
        import sys
        sys.path.insert(0, 'modules')
        from BoxField import fenics_function2BoxField
        u_box = fenics_function2BoxField(
            u2, u.function_space().mesh(), self.divisions)
        u_ = u_box.values
        X = 0; Y = 1
        start = (0, 0.5)
        x, u_val, y_fixed, snapped = u_box.gridline(start, direction=X)
        plt.plot(x, u_val) #, 'r-')
        plt.xlabel('x');  plt.ylabel('y')

    def mesh_degree(self):
        return self.mesh, 1

    def p_coeff(self):
        return self.p

    def f_rhs(self):
        return Constant(0)

    def Dirichlet_conditions(self):
        """Return list of (value,boundary) pairs."""
        # return [(DirichletBC, self.boundary_parts, 2),
        return [(1.0, self.boundary_parts, 0),
                (0.0, self.boundary_parts, 1)]

    def Neumann_conditions(self, t):
        """Return list of g*ds(n) values."""
        return [(0, self.ds(0)), (0, self.ds(1))]

class Problem2(Problem1):
    """Oscillating surface temperature."""
    def __init__(self, Nx, Ny, p_values):
        Problem1.__init__(self, Nx, Ny, p_values)
        self.surface_temp = lambda t: T_A*sin(w*t)

    def Dirichlet_conditions(self):
        """Return list of (value,boundary) pairs."""
        # return [(DirichletBC, self.boundary_parts, 2),
        # t: self.solver.t
        return [(self.surface_temp(self.solver.t),
                 self.boundary_parts, 0),
                (0.0, self.boundary_parts, 1)]

def demo():
    problem = Problem1(Nx=20, Ny=5, p_values=[1,1])
    problem.solve(linear_solver='direct')
    u = problem.solution()
    plt.savefig('tmp1.png')
    plt.show()

def test_Solver():
    class TestProblemExact(Problem):
        def __init__(self, Nx, Ny):
            self.mesh = UnitSquareMesh(Nx, Ny)
            alpha = 3; beta = 1.2
            self.u0 = Expression(
                '1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                alpha=alpha, beta=beta, t=0)
            self.f = Constant(beta - 2 - 2*alpha)

        def time_step(self, t):
            return 0.3

        def end_time(self):
            return 0.9

        def mesh_degree(self):
            return self.mesh, 1

        def I(self):
            """Return initial condition."""
            return self.u0

        def f_rhs(self):
            return self.f

        def Dirichlet_conditions(self):
            return self.u0

        def update_boundary_conditions(self, t):
            """Update t parameter in Expression objects in BCs."""
            self.u0.t = t

        def user_action(self, t, u):
            """Post process solution u at time t."""
            u_e = interpolate(self.u0, u.function_space())
            error = np.abs(u_e.vector().array() -
                           u.vector().array()).max()
            print('error at %g: %g' % (t, error))
            assert error < 2E-15, 'max_error: %g' % error

    problem = TestProblemExact(Nx=2, Ny=2)
    problem.solve(linear_solver='direct')
    u = problem.solution()

if __name__ == '__main__':
    #demo()
    test_Solver()
    interactive()
