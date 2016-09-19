from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

class DiffusionSolver(object):
    """Solve a heat conduction problem by the theta-rule."""
    def __init__(self, problem, theta=0.5):
        self.problem = problem
        self.theta = theta

    def solve(self):
        """Run time loop."""
        tol = 1E-14
        T = self.problem.end_time()
        t = self.problem.time_step(0)
        self.initial_condition()
        timestep = 1

        while t <= T+tol:
            self.step(t)
            self.problem.user_action(t, self.u, timestep)

            # Updates
            self.dt = self.problem.time_step(
                t+self.problem.time_step(t))
            t += self.dt
            timestep += 1
            self.u_1.assign(self.u)

    def initial_condition(self):
        self.mesh, degree = self.problem.mesh_degree()
        self.V = V = FunctionSpace(self.mesh, 'P', degree)

        if hasattr(self.problem, 'I_project'):
            I_project = getattr(self.problem, 'I_project')
        else:
            I_project = False
        self.u_1 = project(self.problem.I(), V) if I_project \
                   else interpolate(self.problem.I(), V)
        self.u_1.rename('u', 'initial condition')
        self.u = self.u_1 # needed if flux is computed in the next step
        self.problem.user_action(0, self.u_1, 0)

    def step(self, t, linear_solver='direct',
             abs_tol=1E-6, rel_tol=1E-5, max_iter=1000):
        """Advance solution one time step."""
        # Find new Dirichlet conditions at this time step
        Dirichlet_cond = self.problem.Dirichlet_conditions(t)
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

        #debug_Dirichlet_conditions(self.bcs, self.mesh, self.V)

        self.define_variational_problem(t)
        a, L = lhs(self.F), rhs(self.F)
        A = assemble(a)
        b = assemble(L)

        [bc.apply(A, b) for bc in self.bcs]
        if self.V.dim() < 50:
            print('A:\n', A.array(), '\nb:\n', b.array())

        # Solve linear system
        if linear_solver == 'direct':
            solve(A, self.u.vector(), b)
        else:
            solver = KrylovSolver('gmres', 'ilu')
            solver.solve(A, self.u.vector(), b)

    def define_variational_problem(self, t):
        """Set up variational problem at time t."""
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        dt     = self.problem.time_step(t)
        kappa  = self.problem.heat_conduction()
        varrho = self.problem.density()
        c      = self.problem.heat_capacity()
        f      = self.problem.heat_source(t)
        f_1    = self.problem.heat_source(t-dt)

        theta = Constant(self.theta)
        u_n = self.u_1  # first computed in initial_condition

        F = varrho*c*(u - u_n)/dt*v
        F += theta    *dot(kappa*grad(u),   grad(v))
        F += (1-theta)*dot(kappa*grad(u_n), grad(v))
        F -= theta*f*v + (1-theta)*f_1*v
        F = F*dx
        F += theta*sum(
            [g*v*ds_
             for g, ds_ in self.problem.Neumann_conditions(t)])
        F += (1-theta)*sum(
            [g*v*ds_
             for g, ds_ in self.problem.Neumann_conditions(t-dt)])
        F += theta*sum(
            [r*(u - U_s)*v*ds_
             for r, U_s, ds_ in self.problem.Robin_conditions(t)])
        F += (1-theta)*sum(
            [r*(u - U_s)*v*ds_
             for r, U_s, ds_ in self.problem.Robin_conditions(t-dt)])
        self.F = F

        self.u = Function(self.V)
        self.u.rename('u', 'solution')

def debug_Dirichlet_conditions(bcs, mesh, V, max_unknowns=50):
    if V.dim() > max_unknowns:
        return
    # Print the Dirichlet conditions
    print('No of Dirichlet conditions:', len(bcs))
    coor = mesh.coordinates()
    d2v = dof_to_vertex_map(V)
    for bc in bcs:
        bc_dict = bc.get_boundary_values()
        for dof in bc_dict:
            print('dof %2d: u=%g' % (dof, bc_dict[dof]))
            if V.ufl_element().degree() == 1:
                print('   at point %s' %
                      (str(tuple(coor[d2v[dof]].tolist()))))

class DiffusionProblem(object):
    """Abstract base class for specific diffusion applications."""

    def solve(self, solver_class=DiffusionSolver,
              theta=0.5, linear_solver='direct',
              abs_tol=1E-6, rel_tol=1E-5, max_iter=1000):
        """Solve the PDE problem for the primary unknown."""
        self.solver = solver_class(self, theta)
        iterative_solver = KrylovSolver('gmres', 'ilu')
        prm = iterative_solver.parameters
        prm['absolute_tolerance'] = abs_tol
        prm['relative_tolerance'] = rel_tol
        prm['maximum_iterations'] = max_iter
        prm['nonzero_initial_guess'] = True  # Use u (last sol.)
        return self.solver.solve()

    def flux(self):
        """Compute and return flux -p*grad(u)."""
        degree = self.solution().ufl_element().degree()
        V_g = VectorFunctionSpace(self.mesh, 'P', degree)
        flux_u = -self.heat_conduction()*grad(self.solution())
        self.flux_u = project(flux_u, V_g)
        self.flux_u.rename('flux(u)', 'continuous flux field')
        return self.flux_u

    def mesh_degree(self):
        """Return mesh, degree."""
        raise NotImplementedError('Must implement mesh')

    def I(self):
        """Return initial condition."""
        return Constant(0.0)

    def heat_conduction(self):  # kappa
        return Constant(1.0)

    def density(self):          # rho
        return Constant(1.0)

    def heat_capacity(self):    # c
        return Constant(1.0)

    def heat_source(self, t):   # f
        return Constant(0.0)

    def time_step(self, t):
        raise NotImplentedError('Must implement time_step')

    def end_time(self):
        raise NotImplentedError('Must implement end_time')

    def solution(self):
        return self.solver.u

    def user_action(self, t, u):
        """Post process solution u at time t."""
        pass

    def Dirichlet_conditions(self, t):
        """Return either an Expression (for the entire boundary) or
        a list of (value,boundary_parts,index) triplets."""
        return []

    def Neumann_conditions(self, t):
        """Return list of (g,ds(n)) pairs."""
        return []

    def Robin_conditions(self, t):
        """Return list of (r,s,ds(n)) triplets."""
        return []


import cbcpost as post
class ProcessSolution(object):
    """user_action function for storing the solution and flux."""
    def __init__(self, problem, u_min=0, u_max=1):
        """Define fields to be stored/plotted."""
        self.problem = problem  # this user_action belongs to problem
        self.pp = post.PostProcessor(
            dict(casedir='Results', clean_casedir=True))

        self.pp.add_field(
            post.SolutionField(
                'Solution',
                dict(save=True,
                     save_as=['hdf5', 'xdmf'],  # format
                     plot=True,
                     plot_args=
                     dict(range_min=float(u_min),
                          range_max=float(u_max))
                     )))

        self.pp.add_field(
            post.SolutionField(
                "Flux",
                dict(save=True,
                     save_as=["hdf5", "xdmf"],  # format
                     )))

    def __call__(self, t, u, timestep):
        """Store u and its flux to file."""
        u.rename('u', 'Solution')
        self.pp.update_all(
            {'Solution': lambda: u,
             'Flux': lambda: self.problem.flux()},
            t, timestep)
        info('saving results at time %g, max u: %g' %
             (t, u.vector().array().max()))

def mark_boundaries_in_hypercube(
    mesh, d=2, x0=0, x1=1, y0=0, y1=1, z0=0, z1=1):
    """
    Return mesh function FacetFunction with each side in a hypercube
    in d dimensions. Sides are marked by indicators 0, 1, 2, ..., 6.
    Side 0 is x=x0, 1 is x=x1, 2 is y=y0, 3 is y=y1, and so on.
    """
    side_definitions = [
        'near(x[0], %(x0)s, tol)', 'near(x[0], %(x1)s, tol)',
        'near(x[1], %(y0)s, tol)', 'near(x[1], %(y1)s, tol)',
        'near(x[2], %(z0)s, tol)', 'near(x[2], %(z1)s, tol)']
    boundaries = [CompiledSubDomain(
        ('on_boundary && ' + side_definition) % vars(), tol=1E-14)
                  for side_definition in side_definitions[:2*d]]
    # Mark boundaries
    boundary_parts = FacetFunction('uint', mesh)
    boundary_parts.set_all(9999)
    for i in range(len(boundaries)):
        boundaries[i].mark(boundary_parts, i)
    return boundary_parts


class Problem1(DiffusionProblem):
    """Evolving boundary layer, I=0, but u=1 at x=0."""
    def __init__(self, Nx, Ny):
        self.make_mesh(self, Nx, Ny)
        # Storage and visualization
        self.user_action_object = \
                   ProcessSolution(self, u_min=0, u_max=1)
        # Compare u(x,t) as curve plots for the following times
        self.times4curveplots = [
            self.time_step(0), 4*self.time_step(0),
            8*self.time_step(0), 12*self.time_step(0),
            16*self.time_step(0), 0.02, 0.1, 0.2, 0.3]
        plt.ion()  # for animation

    def make_mesh(self, Nx, Ny):
        """Initialize mesh, boundary parts, and p."""
        print('XXX in Problem1.make_mesh')
        self.mesh = UnitSquareMesh(Nx, Ny)
        self.divisions = (Nx, Ny)

        self.boundary_parts = \
             mark_boundaries_in_hypercube(self.mesh, d=2)
        self.ds =  Measure(
            'ds', domain=self.mesh,
            subdomain_data=self.boundary_parts)

    def time_step(self, t):
        # Small time steps in the beginning when the boundary
        # layer develops
        if t < 0.02:
            return 0.0005
        else:
            return 0.025

    def end_time(self):
        return 0.3

    def mesh_degree(self):
        return self.mesh, 1

    def Dirichlet_conditions(self, t):
        """Return list of (value,boundary) pairs."""
        return [(1.0, self.boundary_parts, 0),
                (0.0, self.boundary_parts, 1)]

    def user_action(self, t, u, timestep):
        """Post process solution u at time t."""
        tol = 1E-14
        self.user_action_object(t, u, timestep)
        # Also plot u along line y=1/2
        x_coor = np.linspace(tol, 1-tol, 101)
        x = [(x_,0.5) for x_ in x_coor]
        u = self.solution()
        u_line = [u(x_) for x_ in x]
        # Animation in figure(1)
        plt.figure(1)
        if timestep == 0:
            self.lines = plt.plot(x_coor, u_line, 'b-')
            plt.legend(['u, t=%.4f' % t])
            plt.title('Solution along y=1/2, time step: %g' %
                      self.time_step(t))
            plt.xlabel('x'), plt.ylabel('u')
            plt.axis([0, 1, 0, 1])
            plt.savefig('tmp_%04d.png' % timestep)
        else:
            self.lines[0].set_ydata(u_line)
            plt.title('Solution along y=1/2, time step: %g' %
                      self.time_step(t))
            plt.draw()
            plt.savefig('tmp_%04d.png' % timestep)
        # Accumulated selected curves in one plot in figure(2)
        plt.figure(2)
        for t_ in self.times4curveplots:
            if abs(t - t_) < 0.5*self.time_step(t):
                plt.plot(x_coor, u_line, '-')
                plt.legend(['u, t=%.4f' % t])
                plt.xlabel('x'); plt.ylabel('u')
                plt.axis([0, 1, 0, 1])
                plt.hold('on')
        # Classical matplotlib commands (no animation, just
        # accumulation of curves)
        """
        for t_ in self.times4curveplots:
            if abs(t - t_) < 0.5*self.time_step(t):
                plt.plot(x_coor, u_line) #, 'r-')
                plt.xlabel('x');  plt.ylabel('u')
                plt.legend(['u, t=%.4f' % t])
                plt.axis([0, 1, 0, 1])
        """

class Problem2(Problem1):
    """As Problem 1, but du/dn at x=1 and varying kappa."""
    def __init__(self, Nx, Ny, kappa_values, subdomain_corners):
        """
        Nx x Ny mesh. kappa_values=[1,a],
        subdomain_corners=[(0.3,0.3), (0.7,0.9)]
        """
        self.make_mesh(Nx, Ny, kappa_values, subdomain_corners)
        self.user_action_object = \
                   ProcessSolution(self, u_min=0, u_max=1)
        # Compare u(x,t) as curve plots for the following times
        self.times4curveplots = [
            12*self.time_step(0),
            0.02, 0.1, 0.3, 0.5]
        plt.ion()  # for animation


    def make_mesh(self, Nx, Ny, kappa_values, subdomain_corners):
        """Initialize mesh, boundary parts, and p."""
        self.mesh = UnitSquareMesh(Nx, Ny)
        self.divisions = (Nx, Ny)

        self.boundary_parts = \
             mark_boundaries_in_hypercube(self.mesh, d=2)
        self.ds =  Measure(
            'ds', domain=self.mesh,
            subdomain_data=self.boundary_parts)

        # The domain is the unit square with an embedded rectangle
        class Rectangle(SubDomain):
            def __init__(self, subdomain_def):
                self.subdomain_def = subdomain_def
                SubDomain.__init__(self)

            def inside(self, x, on_boundary):
                # subdomain_def:
                # 0.3 <= x[0] <= 0.7 && 0.5 <= x[1] <= 0.9
                return eval(self.subdomain_def)

        self.materials = CellFunction('size_t', self.mesh)
        self.materials.set_all(0)  # "the rest"
        # Give subdomain_corners as (c,c),(b,b)
        c = subdomain_corners
        subdomain_str = '%g <= x[0] <= %g and %g <= x[1] <= %g' % \
                        (c[0][0], c[1][0], c[0][1], c[1][1])
        #subdomain = CompiledSubDomain(subdomain_str.replace('and', '&&'))
        subdomain = Rectangle(subdomain_str)
        subdomain.mark(self.materials, 1)
        self.V0 = FunctionSpace(self.mesh, 'DG', 0)
        self.kappa = Function(self.V0)
        help = np.asarray(self.materials.array(), dtype=np.int32)
        self.kappa.vector()[:] = np.choose(help, kappa_values)
        plot(self.materials, title='Subdomain', interactive=True)

    def time_step(self, t):
        if t < 0.04:
            return 0.0005
        else:
            return 0.025

    def end_time(self):
        return 0.5

    def heat_conduction(self):
        return self.kappa

    def Dirichlet_conditions(self, t):
        """Return list of (value,boundary) pairs."""
        return [(1.0, self.boundary_parts, 0)]

class Problem3(Problem2):
    """Oscillating surface temperature."""
    def __init__(self, Nx, Ny, kappa_values, subdomain_corners):
        # Oscillating temperature at x=0:
        self.surface_temp = lambda t: sin(2*t)
        w = 2.0
        period = 2*np.pi/w
        self.dt = period/30  # need this before Problem2.__init__
        self.T = 4*period

        Problem2.__init__(
            self, Nx, Ny, kappa_values, subdomain_corners)
        # Stretch the mesh in y direction so we get [0,4]x[0,4]
        self.mesh.coordinates()[:] *= 4

        self.user_action_object = \
                   ProcessSolution(self, u_min=-1, u_max=1)
        # Compare u(x,t) as curve plots for the following times
        self.times4curveplots = [
            period/4, 4*period/8, 3*period/4, 3*period]

    def time_step(self, t):
        return self.dt

    def end_time(self):
        return self.T

    def Dirichlet_conditions(self, t):
        """Return list of (value,boundary) pairs."""
        return [(self.surface_temp(t), self.boundary_parts, 3)]

    def user_action(self, t, u, timestep):
        """Post process solution u at time t."""
        tol = 1E-14
        self.user_action_object(t, u, timestep)
        # Also plot u along line x=2
        y_coor = np.linspace(tol, 4-tol, 101)
        y = [(2, y_) for y_ in y_coor]
        u = self.solution()
        u_line = [u(y_) for y_ in y]
        # Animation in figure(1)
        plt.figure(1)
        if timestep == 0:
            self.lines = plt.plot(y_coor, u_line, 'b-')
            plt.legend(['u, t=%.4f' % t])
            plt.title('Solution along x=2, time step: %g' %
                      self.time_step(t))
            plt.xlabel('y'); plt.ylabel('u')
            plt.axis([0, 4, -1.2, 1.2])
            plt.savefig('tmp_%04d.png' % timestep)
        else:
            self.lines[0].set_ydata(u_line)
            plt.title('Solution along x=2, time step: %g' %
                      self.time_step(t))
            plt.draw()
            plt.savefig('tmp_%04d.png' % timestep)

        # Accumulated selected curves in one plot in figure(2)
        plt.figure(2)
        for t_ in self.times4curveplots:
            if abs(t - t_) < 0.5*self.time_step(t):
                plt.plot(y_coor, u_line, '-')
                plt.legend(['u, t=%.4f' % t])
                plt.xlabel('y'); plt.ylabel('u')
                plt.axis([0, 4, -1.2, 1.2])
                plt.hold('on')

def demo_Problem1():
    problem = Problem1(Nx=20, Ny=5)
    problem.solve(theta=1, linear_solver='direct')
    plt.figure(2)
    plt.savefig('tmp1.png'); plt.savefig('tmp1.pdf')

def demo_Problem2():
    problem = Problem2(Nx=20, Ny=5, kappa_values=[1,1000],
              subdomain_corners=[(0.3,0.3), (0.7,0.7)])
    problem.solve(theta=0.5, linear_solver='direct')
    plt.figure(2)
    plt.savefig('tmp1.png'); plt.savefig('tmp1.pdf')

def demo_Problem3():
    problem = Problem3(Nx=5, Ny=20, kappa_values=[1, 1000],
              subdomain_corners=[(0.3,0.4), (0.7,0.8)])
    problem.solve(theta=0.5, linear_solver='direct')
    plt.figure(2)
    plt.savefig('tmp1.png'); plt.savefig('tmp1.pdf')

class TestProblemExact(DiffusionProblem):
    def __init__(self, Nx, Ny, Nz=None, degree=1, num_time_steps=3):
        if Nz is None:
            self.mesh = UnitSquareMesh(Nx, Ny)
        else:
            self.mesh = UnitCubeMesh(Nx, Ny, Nz)
        self.degree = degree
        self.num_time_steps = num_time_steps

        alpha = 3; beta = 1.2
        self.u0 = Expression(
            '1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
            alpha=alpha, beta=beta, t=0)
        self.f = Constant(beta - 2 - 2*alpha)

    def time_step(self, t):
        return 0.3

    def end_time(self):
        return self.num_time_steps*self.time_step(0)

    def mesh_degree(self):
        return self.mesh, self.degree

    def I(self):
        """Return initial condition."""
        return self.u0

    def heat_source(self, t):
        return self.f

    def Dirichlet_conditions(self, t):
        self.u0.t = t
        return self.u0

    def user_action(self, t, u, timestep):
        """Post process solution u at time t."""
        u_e = interpolate(self.u0, u.function_space())
        error = np.abs(u_e.vector().array() -
                       u.vector().array()).max()
        print('error at %g: %g' % (t, error))
        tol = 2E-11
        assert error < tol, 'max_error: %g' % error


def test_DiffusionSolver():
    problem = TestProblemExact(Nx=2, Ny=2)
    problem.solve(theta=1, linear_solver='direct')
    u = problem.solution()

if __name__ == '__main__':
    #test_DiffusionSolver()
    demo_Problem3()
    interactive()
