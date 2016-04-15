"""Refactored version of diffusion_plain*.py with functions."""
from __future__ import print_function
from fenics import *
import time

def solver(alpha, f, u0, I, dt, T, divisions, L, degree=1,
           user_action=None, I_project=False):
    """
    Solve diffusion PDE: -div(alpha*grad(u))=f in a box-shaped
    domain [0,L[0]]x[0,L[1]]x[0,L[2]] with partitioning
    given by divisions. u=u0 on the boundary.
    user_action(t, u, timestep) is a callback function for
    problem-dependent processing the solution at each time step.
    If I_project is False, the initial condition I is interpolated
    at t=0.
    """
    # Create mesh and define function space
    assert len(divisions) == len(L)
    d = len(L)  # No of space dimensions
    if d == 1:
        mesh = IntervalMesh(divisions[0], 0, L[0])
    elif d == 2:
        mesh = RectangleMesh(Point(0,0), Point(*L), *divisions)
    elif d == 3:
        mesh = BoxMesh(Point(0,0), Point(*L), *divisions)
    V = FunctionSpace(mesh, 'P', degree)

    class Boundary(SubDomain):  # define the Dirichlet boundary
        def inside(self, x, on_boundary):
            return on_boundary

    boundary = Boundary()
    bc = DirichletBC(V, u0, boundary)

    # Initial condition
    u_1 = project(I, V) if I_project else interpolate(I, V)
    u_1.rename('u', 'initial condition')
    if user_action is not None:
        user_action(0, u_1, 0)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = u*v*dx + dt*alpha*dot(grad(u), grad(v))*dx
    L = (u_1 + dt*f)*v*dx

    A = assemble(a)   # assemble only once, before the time stepping
    b = None          # necessary for memory saving assemeble call

    # Compute solution
    u = Function(V)   # the unknown at a new time level
    u.rename('u', 'solution')
    b_assemble = 0  # CPU time for assembling all the b vectors
    timestep = 1
    t = dt
    while t <= T:
        t0 = time.clock()
        b = assemble(L, tensor=b)
        b_assemble += time.clock() - t0
        try:
            u0.t = t
            f.t = t
        except AttributeError:
            pass  # ok if no t attribute in u0
        bc.apply(A, b)
        solve(A, u.vector(), b)

        if user_action is not None:
            user_action(t, u, timestep)
        t += dt
        timestep += 1
        u_1.assign(u)
    info('total time for assembly of right-hand side: %.2f' % b_assemble)

def application():
    import numpy as np
    alpha = 3; beta = 1.2
    u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                    alpha=alpha, beta=beta, t=0)
    f = Constant(beta - 2 - 2*alpha)

    def print_error(t, u, timestep):
        u_e = interpolate(u0, u.function_space())
        error = np.abs(u_e.vector().array() -
                       u.vector().array()).max()
        print('t=%.2f, error: %-10.3f max u: %-10.3f' %
              (t, error, u.vector().array().max()))

    dt = 0.3; T = 1.9
    Nx = Ny = 20
    solver(1.0, f, u0, u0, dt, T, (Nx, Ny), (1, 1), degree=2,
           user_action=print_error, I_project=False)

def solver_minimize_assembly(
    alpha, f, u0, I, dt, T, divisions, L, degree=1,
    user_action=None, I_project=False):
    """
    Solve diffusion PDE u_t = div(alpha*grad(u)) + f on
    an interval, rectangle, or box with side lengths in L.
    divisions reflect cell partitioning, degree the element
    degree. user_action(t, u, timetesp) is a callback function
    where the calling code can process the solution.
    If I_project is false, use interpolation for the initial
    condition.
    """
    # Create mesh and define function space
    d = len(L)  # No of space dimensions
    if d == 1:
        mesh = IntervalMesh(divisions[0], 0, L[0])
    elif d == 2:
        mesh = RectangleMesh(Point(0,0), Point(*L), *divisions)
    elif d == 3:
        mesh = BoxMesh(Point(0,0), Point(*L), *divisions)
    V = FunctionSpace(mesh, 'P', degree)

    class Boundary(SubDomain):  # define the Dirichlet boundary
        def inside(self, x, on_boundary):
            return on_boundary

    boundary = Boundary()
    bc = DirichletBC(V, u0, boundary)

    # Initial condition
    u_1 = project(I, V) if I_project else interpolate(I, V)
    if user_action is not None:
        user_action(0, u_1, 0)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a_M = u*v*dx
    a_K = alpha*dot(grad(u), grad(v))*dx

    M = assemble(a_M)
    K = assemble(a_K)
    A = M + dt*K
    # Compute solution
    u = Function(V)   # the unknown at a new time level
    b_assemble = 0  # CPU time for assembling all the b vectors
    timestep = 1
    t = dt
    while t <= T:
        t0 = time.clock()
        f_k = interpolate(f, V)
        F_k = f_k.vector()
        b = M*u_1.vector() + dt*M*F_k
        b_assemble += time.clock() - t0
        try:
            u0.t = t
            f.t = t
        except AttributeError:
            pass  # ok if no t attribute in u0
        bc.apply(A, b)
        solve(A, u.vector(), b)

        if user_action is not None:
            user_action(t, u, timestep)
        t += dt
        timestep += 1
        u_1.assign(u)
    info('total time for assembly of right-hand side: %.2f' % b_assemble)

def application_animate(model_problem):
    import numpy as np, time

    if model_problem == 1:
        # Test problem with exact solution at the nodes also for P1 elements
        alpha = 3; beta = 1.2
        u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                        alpha=alpha, beta=beta, t=0)
        f = Constant(beta - 2 - 2*alpha)
        I = u0
        dt = 0.05; T = 2
        Nx = Ny = 20
        u_range = [1, 1+1+alpha*1+beta*T]
    elif model_problem == 2:
        # Diffusion of a sin^8 spike
        I = Expression('pow(sin(pi*x[0])*sin(pi*x[1]), 8)')
        f = Constant(0)
        u0 = Constant(0)
        dt = 0.0005; T = 20*dt
        Nx = Ny = 60
        u_range = [0, 1]

    vtkfile = File('diffusion.pvd')

    def animate(t, u, timestep):
        global p
        if t == 0:
            p = plot(u, title='u',
                     range_min=float(u_range[0]),  # must be float
                     range_max=float(u_range[1]))  # must be float
        else:
            p.plot(u)
        print('t=%g' % t)
        time.sleep(0.5)
        vtkfile << (u, float(t))  # store time-dep Function

    solver_minimize_assembly(
        1.0, f, u0, I, dt, T, (Nx, Ny), (1, 1), degree=2,
        user_action=animate, I_project=False)


def solver_bc(
    p, f,                   # Coefficients in the PDE
    I,                      # Initial condition
    dt,                     # Constant time step
    T,                      # Final simulation time
    boundary_conditions,    # Dict of boundary conditions
    Nx, Ny,                 # Cell division of the domain
    degree=1,               # Polynomial degree
    subdomains=[],          # List of SubDomain objects in domain
    user_action=None,       # Callback function
    I_project=False,        # Project or interpolate I
    linear_solver='Krylov', # Alt: 'direct'
    abs_tol=1E-5,           # Absolute tolerance in Krylov solver
    rel_tol=1E-3,           # Relative tolerance in Krylov solver
    max_iter=1000,          # Max no of iterations in Krylov solver
    log_level=PROGRESS,     # Amount of solver output
    dump_parameters=False,  # Write out parameter database?
    debug=False,
    ):
    """
    Solve du/dt = -alpha*div(p*grad(u)) + f on the unit square.
    """
    # Create mesh and define function space
    mesh = UnitSquareMesh(Nx, Ny)
    V = FunctionSpace(mesh, 'P', degree)

    tol = 1E-14

    # Subdomains in the domain?
    import numpy as np
    if subdomains:
        # subdomains is list of SubDomain objects,
        # p is array of corresponding constant values of p
        # in each subdomain
        if not isinstance(p, (list, tuple, np.ndarray)):
            raise TypeError(
                'p must be array if we have sudomains, not %s'
                % type(p))
        materials = CellFunction('size_t', mesh)
        materials.set_all(0)  # "the rest"
        for m, subdomain in enumerate(subdomains[1:], 1):
            subdomain.mark(materials, m)

        p_values = p
        V0 = FunctionSpace(mesh, 'DG', 0)
        p  = Function(V0)
        help = np.asarray(materials.array(), dtype=np.int32)
        p.vector()[:] = np.choose(help, p_values)
    else:
        if not isinstance(p, (Expression, Constant)):
            raise TypeError(
                'p is type %s, must be Expression or Constant'
                % type(p))

    # Boundary subdomains
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
    boundary_parts = FacetFunction('size_t', mesh)
    boundary_parts.set_all(9999)
    bx0 = BoundaryX0()
    bx1 = BoundaryX1()
    by0 = BoundaryY0()
    by1 = BoundaryY1()
    bx0.mark(boundary_parts, 0)
    bx1.mark(boundary_parts, 1)
    by0.mark(boundary_parts, 2)
    by1.mark(boundary_parts, 3)
    # boundary_parts.array() is a numpy array

    ds = Measure('ds', domain=mesh, subdomain_data=boundary_parts)

    # boundary_conditions is a dict of dicts:
    # {0: {'Dirichlet': u0},
    #  1: {'Robin': (r, s)},
    #  2: {'Neumann: g}},
    #  3: {'Neumann', 0}}

    bcs = []  # List of Dirichlet conditions
    for n in boundary_conditions:
        if 'Dirichlet' in boundary_conditions[n]:
            bcs.append(
                DirichletBC(V, boundary_conditions[n]['Dirichlet'],
                            boundary_parts, n))

    if debug:
        # Print the vertices that are on the boundaries
        coor = mesh.coordinates()
        for x in coor:
            if bx0.inside(x, True): print('%s is on x=0' % x)
            if bx1.inside(x, True): print('%s is on x=1' % x)
            if by0.inside(x, True): print('%s is on y=0' % x)
            if by1.inside(x, True): print('%s is on y=1' % x)


        # Print the Dirichlet conditions
        print('No of Dirichlet conditions:', len(bcs))
        d2v = dof_to_vertex_map(V)
        for bc in bcs:
            bc_dict = bc.get_boundary_values()
            for dof in bc_dict:
                print('dof %2d: u=%g' % (dof, bc_dict[dof]))
                if V.ufl_element().degree() == 1:
                    print('   at point %s' %
                          (str(tuple(coor[d2v[dof]].tolist()))))

    # Collect Neumann integrals
    u = TrialFunction(V)
    v = TestFunction(V)

    Neumann_integrals = []
    for n in boundary_conditions:
        if 'Neumann' in boundary_conditions[n]:
            if boundary_conditions[n]['Neumann'] != 0:
                g = boundary_conditions[n]['Neumann']
                Neumann_integrals.append(g*v*ds(n))

    # Collect Robin integrals
    Robin_a_integrals = []
    Robin_L_integrals = []
    for n in boundary_conditions:
        if 'Robin' in boundary_conditions[n]:
            r, s = boundary_conditions[n]['Robin']
            Robin_a_integrals.append(r*u*v*ds(n))
            Robin_L_integrals.append(r*s*v*ds(n))

    # Simpler Robin integrals
    Robin_integrals = []
    for n in boundary_conditions:
        if 'Robin' in boundary_conditions[n]:
            r, s = boundary_conditions[n]['Robin']
            Robin_integrals.append(r*(u-s)*v*ds(n))

    # Initial condition
    u_1 = project(I, V) if I_project else interpolate(I, V)
    if user_action is not None:
        user_action(0, u_1, 0)

    # Define variational problem
    a_M = u*v*dx
    a_K = p*dot(grad(u), grad(v))*dx + \
          sum(Robin_a_integrals)

    M = assemble(a_M)
    K = assemble(a_K)
    A = M + dt*K
    L = Constant(0)*v*dx # Must initialize L if next line has empty lists
    L += - sum(Neumann_integrals) + sum(Robin_L_integrals)
    b_surface_int = assemble(L)

    def update_boundary_conditions(boundary_conditions, t):
        """Update t parameter in Expression objects in BCs."""
        # This is more flexible and elegant in the class version
        for n in boundary_conditions:
            bc = boundary_conditions[n]
            if 'Robin' in bc:
                r, s = bc['Robin']
                try:
                    r.t = t
                    s.t = t
                except AttributeError:
                    pass
            else:
                for tp in 'Neumann', 'Dirichlet':
                    try:
                        bc[tp].t = t
                    except (AttributeError, KeyError):
                        pass

    # Compute solution
    u = Function(V)   # the unknown at a new time level
    timestep = 1
    t = dt
    while t <= T:
        f_k = interpolate(f, V)
        F_k = f_k.vector()
        b = M*u_1.vector() + dt*M*F_k + dt*b_surface_int
        update_boundary_conditions(boundary_conditions, t)
        [bc.apply(A, b) for bc in bcs]

        if linear_solver == 'direct':
            solve(A, u.vector(), b)
        else:
            solver = KrylovSolver('gmres', 'ilu')
            prm = solver.parameters
            prm['absolute_tolerance'] = abs_tol
            prm['relative_tolerance'] = rel_tol
            prm['maximum_iterations'] = max_iter
            prm['nonzero_initial_guess'] = True  # Use u (last sol.)
            solver.solve(A, u.vector(), b)

        if user_action is not None:
            user_action(t, u, timestep)
        t += dt
        timestep += 1
        u_1.assign(u)

def test_solvers():
    """Reproduce simple exact solution to "machine precision"."""
    tol = 5E-12  # This problem's precision
    # tol increases with degree and Nx,Ny
    # P1 2: E-15, P1 20: E-13, P2 20: E-12, same for P3
    import numpy as np
    alpha = 3; beta = 1.2
    u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                    alpha=alpha, beta=beta, t=0)
    f = Constant(beta - 2 - 2*alpha)

    def assert_error(t, u, timestep):
        u_e = interpolate(u0, u.function_space())
        error = np.abs(u_e.vector().array() -
                       u.vector().array()).max()
        assert error < tol, 'error: %g' % error

    for Nx, Ny in [(2,2), (3,5), (5,3), (20,20)]:
        for degree in 1, 2, 3:
            print('--- solving on 2(%dx%d) mesh with P%d elements'
                  % (Nx, Ny, degree))
            dt = 0.3; T = 1.2
            u0.t = 0 # Important, otherwise I is wrong
            solver(
                1.0, f, u0, u0, dt, T, (Nx, Ny), (1, 1), degree,
                user_action=assert_error, I_project=False)
            u0.t = 0 # Important, otherwise I is wrong
            solver_minimize_assembly(
                1.0, f, u0, u0, dt, T, (Nx, Ny), (1, 1), degree,
                user_action=assert_error, I_project=False)
            u0.t = 0 # Important, otherwise I is wrong
            solver_bc(
                Constant(1), f, u0, dt, T,
                {0: {'Dirichlet': u0}, 1: {'Dirichlet': u0},
                 2: {'Dirichlet': u0}, 3: {'Dirichlet': u0}},
                Nx, Ny, degree, subdomains=[],
                user_action=assert_error, I_project=False,
                linear_solver='direct')

def application_welding(gamma=1, delta=1, beta=10, num_rotations=2):
    """Circular moving heat source for simulating welding."""
    from math import pi, sin, cos
    u0 = Constant(0)
    I  = Constant(0)
    R = 0.2
    f = Expression(
        'delta*exp(-0.5*pow(beta,2)*(pow(x[0]-(0.5+R*cos(t)),2) + '
                                    'pow(x[1]-(0.5+R*sin(t)),2)))',
        delta=delta, beta=beta, R=R, t=0)
    omega = 1.0      # Scaled angular velocity
    P = 2*pi/omega   # One period of rotation
    T = num_rotations*P
    dt = P/40        # 40 steps per rotation

    import cbcpost as post
    class ProcessResults(object):
        def __init__(self):
            """Define fields to be stored/plotted."""
            self.pp = post.PostProcessor(
                dict(casedir='Results', clean_casedir=True))

            self.pp.add_field(
                post.SolutionField(
                    'Temperature',
                    dict(save=True,
                         save_as=['hdf5', 'xdmf'],  # format
                         plot=True,
                         plot_args=
                         dict(range_min=0.0, range_max=1.1)
                         )))

            self.pp.add_field(
                post.SolutionField(
                    "Heat_source",
                    dict(save=True,
                         save_as=["hdf5", "xdmf"],  # format
                         plot=True,
                         plot_args=
                         dict(range_min=0.0, range_max=float(delta))
                         )))
            # Save separately to VTK files as well
            self.vtkfile_T = File('temperature.pvd')
            self.vtkfile_f = File('source.pvd')

        def __call__(self, t, T, timestep):
            """Store T and f to file (cbcpost and VTK)."""
            T.rename('T', 'solution')
            f_Function = interpolate(f, T.function_space())
            f_Function.rename('f', 'welding equipment')

            self.pp.update_all(
                {'Temperature': lambda: T,
                 'Heat_source': lambda: f_Function},
                t, timestep)

            self.vtkfile_T << (T, float(t))
            self.vtkfile_f << (f_Function, float(t))
            info('saving results at time %g, max T: %g' %
                 (t, T.vector().array().max()))
            # Leave plotting to cbcpost

    Nx = Ny = 40
    solver_minimize_assembly(
        gamma, f, u0, I, dt, T, (Nx, Ny), (1, 1), degree=1,
        user_action=ProcessResults(), I_project=False)

def solver_vs_solver_minimize_assembly():
    """
    Compute the relative efficiency of a standard assembly of b
    and the technique in solver_minimize_assembly.
    """
    """
    Not a dramatic speed-up:
total time for assembly of right-hand side: 0.04
total time for assembly of right-hand side: 0.02
N=40, 1681 unknowns, std solver: 0.26 opt solver: 0.23 speed-up: 1.1
total time for assembly of right-hand side: 0.13
total time for assembly of right-hand side: 0.07
N=80, 6561 unknowns, std solver: 4.00 opt solver: 2.32 speed-up: 1.7
total time for assembly of right-hand side: 0.45
total time for assembly of right-hand side: 0.25
N=160, 25921 unknowns, std solver: 13.78 opt solver: 5.05 speed-up: 2.7
total time for assembly of right-hand side: 1.82
total time for assembly of right-hand side: 1.05
N=320, 103041 unknowns, std solver: 41.42 opt solver: 27.96 speed-up: 1.5

For P2 elements there is hardly any speed-up.
    """
    import time
    alpha = 3; beta = 1.2
    u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                    alpha=alpha, beta=beta, t=0)
    f = Constant(beta - 2 - 2*alpha)
    dt = 0.3; T = 40*dt
    degree = 2

    # 2D tests
    N = 40
    for i in range(4):
        t0 = time.clock()
        u0.t = 0
        solver(
            1.0, f, u0, u0, dt, T, (N, N), (1, 1), degree,
            user_action=None, I_project=False)
        t1 = time.clock()
        u0.t = 0
        solver_minimize_assembly(
            1.0, f, u0, u0, dt, T, (N, N), (1, 1), degree,
            user_action=None, I_project=False)
        t2 = time.clock()
        info('N=%d, std solver: %.2f opt solver: %.2f speed-up: %.1f' %
             (N, t1-t0, t2-t1, (t1-t0)/float(t2-t1)))
        N *= 2

    # 3D tests
    N = 10
    for i in range(3):
        t0 = time.clock()
        u0.t = 0
        solver(
            1.0, f, u0, u0, dt, T, (N, N, N), (1, 1, 1), degree,
            user_action=None, I_project=False)
        t1 = time.clock()
        u0.t = 0
        solver_minimize_assembly(
            1.0, f, u0, u0, dt, T, (N, N, N), (1, 1, 1), degree,
            user_action=None, I_project=False)
        t2 = time.clock()
        info('N=%d, std solver: %.2f opt solver: %.2f speed-up: %.1f' %
             (N, t1-t0, t2-t1, (t1-t0)/float(t2-t1)))
        N *= 2

if __name__ == '__main__':
    #application()
    #test_solvers()
    #application_animate(2)
    solver_vs_solver_minimize_assembly()
    #application_welding(gamma=10)
    interactive()
