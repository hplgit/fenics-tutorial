"""Refactored version of heat.py with functions."""
from __future__ import print_function
from fenics import *
import numpy as np
import sympy as sym
import time

# The next function, mark_boundaries_in_hypercube, is more
# general and shorter. mark_boundaries_in_rectangle is for
# pedagogical purposes mainly.

def mark_boundaries_in_rectangle(mesh, x0=0, x1=1, y0=0, y1=1):
    """
    Return mesh function FacetFunction with each side in a rectangle
    marked by boundary indicator 0, 1, 2, 3.
    Side 0 is x=x0, 1 is x=x1, 2 is y=y0, and 3 is y=y1.
    """
    tol = 1E-14

    class BoundaryX0(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], x0, tol)

    class BoundaryX1(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], x1, tol)

    class BoundaryY0(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], y0, tol)

    class BoundaryY1(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], y1, tol)

    # Mark boundaries
    boundary_parts = FacetFunction('uint', mesh)
    boundary_parts.set_all(9999)
    bx0 = BoundaryX0()
    bx1 = BoundaryX1()
    by0 = BoundaryY0()
    by1 = BoundaryY1()
    bx0.mark(boundary_parts, 0)
    bx1.mark(boundary_parts, 1)
    by0.mark(boundary_parts, 2)
    by1.mark(boundary_parts, 3)
    return boundary_parts

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


def solver(
    rho, c, kappa, f, r, s, u0, T, L,   # physical parameters
    dt, divisions, degree=1, theta=1,   # numerical parameters
    user_action=None,                   # callback function
    u0_project=False,                   # project/interpolate u0
    lumped_mass=False,
    BC='Dirichlet',                     # interpretation of r
    A_is_const=False,                   # is A time independent?
    avoid_b_assembly=False,             # use trick for b
    debug=False):
    """
    Solve heat PDE: rho*c*du/dt = div(kappa*grad(u)) + f
    in a box-shaped domain [0,L[0]]x[0,L[1]]x[0,L[2]] with
    partitioning given by divisions.
    If BC is 'Dirichlet': u = r[i] on boundaary i, else if BC
    is 'Robin': -p*du/dn=r[i]*(u-s) on boundary i.
    user_action(t, u, timestep) is a callback function for
    problem-dependent processing the solution at each time step.
    u0_project is False: the initial condition u0 is interpolated.
    """
    # Do some type and consistency checks via assert
    assert len(divisions) == len(L)
    d = len(L)  # No of space dimensions
    assert len(r) == 2*d
    for obj in kappa, f, s:
        assert isinstance(obj, (Expression, Constant))
    if user_action is not None: assert callable(user_action)
    for obj in u0_project, A_is_const, avoid_b_assembly, debug:
        assert isinstance(obj, (int,bool))
    if lumped_mass: assert A_is_const

    # Create mesh and define function space
    if d == 1:
        mesh = IntervalMesh(divisions[0], 0, L[0])
    elif d == 2:
        mesh = RectangleMesh(Point(0,0), Point(*L), *divisions)
    elif d == 3:
        mesh = BoxMesh(Point(0,0), Point(*L), *divisions)
    V = FunctionSpace(mesh, 'P', degree)
    if lumped_mass: assert V.ufl_element().degree() == 1

    boundary_parts = mark_boundaries_in_hypercube(mesh, d)
    ds =  Measure('ds', domain=mesh, subdomain_data=boundary_parts)
    # subdomain 0: x=0, 1: x=1, 2: y=0, 3: y=1, 4: z=0, 5: z=1
    bcs = []
    if BC == 'Dirichlet':
        for i in range(2*d):
            bcs.append(DirichletBC(V, r[i], boundary_parts, i))

    if debug:
        # Print the Dirichlet conditions
        if V.ufl_element().degree() == 1:  # P1 elements
            d2v = dof_to_vertex_map(V)
            coor = mesh.coordinates()
            print('vertex_to_dof:', vertex_to_dof_map(V))
        for n, bc in enumerate(bcs):
            boundary_values = bc.get_boundary_values()
            print('Dirichlet condition %d' % n)
            for dof in boundary_values:
                print(' dof %2d: u=%g' % (dof, boundary_values[dof]))
                if V.ufl_element().degree() == 1:
                    print('   at point %s' %
                          (str(tuple(coor[d2v[dof]].tolist()))))

    # Initial condition
    u_n = project(u0, V) if u0_project else interpolate(u0, V)
    u_n.rename('u', 'initial condition')
    if user_action is not None:
        user_action(0, u_n, 0)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    # Bug: had + f instead of -f
    # had forgotten u_exact.t =t update
    # before error computation, wrong variable name bc prevented updating
    # of Dirichlet conditions
    def D(u):
        return kappa*dot(grad(u), grad(v))*dx

    def B(u, i):
        return r[i]*(u-s)*v*ds(i)

    # In time loop: must set the t attribute in f, r[i], s to
    # theta*t + (1-theta)*(t-dt) before evaluating the forms
    U = theta*u + (1-theta)*u_n
    F_M = rho*c*(u-u_n)/dt*v*dx
    F_K = D(U)
    F_f = f*v*dx
    F = F_M + F_K - F_f
    #F = rho*c*(u-u_n)/dt*v*dx + theta*D(u) + (1-theta)*D(u_n)
    if BC == 'Robin':
        # Add cooling condition integrals from each side
        F_R = sum(B(U, i) for i in range(2*d))
        F += F_R
    if debug:
        print('M:\n', assemble(lhs(F_M)).array())
        if theta != 0:
            print('K:\n', assemble(lhs(F_K)).array())
            if BC == 'Robin':
                print('R:\n', assemble(lhs(F_R)).array())
        print('A:\n', assemble(lhs(F)).array())
        print('rhs M:', assemble(rhs(F_M)).array())
        print('rhs f:', assemble(rhs(F_f)).array())
        if BC == 'Robin':
            print('rhs R:', assemble(rhs(F_R)).array())
        print('b:', assemble(rhs(F)).array())
    a, L = lhs(F), rhs(F)

    if lumped_mass:
        # Need the function 1 for creating lumped mass by row sum
        unity = Function(V)
        unity.vector()[:] = 1.
        M = assemble(u*v*dx)
        K = assemble(D(u))

    if A_is_const:
        if lumped_mass and theta == 0:
            print('initial assembly of mass matrix for Forward Euler')
            A = assemble(lhs(F_M)) # make consistent mass, lump later
        else:
            print('initial assembly of A')
            A = assemble(a)

    if avoid_b_assembly:
        # Need M and K matrices
        M = assemble(u*v*dx)
        K = assemble(D(u))

    b = None          # necessary for memory saving assemeble call

    u = Function(V)   # the unknown at a new time level
    u.rename('u', 'solution')
    cpu_assemble = 0  # CPU time for assembling
    timestep = 1
    t = dt
    while t <= T:
        # Evaluate f, s, r[i] for right t value
        t_m = theta*t + (1-theta)*(t-dt)
        if hasattr(f, 't'): f.t = t_m
        if hasattr(s, 't'): s.t = t_m
        for i in range(len(r)):
            if BC == 'Robin':
                if hasattr(r[i], 't'): r[i].t = t_m
            elif BC == 'Dirichlet':
                if hasattr(r[i], 't'): r[i].t = t
            else:
                raise ValueError('BC=%s' % BC)
        t0 = time.clock()
        if not A_is_const:
            print('assembly of A, ', end='')
            A = assemble(a)

        if avoid_b_assembly:
            assert BC == 'Dirichlet'  # Robin is not impl.
            f_m = interpolate(f, V)
            F_m = f_m.vector()
            if lumped_mass:
                # Problem: cannot do rho*c/dt*M with Constant rho,c
                # A is rho*c*(1./dt)*M if rho, c, or dt is Constant
                ML = M*unity.vector()  # lump M by row sum
                # A = const*ML + theta*K = vector + matrix does not work
                # Must split in two
                A1 = (1./dt)*float(rho)*float(c)*ML
                A2 = theta*K
                if debug:
                    print('A1:\n', A1.array())
                    print('A2:\n', A2.array())
                diag = unity.vector().copy()
                A2.get_diagonal(diag)
                A2.set_diagonal(diag + A1)
                if debug:
                    print('diag of A2:\n', diag.array())
                    print('new A2:\n', A2.array())

                A = A2
                b = float(rho)*float(c)*1./dt*ML*u_n.vector() + \
                    ML*F_m - (1-theta)*K*u_n.vector()
                print('lumped M in A, lumped b mat-vec, ', end='')
            else:
                # Now A is matrix + matrix, works fine
                A = float(rho)*float(c)*(1./dt)*M + theta*K
                # Note that M = assemble(u*v), without any dt
                b = float(rho)*float(c)*(1./dt)*M*u_n.vector() + \
                    M*F_m - (1-theta)*K*u_n.vector()
                print('consistent A, but b via mat-vec, ', end='')
        else:
            b = assemble(L, tensor=b)
            print('assembly of b, ', end='')
        cpu_assemble += time.clock() - t0

        [bc.apply(A, b) for bc in bcs]
        if lumped_mass and theta == 0:
            # Could use A1 directly
            # u.vector()[:] = b.array()/A1.array()
            # but A1 has not boundary conditions incorporated,
            # need to set these in A first (as above) and then
            # lump
            A = A*unity.vector()  # lump mass via row sum
            u.vector()[:] = b.array()/A.array()
            print('lumped solve')
            if debug:
                print('lumped A:\n', A.array())
                print('b:\n', b.array())
        else:
            solve(A, u.vector(), b)
            print('standard solve')

            if debug:
                print('A:\n', A.array())
                print('b M:', assemble(rhs(F_M)).array())
                print('b f:', assemble(rhs(F_f)).array())
                print('b R:', assemble(rhs(F_R)).array())
                print('b:', b.array())
                print('u:', u.vector().array())

        if user_action is not None:
            user_action(t, u, timestep)
        t += dt
        timestep += 1
        u_n.assign(u)
    info('total time for assembly: %.2f for %d unknowns' %
         (cpu_assemble, V.dim()))

def verify(
    manufactured_u,         # SymPy expression
    d=2,                    # no of space dimensions
    degree=1,               # degree of finite element polynomials
    BC='Robin',             # type of boundary condition
    N=16,                   # partitioning in each space direction
    theta=1,                # time discretization parameter
    expect_exact_sol=True,  # True: no approximation errors
    lumped_mass=False,      # True: lump mass matrix
    avoid_b_assembly=False, # True: construct b as matrix-vector products
    A_is_const=None,        # None: set True if BC is 'Dirichlet'
    debug=False,            # True: dump a lot of debugging info
    error_tol=1E-13):       # tolerance for exact numerical solution

    u = manufactured_u  # short form
    x, y, z, t = sym.symbols('x[0] x[1] x[2] t')

    if d == 1:  # 1D test problem
        kappa = 1
        s = 1
        rho = c = 1
        # Fit f, r[i]
        f = rho*c*sym.diff(u, t) - sym.diff(kappa*sym.diff(u, x), x)
        f = sym.simplify(f)
        # Boundary conditions: r = -p*(du/dn)/(u-s)
        r = [None]*(2*d)
        r[0] = (+kappa*sym.diff(u, x)/(u-s)).subs(x, 0)
        r[1] = (-kappa*sym.diff(u, x)/(u-s)).subs(x, 1)

    elif d == 2:  # 2D
        kappa = 1
        s = 2
        rho = c = 1
        f = rho*c*sym.diff(u, t) \
            - sym.diff(kappa*sym.diff(u, x), x) \
            - sym.diff(kappa*sym.diff(u, y), y)
        f = sym.simplify(f)           # fitted source term
        # For Robin boundary conditions: r = -p*(du/dn)/(u-s)
        r = [None]*(2*d)
        r[0] = (+kappa*sym.diff(u, x)/(u-s)).subs(x, 0)
        r[1] = (-kappa*sym.diff(u, x)/(u-s)).subs(x, 1)
        r[2] = (+kappa*sym.diff(u, y)/(u-s)).subs(y, 0)
        r[3] = (-kappa*sym.diff(u, y)/(u-s)).subs(y, 1)

    elif d == 3:  # 3D
        p = kappa
        s = 2
        rho = c = 1
        f = rho*c*sym.diff(u, t) \
            - sym.diff(kappa*sym.diff(u, x), x) \
            - sym.diff(kappa*sym.diff(u, y), y) \
            - sym.diff(kappa*sym.diff(u, z), z)
        f = sym.simplify(f)           # fitted source term
        # For Robin boundary conditions: r = -p*(du/dn)/(u-s)
        r = [None]*(2*d)
        r[0] = (+kappa*sym.diff(u, x)/(u-s)).subs(x, 0)
        r[1] = (-kappa*sym.diff(u, x)/(u-s)).subs(x, 1)
        r[2] = (+kappa*sym.diff(u, y)/(u-s)).subs(y, 0)
        r[3] = (-kappa*sym.diff(u, y)/(u-s)).subs(y, 1)
        r[4] = (+kappa*sym.diff(u, z)/(u-s)).subs(z, 0)
        r[5] = (-kappa*sym.diff(u, z)/(u-s)).subs(z, 1)

    r = [sym.simplify(r[i]) for i in range(len(r))]
    print('f:', f, 'r:', r)

    # Convert symbolic expressions to Expression or Constant
    s = Constant(s)
    rho = Constant(rho)
    c = Constant(c)
    f = Expression(sym.printing.ccode(f), t=0)
    kappa = Expression(sym.printing.ccode(kappa))
    u_exact = Expression(sym.printing.ccode(u), t=0)

    if BC == 'Dirichlet':
        r = [u_exact for i in range(len(r))]
    elif BC == 'Robin':
        r = [Expression(sym.printing.ccode(r[i]), t=0)
             for i in range(len(r))]

    def print_error(t, u, timestep):
        """user_action function: print max error at dofs."""
        u_exact.t = t
        u_e = interpolate(u_exact, u.function_space())
        error = np.abs(u_e.vector().array() -
                       u.vector().array()).max()
        print('t=%.4f, error: %-10.3E max u: %-10.3f' %
              (t, error, u.vector().array().max()))
        if debug:
            print('u exact:', u_e.vector().array())
        if expect_exact_sol:
            assert error < error_tol, error

    # Match dt to N to keep dt/(2*d*dx**q) const,
    # q=1 for theta=0.5 else q=2
    dx = 1./N
    q = 1 if theta == 0.5 else 2
    dt = (0.05/(2*d*0.5**q))*2*d*dx**q
    print('dx=%g, dt=%g (kappa=1)' % (dx, dt))
    T = 5*dt  # always 5 steps
    divisions = [N]*d
    L = [1]*d
    if A_is_const is None:
        A_is_const = BC == 'Dirichlet'
    if lumped_mass:
        assert A_is_const

    solver(rho, c, kappa, f, r, s, u_exact, T, L,
           dt, divisions, degree=degree, theta=theta,
           user_action=print_error,
           u0_project=False, BC=BC, A_is_const=A_is_const,
           lumped_mass=lumped_mass,
           avoid_b_assembly=avoid_b_assembly, debug=debug)

def test_efficiency():
    """
    Measure the efficiency of various versions of a Forward
    Euler method: lumped coefficient matrix without any assembly,
    consistent coefficient matrix but no assembly, initial assembly
    of coefficient matrix and assembly of right-hand side,
    and full assembly at each time level.
    Gain: a factor of 6, 4, 2, and 1.
    """
    x, y, z, t = sym.symbols('x[0] x[1] x[2] t')

    # 2D efficiency test: N=180, same gain
    # 3D efficiency test
    u = 1 + x**2 + y + z**2 + 3*t
    # Lumped 0.14
    verify(u, d=3, degree=1, BC='Dirichlet', N=30, theta=0,
           lumped_mass=True, avoid_b_assembly=True, error_tol=1E-11)
    # No assembly of A and b, just matrix-vector operations 0.22
    verify(u, d=3, degree=1, BC='Dirichlet', N=30, theta=0,
           lumped_mass=False, avoid_b_assembly=True, error_tol=1E-10)
    # Initial assembly of A, assembly of b at each time level 0.42
    verify(u, d=3, degree=1, BC='Dirichlet', N=30, theta=0,
           lumped_mass=False, avoid_b_assembly=False, error_tol=1E-10)
    # Aseembly of A and b the normal way 0.86
    verify(u, d=3, degree=1, BC='Dirichlet', N=30, theta=0,
           lumped_mass=False, avoid_b_assembly=False, A_is_const=False,
           error_tol=1E-10)

def test_solver():
    x, y, z, t = sym.symbols('x[0] x[1] x[2] t')

    # 1D
    u = 1 + x**2 + 3*t
    verify(u, d=1, degree=1, BC='Dirichlet', N=20, theta=1)
    verify(u, d=1, degree=2, BC='Dirichlet', N=2, theta=1)
    verify(u, d=1, degree=1, BC='Robin', N=2,  theta=1)
    verify(u, d=1, degree=1, BC='Robin', N=20, theta=1)
    verify(u, d=1, degree=2, BC='Robin', N=2,  theta=1, error_tol=1.5E-13)
    verify(u, d=1, degree=2, BC='Robin', N=2,  theta=0.5)
    # Optimized versions
    verify(u, d=1, degree=1, BC='Dirichlet', N=2, theta=1,
           lumped_mass=True, avoid_b_assembly=True)
    verify(u, d=1, degree=1, BC='Dirichlet', N=2, theta=0.5,
           lumped_mass=True, avoid_b_assembly=True)
    verify(u, d=1, degree=1, BC='Dirichlet', N=2, theta=0,
           lumped_mass=True, avoid_b_assembly=True)

    # 2D
    u = 1 + x - 4*y**2 + 3*t
    verify(u, d=2, degree=1, BC='Dirichlet', N=2, theta=0.5)
    verify(u, d=2, degree=1, BC='Dirichlet', N=2, theta=1)
    verify(u, d=2, degree=1, BC='Dirichlet', N=2, theta=0)

    # 2D Robin will not give exact solutions - must measure
    # convergence rates.
    # The above u is not a good choice with Robin in 2D since
    # we get rational functions of t for r[i]
    u = x*(1-x)*(1-y)*sym.exp(-t)
    # Compare N=8 and N=16 for theta=1, 0.5: error reduction about 4
    # for theta in [0, 0.5]:
    #     verify(u, d=2, degree=1, BC='Robin', N=8, theta=theta,
    #                 expect_exact_sol=False)
    #     verify(u, d=2, degree=1, BC='Robin', N=16, theta=theta,
    #                 expect_exact_sol=False)


def animate_sine_spike(m=2):
    import numpy as np, time

    # Diffusion of a sin^8 spike, scaled homogeneous PDE
    u0 = Expression('pow(sin(pi*x[0])*sin(pi*x[1]), m)', m=m)
    c = rho = kappa = Constant(1)
    f = Constant(0)
    dt = 0.0005
    T = 20*dt
    L = [1, 1]
    divisions = (60, 60)
    u_range = [0, 1]

    # Neumann conditions (insulated boundary)
    BC = 'Robin'
    Nu = 0
    r = [Constant(Nu) for i in range(4)]   # Scaled Robin cond
    s = Constant(0)  # dummy, not used

    vtkfile = File('diffusion.pvd')

    def animate(t, u, timestep):
        global plt
        if t == 0:
            plt = plot(u, title='u',
                       range_min=float(u_range[0]),  # must be float
                       range_max=float(u_range[1]))  # must be float
        else:
            plt.plot(u)
            # Integral of u should remain constant
            u_integral = {2: 1./4, 8: 1225./16384}
            if m in u_integral:
                assert abs(assemble(u*dx) - u_integral[m]) < 1E-12
        print('t=%g' % t)
        time.sleep(0.5)           # pause between frames
        vtkfile << (u, float(t))  # store time-dep Function

    solver(
        rho, c, kappa, f, r, s, u0, T, L,
        dt, divisions, degree=1, theta=0.5,
        user_action=animate,
        u0_project=False,
        lumped_mass=False,
        BC=BC)
        #avoid_b_assembly=True does not yet work with Robin cond.


def welding(gamma=1, delta=70, beta=10, num_rotations=2, Nu=1):
    """Circular moving heat source for simulating welding."""
    d = 3  # no space dim
    from math import pi, sin, cos
    # Define physical parameters and boundary conditions
    u0 = Constant(0)
    rho = c = Constant(1)
    kappa = Constant(1.0/gamma)
    BC = 'Robin'
    Nu = 1
    r = [Constant(Nu) for i in range(2*d)]
    s = Constant(0)

    # Define welding source
    R = 0.2
    f = Expression(
        'delta*exp(-b*(pow(x[0]-(0.5+R*cos(t)),2) + '
        'pow(x[1]-(0.5+R*sin(t)),2)))',
        delta=delta, b=0.5*beta**2, R=R, t=0)
    omega = 1.0      # Scaled angular velocity
    P = 2*pi/omega   # One period of rotation
    T = num_rotations*P
    dt = P/40        # 40 steps per rotation

    import cbcpost as post
    class ProcessResults(object):
        def __init__(self):
            """Define fields to be stored/plotted."""
            # Dump temperature solution to std FEniCS file temp.h5
            self.timeseries_T = TimeSeries(mpi_comm_world(), 'temp')

            # Also dump temperature and source to cbcpost files
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

            self.timeseries_T.store(T.vector(), t)

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

    divisions = (40, 40, 4)
    L = (1, 1, 0.05)
    solver(
        rho, c, kappa, f, r, s, u0, T, L,
        dt, divisions, degree=1,
        theta=1,  # some oscillations in the beginning with theta=0.5
        user_action=ProcessResults(),
        u0_project=False,
        lumped_mass=False,
        BC=BC,
        A_is_const=False,
        avoid_b_assembly=False)

def solver_vs_solver_minimize_assembly():
    """
    Compute the relative efficiency of a standard assembly of b
    and the technique in solver_minimize_assembly.
    """
    import time
    alpha = 3; beta = 1.2
    u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                    alpha=alpha, beta=beta, t=0)
    f = Constant(beta - 2 - 2*alpha)
    dt = 0.3
    Nt = 10
    T = Nt*dt
    degree = 1

    from heat_class import TestProblemExact
    class ClassVersion(TestProblemExact):
        def user_action(self, t, u, timestep):
            return

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

        problem = ClassVersion(N, N, degree=degree, num_time_steps=Nt)
        problem.solve(theta=1, linear_solver='direct')
        t3 = time.clock()
        info('N=%3d, std solver: %.2f opt solver: %.2f class solver: %.2f speed-up: %.1f' %
             (N, t1-t0, t2-t1, t3-t2, (t1-t0)/float(t2-t1)))
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

        problem = ClassVersion(N, N, N, degree=degree, num_time_steps=Nt)
        problem.solve(theta=1, linear_solver='direct')
        t3 = time.clock()
        info('N=%3d, std solver: %.2f opt solver: %.2f class solver: %.2f speed-up: %.1f' %
             (N, t1-t0, t2-t1, t3-t2, (t1-t0)/float(t2-t1)))
        N *= 1.5
        N = int(round(N))
"""
P1:
N= 40, std solver: 0.10 opt solver: 0.08 class solver: 0.28 speed-up: 1.2
N= 80, std solver: 0.31 opt solver: 0.29 class solver: 0.54 speed-up: 1.1
N=160, std solver: 1.46 opt solver: 1.44 class solver: 2.06 speed-up: 1.0
N=320, std solver: 7.65 opt solver: 7.87 class solver: 10.58 speed-up: 1.0
N= 10, std solver: 0.20 opt solver: 0.11 class solver: 0.69 speed-up: 1.8
N= 15, std solver: 1.03 opt solver: 0.56 class solver: 1.40 speed-up: 1.9
N= 23, std solver: 13.39 opt solver: 6.15 class solver: 13.95 speed-up: 2.2

P2:
N= 40, std solver: 2.34 opt solver: 0.68 class solver: 1.06 speed-up: 3.4
N= 80, std solver: 2.58 opt solver: 1.83 class solver: 2.08 speed-up: 1.4
N=160, std solver: 20.63 opt solver: 22.82 class solver: 11.45 speed-up: 0.9
N=320, std solver: 165.11 opt solver: 100.15 class solver: 71.01 speed-up: 1.6
N= 10, std solver: 6.84 opt solver: 3.60 class solver: 8.84 speed-up: 1.9
N= 15, std solver: 96.20 opt solver: 56.10 class solver: 98.82 speed-up: 1.7
"""
if __name__ == '__main__':
    #test_solvers()
    #solver_vs_solver_minimize_assembly()
    #application_welding(gamma=10)
    #test_solver()
    #test_efficiency()
    #animate_sine_spike(m=2)
    #welding(gamma=1, delta=90, beta=10, num_rotations=2, Nu=1)
    #welding(gamma=0.1, delta=280, beta=10, num_rotations=2, Nu=1)
    #welding(gamma=30, delta=15, beta=10, num_rotations=2, Nu=1)
    welding(gamma=2000, delta=1, beta=10, num_rotations=2, Nu=1)
    interactive()
