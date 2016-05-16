"""Refactored version of diffusion_plain*.py with functions."""
from __future__ import print_function
from fenics import *
import time

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
    rho, c, p, f, r, s, u0, T, L,       # physical parameters
    dt, divisions, degree=1, theta=1,   # numerical parameters
    user_action=None,                   # callback function
    u0_project=False,                   # project/interpolate u0
    lumped_mass=False,
    BC='Dirichlet',                     # interpretation of r
    A_is_const=False,                   # is A time independent?
    avoid_b_assemble=False,             # use trick for b
    debug=False):
    """
    Solve heat PDE: rho*c*du/dt = div(alpha*grad(u)) + f
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
    for obj in p, f, s:
        assert isinstance(obj, (Expression, Constant))
    if user_action is not None: assert callable(user_action)
    for obj in u0_project, A_is_const, avoid_b_assemble, debug:
        assert isinstance(obj, (int,bool))

    # Create mesh and define function space
    if d == 1:
        mesh = IntervalMesh(divisions[0], 0, L[0])
    elif d == 2:
        mesh = RectangleMesh(Point(0,0), Point(*L), *divisions)
    elif d == 3:
        mesh = BoxMesh(Point(0,0), Point(*L), *divisions)
    V = FunctionSpace(mesh, 'P', degree)

    boundary_parts = mark_boundaries_in_hypercube(mesh, d)
    ds =  Measure('ds', domain=mesh, subdomain_data=boundary_parts)
    # subdomain 0: x=0, 1: x=1, 2: y=0, 3: y=1, 4: z=0, 5: z=1
    bcs = []
    if BC == 'Dirichlet':
        for i in range(2*d):
            bcs.append(DirichletBC(V, r[i], boundary_parts, i))

    if debug:
        coor = mesh.coordinates()
        # Print the Dirichlet conditions
        d2v = dof_to_vertex_map(V)
        print('vertex_to_dof:', vertex_to_dof_map(V))
        for bc in bcs:
            bc_dict = bc.get_boundary_values()
            for dof in bc_dict:
                print('dof %2d: u=%g' % (dof, bc_dict[dof]))
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

    unit_func = interpolate(Constant(1.0), V) # for lumped mass

    # Bug: had + f instead of -f, reduce to 1D, split into M, K, R,
    # 2 elements, hand-calc, had forgotten u_exact.t =t update
    # before error computation, wrong variable name bc prevented updating
    # of Dirichlet conditions
    def D(u):
        return p*dot(grad(u), grad(v))*dx
    def B(u, i):
        return r[i]*(u-s)*v*ds(i)  # s can be time-dep, must have right value

    # Must set the t attribute in f and r[i] to
    # theta*t + (1-theta)*(t-dt) before evaluating the forms
    U = theta*u + (1-theta)*u_n
    F_M = rho*c*(u-u_n)/dt*v*dx
    F_K = D(U)
    F_f = -f*v*dx
    F = F_M + F_K + F_f
    #F = rho*c*(u-u_n)/dt*v*dx + theta*D(u) + (1-theta)*D(u_n)
    if BC == 'Robin':
        F_R = sum(B(U, i) for i in range(2*d))
        F += F_R
    if debug:
        print('M:\n', assemble(lhs(F_M)).array())
        print('K:\n', assemble(lhs(F_K)).array())
        print('R:\n', assemble(lhs(F_R)).array())
        print('A:\n', assemble(lhs(F)).array())
        print('b M:', assemble(rhs(F_M)).array())
        print('b f:', assemble(rhs(F_f)).array())
        print('b R:', assemble(rhs(F_R)).array())
        print('b:', assemble(rhs(F)).array())
    a, L = lhs(F), rhs(F)

    if A_is_const:
        if lumped_mass and theta == 0:
            A = assemble(lhs(F_M))*unit_func.vector() # sum rows
            print('Lumped:\n', A.array())
        else:
            A = assemble(a)
    if avoid_b_assemble:
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
            if lumped_mass and theta == 0:
                A = assemble(lhs(F_M))*unit_func.vector() # sum rows
                print('Lumped in loop:\n', A.array())
            else:
                A = assemble(a)

        if avoid_b_assemble:
            f_m = interpolate(f, V)
            F_m = f_m.vector()
            # Note that M = assemble(u*v), not assemble(F_M) with dt!
            b = 1./dt*M*u_n.vector() + M*F_m + (1-theta)*K*u_n.vector()
        else:
            b = assemble(L, tensor=b)
        cpu_assemble += time.clock() - t0
        # Doesn't work for lumped A:
        [bc.apply(A, b) for bc in bcs]
        if lumped_mass and theta == 0:
            u.vector()[:] = b.array()/A.array()
        else:
            solve(A, u.vector(), b)

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
    info('total time for assembly: %.2f' % cpu_assemble)

def verify(
    manufactured_u, d=2, degree=1, BC='Robin',
    N=16, theta=1, expect_exact_sol=True,
    lumped_mass=False, debug=False):

    import sympy as sym
    u = manufactured_u  # short form
    x, y, t = sym.symbols('x[0] x[1] t')
    if d == 1:  # 1D
        p = 1
        s = 1
        rho = c = 1
        # Fit f, r[i]
        f = rho*c*sym.diff(u, t) - sym.diff(p*sym.diff(u, x), x)
        f = sym.simplify(f)
        # Boundary conditions: r = -p*(du/dn)/(u-s)
        r = [None]*(2*d)
        r[0] = (+p*sym.diff(u, x)/(u-s)).subs(x, 0)
        r[1] = (-p*sym.diff(u, x)/(u-s)).subs(x, 1)
    elif d == 2:  # 2D
        #p = 2 + x + 2*y
        p = 1
        s = 2
        rho = c = 1
        f = rho*c*sym.diff(u, t) \
            - sym.diff(p*sym.diff(u, x), x) \
            - sym.diff(p*sym.diff(u, y), y)
        f = sym.simplify(f)           # fitted source term
        # Boundary conditions: r = -p*(du/dn)/(u-s)
        r = [None]*(2*d)
        r[0] = (+p*sym.diff(u, x)/(u-s)).subs(x, 0)
        r[1] = (-p*sym.diff(u, x)/(u-s)).subs(x, 1)
        r[2] = (+p*sym.diff(u, y)/(u-s)).subs(y, 0)
        r[3] = (-p*sym.diff(u, y)/(u-s)).subs(y, 1)

    for i in range(len(r)):
        r[i] = sym.simplify(r[i])
    print('f:', f, 'r:', r)

    # Convert symbolic expressions to Expression or Constant
    s = Constant(s)
    rho = Constant(rho)
    c = Constant(c)
    f = Expression(sym.printing.ccode(f), t=0)
    p = Expression(sym.printing.ccode(p))
    u_exact = Expression(sym.printing.ccode(u), t=0)

    if BC == 'Dirichlet':
        for i in range(len(r)):
            r[i] = u_exact
    elif BC == 'Robin':
        for i in range(len(r)):
            r[i] = Expression(sym.printing.ccode(r[i]), t=0)

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
            assert error < 1E-13, error

    A_is_const = BC == 'Dirichlet'
    # Match dt to N to keep dt/(2*d*dx**q) const,
    # q=1 for theta=0.5 else q=2
    dx = 1./N
    q = 1 if theta == 0.5 else 2
    dt = (0.05/(2*d*0.5**q))*2*d*dx**q
    T = 5*dt  # always 5 steps
    if d == 1:
        divisions = (N,)
        L = (1,)
    elif d == 2:
        divisions = (N, N)
        L = (1, 1)
    solver(rho, c, p, f, r, s, u_exact, T, L,
           dt, divisions, degree=degree, theta=theta,
           user_action=print_error,
           u0_project=False, BC=BC, A_is_const=A_is_const,
           lumped_mass=lumped_mass, debug=debug)

def test_solver():
    import sympy as sym
    x, y, t = sym.symbols('x[0] x[1] t')
    u = 1 + x**2 + 3*t   # manufactured solution
    #verify(u, d=1, degree=1, BC='Dirichlet', N=4, theta=0, lumped_mass=True)
    #1/0

    verify(u, d=1, degree=1, BC='Dirichlet', N=2, theta=1)
    verify(u, d=1, degree=1, BC='Dirichlet', N=20, theta=1)
    verify(u, d=1, degree=2, BC='Dirichlet', N=2, theta=1) # 1E-13
    verify(u, d=1, degree=1, BC='Robin', N=2,  theta=1)
    verify(u, d=1, degree=1, BC='Robin', N=20, theta=1)
    verify(u, d=1, degree=2, BC='Robin', N=2,  theta=1)
    verify(u, d=1, degree=2, BC='Robin', N=2,  theta=0.5)
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
        f_n = interpolate(f, V)
        F_n = f_n.vector()
        b = M*u_1.vector() + dt*M*F_n
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
    #info('total time for assembly of right-hand side: %.2f' % b_assemble)

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
        f_n = interpolate(f, V)
        F_n = f_n.vector()
        b = M*u_1.vector() + dt*M*F_n + dt*b_surface_int
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
    #application_animate(2)
    #solver_vs_solver_minimize_assembly()
    #application_welding(gamma=10)
    test_solver()
    interactive()
