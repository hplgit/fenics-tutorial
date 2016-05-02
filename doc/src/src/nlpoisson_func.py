"""
FEniCS tutorial demo program:
Nonlinear Poisson equation with Dirichlet conditions
in x-direction and homogeneous Neumann (symmetry) conditions
in all other directions. The domain is the unit hypercube in
of a given dimension.

-div(q(u)*grad(u)) = 0,
u = 0 at x=0, u=1 at x=1, du/dn=0 at all other boundaries.
q(u) = (1+u)^m

Solution method: automatic, i.e., by a
NonlinearVariationalProblem/Solver (Newton method).
"""

from __future__ import print_function
from fenics import *
import numpy, sys

def solver(
    q, Dq, f, divisions, degree=1,
    method='u', J_comp='manual',
    linear_solver='Krylov', # Alt: 'direct'
    abs_tol_Krylov=1E-5,
    rel_tol_Krylov=1E-5,
    abs_tol_Newton=1E-5,
    rel_tol_Newton=1E-5,
    max_iter_Krylov=1000,
    max_iter_Newton=50,
    relaxation_prm_Newton=1.0,
    log_level=PROGRESS,
    dump_parameters=False,
    ):
    # Create mesh and define function space
    d = len(divisions)
    domain_type = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
    mesh = domain_type[d-1](*divisions)
    V = FunctionSpace(mesh, 'P', degree)

    # Define boundary conditions
    tol = 1E-14
    def left_boundary(x, on_boundary):
        return on_boundary and abs(x[0]) < tol

    def right_boundary(x, on_boundary):
        return on_boundary and abs(x[0]-1) < tol

    # ***** Can this be right? u=1 at x=1? Init guess u_ is just zero at x=1
    # Or will NonlinearVariationalSolver employ Dirichlet values first
    # and then use zeros? Seems so...because it's getting right...................
    Gamma_0 = DirichletBC(V, Constant(0.0), left_boundary)
    Gamma_1 = DirichletBC(V, Constant(1.0), right_boundary)
    bcs = [Gamma_0, Gamma_1]

    # Define variational problem
    if method == 'alg_Newton':
        v  = TestFunction(V)
        u  = TrialFunction(V)
        F  = dot(q(u)*grad(u), grad(v))*dx
        u_ = Function(V)  # most recently computed solution
        F  = action(F, u_)
        # J must be a Jacobian (Gateaux deriv. in direction of du)
        if J_comp == 'manual':
            J = dot(q(u_)*grad(u), grad(v))*dx + \
                dot(Dq(u_)*u*grad(u_), grad(v))*dx
        else:
            J = derivative(F, u_, u)
    elif method == 'pde_Newton':
        v  = TestFunction(V)
        du = TrialFunction(V)
        u_ = Function(V)  # most recently computed solution
        F  = dot(q(u_)*grad(u_), grad(v))*dx
        if J_comp == 'manual':
            J = dot(q(u_)*grad(du), grad(v))*dx + \
                dot(Dq(u_)*du*grad(u_), grad(v))*dx
        else:
            J = derivative(F, u_, du)

    # Compute solution
    problem = NonlinearVariationalProblem(F, u_, bcs, J)
    solver  = NonlinearVariationalSolver(problem)

    prm = solver.parameters
    prm_n = prm['newton_solver']
    prm_n['absolute_tolerance'] = abs_tol_Newton
    prm_n['relative_tolerance'] = rel_tol_Newton
    prm_n['maximum_iterations'] = max_iter_Newton
    prm_n['relaxation_parameter'] = relaxation_prm_Newton
    if linear_solver == 'Krylov':
        prec = 'jacobi' if 'jacobi' in \
               list(zip(*krylov_solver_preconditioners()))[0] \
               else 'ilu'
        prm_n['linear_solver'] = 'gmres'
        prm_n['preconditioner'] = prec
        prm_nk = prm_n['krylov_solver']
        prm_nk['absolute_tolerance'] = abs_tol_Krylov
        prm_nk['relative_tolerance'] = rel_tol_Krylov
        prm_nk['maximum_iterations'] = max_iter_Krylov
        prm_nk['monitor_convergence'] = True
        prm_nk['nonzero_initial_guess'] = False
        prm_nk['gmres']['restart'] = 40
        prm_nk['preconditioner']['structure'] = \
                                        'same_nonzero_pattern'
        prm_nk['preconditioner']['ilu']['fill_level'] = 0

    set_log_level(log_level)
    solver.solve()
    if dump_parameters:
        info(parameters, True)
    return u_

def application_test():
    """Run the test problem with input from the command line."""
    # Choice of nonlinear coefficient
    m = 2

    def q(u):
        return (1+u)**m

    def Dq(u):
        return m*(1+u)**(m-1)

    usage = 'manual|automatic Krylov|direct degree nx ny nz'
    try:
        import sys
        J_comp = sys.argv[1]
        linear_solver = sys.argv[2]
        degree = int(sys.argv[3])
        divisions = [int(arg) for arg in sys.argv[4:]]
    except:
        print('Usage: %s' % sys.argv[0], usage)
        sys.exit(0)

    u = solver(q, Dq, f, divisions, degree,
               'pde_Newton', J_comp, linear_solver)

    # Find max error
    u_exact = Expression(
        'pow((pow(2, m+1)-1)*x[0] + 1, 1.0/(m+1)) - 1', m=m)
    u_e = interpolate(u_exact, u.function_space())
    import numpy as np
    error = np.abs(u_e.vector().array() -
                   u.vector().array()).max()
    print('error: %.2E' % error)


def test_solver():
    """Perform convergence test with three meshes on solver function."""
    # Choice of nonlinear coefficient
    m = 2

    def q(u):
        return (1+u)**m

    def Dq(u):
        return m*(1+u)**(m-1)

    u_exact = Expression(
        'pow((pow(2, m+1)-1)*x[0] + 1, 1.0/(m+1)) - 1', m=m)
    linear_solver = 'direct'
    errors = []
    for method in 'alg_Newton', 'pde_Newton':
        for J_comp in 'manual', 'automatic':
            for degree in 1, 2, 3:
                error_prev = -1
                for divisions in [(10, 10), (20, 20), (40, 40)]:
                    u = solver(
                        q, Dq, f, divisions, degree,
                        method, J_comp,
                        linear_solver,
                        abs_tol_Krylov=1E-10,
                        rel_tol_Krylov=1E-10,
                        abs_tol_Newton=1E-10,
                        rel_tol_Newton=1E-10)

                    # Find max error
                    u_e = interpolate(u_exact, u.function_space())
                    import numpy as np
                    error = np.abs(u_e.vector().array() -
                                   u.vector().array()).max()
                    # Expect convergence as h**(degree+1)
                    if error_prev > 0:
                        frac = abs(error - error_prev/2**(degree+1))
                        errors.append(frac)
                    error_prev = error
    tol = 4E-5
    for error_reduction in errors:
        assert error_reduction < tol, error_reduction

if __name__ == '__main__':
    #test_solver()
    application_test()
    # Example: manual direct  1   3  4
