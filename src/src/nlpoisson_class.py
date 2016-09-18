from __future__ import print_function
from fenics import *
import numpy as np

class NonlinearPoissonSolver(object):
    def __init__(self, problem, method='Picard',
                 J_comp='manual', tol=1E-5,
                 max_iter=25, relaxation_prm=1,
                 debug=False):
        self.problem = problem
        self.method = method
        self.J_comp = J_comp
        self.tol = tol
        self.max_iter = max_iter
        self.omega = relaxation_prm
        self.debug = debug

    def solve(self, linear_solver='direct'):
        """Assemble system, incorporate Dirichlet conditions
        and solve the nonlinear linear system."""
        self.mesh, degree = self.problem.mesh_degree()
        self.V = V = FunctionSpace(self.mesh, 'P', degree)
        Dirichlet_cond = self.problem.Dirichlet_conditions()
        if isinstance(Dirichlet_cond, (Expression)):
            # Just one Expression for Dirichlet conditions on
            # the entire boundary
            self.bcs = [DirichletBC(
                V, Dirichlet_cond,
                lambda x, on_boundary: on_boundary)]
        else:
            # Boundary SubDomain markers
            self.bcs = [
                DirichletBC(V, value, boundaries, index)
                for value, boundaries, index
                in Dirichlet_cond]

        if self.debug:
            # Print the Dirichlet conditions
            print('No of Dirichlet conditions:', len(self.bcs))
            coor = self.mesh.coordinates()
            d2v = dof_to_vertex_map(V)
            for bc in self.bcs:
                bc_dict = bc.get_boundary_values()
                for dof in bc_dict:
                    print('dof %2d: u=%g' % (dof, bc_dict[dof]))
                    if V.ufl_element().degree() == 1:
                        print('   at point %s' %
                              (str(tuple(coor[d2v[dof]].tolist()))))

        self.u = Function(V)             # solution to be computed
        self.u_ = u_ = Function(self.V)  # most recently computed sol.

        if linear_solver == 'Krylov':
            solver_parameters = {'linear_solver': 'gmres',
                                 'preconditioner': 'ilu'}
        else:
            solver_parameters = {'linear_solver': 'lu'}

        # Compute initial guess: solution with q=1
        self.define_variational_problem('initial_guess')
        solve(self.a == self.L, self.u_, self.bcs,
              solver_parameters=solver_parameters)
        #plot(self.u_, interactive=True)

        if self.method.endswith('Newton'):
            # The unknown is now a correction and must have
            # zero in Dirichlet conditions
            if isinstance(Dirichlet_cond, (Expression)):
                self.bcs = [DirichletBC(
                    V, Constant(0.0),
                    lambda x, on_boundary: on_boundary)]
            else:
                self.bcs = [
                    DirichletBC(V, Constant(0.0), boundaries, index)
                    for value, boundaries, index
                    in Dirichlet_cond]

        self.define_variational_problem(self.method)
        num_iter = 0   # number of iterations
        eps = 1.0      # error measure ||u-u_||
        while eps > self.tol and num_iter < self.max_iter:
            num_iter += 1

            solve(self.a == self.L, self.u, self.bcs,
                  solver_parameters=solver_parameters)

            # Relax solution
            if self.method.endswith('Newton'):
                # a=J, L=-F, u=du
                self.u.vector()[:] = \
                  self.u_.vector() + self.omega*self.u.vector()
            elif self.method == 'Picard':
                self.u.vector()[:] = \
                  (1-self.omega)*self.u_.vector() + \
                   + self.omega *self.u.vector()
            # Note: self.u = self.u_ + self.u gives UFL Sum object

            du = self.u.vector().array() - self.u_.vector().array()
            eps = np.linalg.norm(du, ord=np.Inf)
            print('iter=%d, norm of change: %g' % (num_iter, eps))

            self.u_.assign(self.u)
        return self.u

    def define_variational_problem(self, method):
        u_ = self.u_  # short form
        V = self.V
        v = TestFunction(V)
        q = self.problem.q_func
        Dq = self.problem.Dq_func
        f = self.problem.f_rhs()

        if method == 'alg_Newton':
            u = TrialFunction(V)
            F = dot(q(u)*grad(u), grad(v))*dx
            F -= f*v*dx
            F -= sum([g*v*ds_ for g, ds_ in
                      self.problem.Neumann_conditions()])
            F = action(F, u_)
            # J must be a Jacobian (Gateaux deriv. in direction of du)
            if self.J_comp == 'manual':
                J = dot(q(u_)*grad(u), grad(v))*dx + \
                    dot(Dq(u_)*u*grad(u_), grad(v))*dx
            else:
                J = derivative(F, u_, u)
            self.a, self.L = J, -F

        elif method == 'pde_Newton':
            du = TrialFunction(V)
            F  = dot(q(u_)*grad(u_), grad(v))*dx
            F -= f*v*dx
            F -= sum([g*v*ds_ for g, ds_ in
                      self.problem.Neumann_conditions()])
            if self.J_comp == 'manual':
                J = dot(q(u_)*grad(du), grad(v))*dx + \
                    dot(Dq(u_)*du*grad(u_), grad(v))*dx
            else:
                J = derivative(F, u_, du)
            self.a, self.L = J, -F

        elif method == 'Picard':
            u = TrialFunction(V)
            F  = dot(q(u_)*grad(u), grad(v))*dx
            F -= f*v*dx
            F -= sum([g*v*ds_ for g, ds_ in
                      self.problem.Neumann_conditions()])
            self.a, self.L = lhs(F), rhs(F)

        elif method == 'initial_guess':
            u = TrialFunction(V)
            F  = dot(grad(u), grad(v))*dx
            F -= f*v*dx
            F -= sum([g*v*ds_ for g, ds_ in
                      self.problem.Neumann_conditions()])
            self.a, self.L = lhs(F), rhs(F)
        else:
            raise ValueError('method=%s is illegal' % method)

    def flux(self):
        """Compute and return flux -p*grad(u)."""
        mesh = self.u.function_space().mesh()
        degree = self.u.ufl_element().degree()
        V_g = VectorFunctionSpace(mesh, 'P', degree)
        self.flux_u = project(-self.p*grad(self.u), V_g)
        self.flux_u.rename('flux(u)', 'continuous flux field')
        return self.flux_u

class NonlinearPoissonProblem(object):
    """Abstract base class for problems."""
    def solve(self,
              method='Picard',
              J_comp='manual',
              tol=1E-5,
              max_iter=25, relaxation_prm=1,
              debug=False,
              linear_solver='direct',
              abs_tol_Krylov=1E-6, rel_tol_Krylov=1E-5,
              max_iter_Krylov=1000):
        self.solver = NonlinearPoissonSolver(
            self, method=method, J_comp=J_comp, max_iter=max_iter,
            relaxation_prm=relaxation_prm, debug=debug, tol=tol)
        prm = parameters['krylov_solver'] # short form
        prm['absolute_tolerance'] = abs_tol_Krylov
        prm['relative_tolerance'] = rel_tol_Krylov
        prm['maximum_iterations'] = max_iter_Krylov
        return self.solver.solve(linear_solver)

    def solution(self):
        return self.solver.u

    def mesh_degree(self):
        """Return mesh, degree."""
        raise NotImplementedError('Must implement mesh!')

    def q_func(self, u):
        return Constant(1.0)

    def Dq_func(self, u):
        return Constant(0.0)

    def f_rhs(self):
        return Constant(0.0)

    def Dirichlet_conditions(self):
        """Return list of (value,boundary_parts,index) triplets,
        or an Expression (if Dirichlet values only)."""
        return []

    def Neumann_conditions(self):
        """Return list of (g,ds(n)) pairs."""
        return []


class TestProblem(NonlinearPoissonProblem):
    def __init__(self, Nx, Ny, Nz=None, m=2):
        """Initialize mesh, boundary parts."""
        if Nz is None:
            self.mesh = UnitSquareMesh(Nx, Ny)
        else:
            self.mesh = UnitCubeMesh(Nx, Ny, Nz)
        self.m = 2
        self.u0 = Expression(
            'pow((pow(2, m+1)-1)*x[0] + 1, 1.0/(m+1)) - 1', m=self.m)
        from heat_class import mark_boundaries_in_hypercube
        self.boundary_parts = \
             mark_boundaries_in_hypercube(self.mesh, d=2)
        self.ds =  Measure(
            'ds', domain=self.mesh,
            subdomain_data=self.boundary_parts)

    def mesh_degree(self):
        return self.mesh, 1

    def q_func(self, u):
        return (1+u)**self.m

    def Dq_func(self, u):
        return self.m*(1+u)**(self.m-1)

    def Dirichlet_conditions(self):
        #return self.u0
        return [(0.0, self.boundary_parts, 0),
                (1.0, self.boundary_parts, 1)]


def test_NonlinearPoissonSolver():
    """Convergence rate tests of manufactured solution."""

    errors = []
    linear_solver = 'direct'
    for method in 'Picard', 'alg_Newton', 'pde_Newton':
        for J_comp in 'manual', 'automatic':
            for degree in 1, 2, 3:
                error_prev = -1
                for Nx, Ny in [(10, 10), (20, 20), (40, 40)]:
                    problem = TestProblemExact(Nx=Nx, Ny=Ny, m=2)
                    problem.solve(
                        method=method,
                        J_comp=J_comp,
                        tol=1E-8,
                        max_iter=25, relaxation_prm=1,
                        abs_tol_Krylov=1E-8,
                        rel_tol_Krylov=1E-8,
                        debug=False,
                        linear_solver=linear_solver)
                    u = problem.solution()
                    u_e = interpolate(problem.u0, u.function_space())
                    error = np.abs(u_e.vector().array() -
                                   u.vector().array()).max()
                    # Expect convergence as h**(degree+1)
                    if error_prev > 0:
                        frac = abs(error - error_prev/2**(degree+1))
                        errors.append(frac)
                    error_prev = error
    #print('errors:', errors)
    tol = 4E-3
    for error_reduction in errors:
        assert error_reduction < tol, error_reduction

if __name__ == '__main__':
    #demo()
    test_NonlinearPoissonSolver()
