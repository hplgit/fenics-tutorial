from __future__ import print_function
from dolfin import *
import numpy as np

class Solver(object):
    def solve(self, problem, linear_solver='direct',
              abs_tol=1E-6, rel_tol=1E-5, max_iter=1000):
        mesh, degree = problem.mesh_degree()
        V = FunctionSpace(mesh, 'Lagrange', degree)
        bcs = [DirichletBC(V, value, boundaries, index)
               for value, boundaries, index
               in problem.Dirichlet_conditions()]

        u = TrialFunction(V)
        v = TestFunction(V)
        p = problem.p_coeff()
        self.p = p  # store for flux computations
        F = inner(p*nabla_grad(u), nabla_grad(v))*dx
        F -= sum([g*v*ds_
                  for g, ds_ in problem.Neumann_conditions()])
        F += sum([r*(u-s)*ds_
                  for r, s, ds_ in problem.Robin_conditions()])
        a, L = lhs(F), rhs(F)

        # Compute solution
        self.u = Function(V)

        if linear_solver == 'Krylov':
            prm = parameters['krylov_solver'] # short form
            prm['absolute_tolerance'] = abs_tol
            prm['relative_tolerance'] = rel_tol
            prm['maximum_iterations'] = max_iter
            solver_parameters = {'linear_solver': 'gmres',
                                 'preconditioner': 'ilu'}
        else:
            solver_parameters = {'linear_solver': 'lu'}

        solve(a == L, self.u, bcs, solver_parameters=solver_parameters)
        return self.u

    def flux(self):
        """Compute and return flux -p*grad(u)."""
        mesh = self.u.function_space().mesh()
        degree = self.u.ufl_element().degree()
        V_g = VectorFunctionSpace(mesh, 'Lagrange', degree)
        self.flux_u = project(-self.p*grad(self.u), V_g)
        self.flux_u.rename('flux(u)', 'continuous flux field')
        return self.flux_u

class Problem(object):
    """Abstract base class for problems."""
    def solve(self):
        self.solver = Solver()
        return self.solver.solve(self)

    def solution(self):
        return solver.u

    def mesh_degree(self):
        """Return mesh, degree."""
        raise NotImpelementedError('Must implement mesh!')

    def p_coeff(self):
        return Constant(1.0)

    def f_rhs(self):
        return Constant(0.0)

    def Dirichlet_conditions(self):
        """Return list of (value,boundary_parts,index) triplets."""
        return []

    def Neumann_conditions(self):
        """Return list of (g,ds(n)) pairs."""
        return []

    def Robin_conditions(self):
        """Return list of (r,u,ds(n)) triplets."""
        return []


class TestProblem1(Problem):
    def init_mesh(self, Nx, Ny):
        """Initialize mesh, boundary parts, and p."""
        self.mesh = UnitSquareMesh(Nx, Ny)

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
        p_values = [1, 1E-3]
        self.p.vector()[:] = np.choose(help, p_values)

    def mesh_degree(self):
        return self.mesh, 2

    def p_coeff(self):
        return self.p

    def f_rhs(self):
        return Constant(0)

    def Dirichlet_conditions(self):
        """Return list of (value,boundary) pairs."""
        return [(1.0, self.boundary_parts, 2),
                (0.0, self.boundary_parts, 3)]

    def Neumann_conditions(self):
        """Return list of g*ds(n) values."""
        return [(0, self.ds(0)), (0, self.ds(1))]

def demo():
    problem = TestProblem1()
    problem.init_meshNx=20, Ny=20)
    problem.solve()
    u = problem.solution()
    plot(u)
    flux_u = problem.solver.flux()
    plot(flux_u)
    interactive()

if __name__ == '__main__':
    demo()
