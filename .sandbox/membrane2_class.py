"""
As membrane2.py, but the code is separated into Solver and
Problem classes.
"""

from dolfin import *

class Problem:
    def __init__(self, T=10.0, A=1.0, R=0.3, theta=0.2, 
                 sigma=0.2, ksi=0.6, ncells=40):
        self.set_prm(T, A, R, theta, sigma, ksi, ncells)

    def load_from_file(self, filename):
        pass

    def set_prm(self, T=10.0, A=1.0, R=0.3, theta=0.2, 
                sigma=50, ksi=0.6, ncells=40):
        self.T = T
        self.A = A
        self.R = R
        self.theta = theta
        self.sigma = sigma
        self.ncells = ncells
        # derived:
        self.x0 = ksi*R*cos(theta)
        self.y0 = ksi*R*sin(theta)
        self.domain_mesh = UnitCircle(self.ncells)
        if self.ncells < 3000:
            linear_solver = 'lu'
            preconditioner = None
        else:
            linear_solver = 'cg'
            preconditioner = 'ilu'
        element_degree = 1
        self.solver = Solver(self, element_degree, 
                             linear_solver, preconditioner)

    # most FEniCS Problem classes has a mesh function:
    def mesh(self):
        return self.domain_mesh

    def boundary_conditions(self, V):
        """Return list of DirichletBC objects."""
        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(V, Constant(0.0), boundary)
        return [bc]

    def pressure_expression(self):
        """Return Expression for pressure function."""
        formula = '4*exp(-0.5*(pow((R*x[0] - x0)/sigma, 2)) '\
                  '     - 0.5*(pow((R*x[1] - y0)/sigma, 2)))'
        return Expression(formula, R=self.R, x0=self.x0,
                          y0=self.y0, sigma=self.sigma)

    def scale_back(self, w):
        """Return A*R**2*w/(8*pi*T)."""
        return self.A**self.R**2*w/(8*pi*self.sigma*self.T)

    def solve(self):
        self.solver.setup()
        self.solver.solve()
        self.solver.compute_derived_quantities()
        self.w, self.max_w = self.solver.w, self.solver.max_w
        self.energy_w = self.solver.E_w
        # Find unscaled quantities
        self.D = Function(self.solver.V)
        self.D.vector()[:] = self.scale_back(self.w.vector().array())
        self.max_D = self.scale_back(self.max_w)
        scaling_derivative = self.A*self.R/(8*pi*self.sigma)
        self.energy_D = scaling_derivative**2*self.energy_w


# Alternative, with parameters in a dictionary (NOT READY!)
class Problem2:
    _parameters = {
        'T': dict(value=10.0, help='tension', unit='Pa'),
        'A': dict(value=1.0, help='pressure amplitude', unit='Pa/m**2'),
        'R': dict(value=0.3, help='radius of domain', unit='m'),
        'theta': dict(value=0.2, help='polar coord. angle of pressure location'),
        'ksi': dict(value=0.6, help='fraction of radial coord. of pressure location', unit='m'),
        'sigma': dict(value=0.025, help='st.dev. of pressure bell function', unit='m'),
        }

    def __init__(self, prm={}):
        self.prm = prm
        self.setup(prm)

    def setup(self, p):
        self.prm.update(p)

        # Make local variables without self.prm[] syntax
        variables = 'ksi R theta ncells'.split()
        for v in variables:
            exec('%s = %s' % (v, 'self.prm["%s"]' % v))

        # Derived quantities:
        self.x0 = ksi*R*cos(theta)
        self.y0 = ksi*R*sin(theta)
        self.domain_mesh = UnitCircle(n)
        if ncells < 3000:
            linear_solver = 'lu'
            preconditioner = None
        else:
            linear_solver = 'cg'
            preconditioner = 'ilu'
        self.solver = Solver(self, 1, linear_solver, preconditioner)

    # most FEniCS Problem classes has a mesh function:
    def mesh(self):
        return self.domain_mesh

    def boundary_conditions(self, V):
        """Return list of DirichletBC objects."""
        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(V, Constant(0.0), boundary)
        return [bc]

    def pressure_expression0(self):
        """Return Expression for pressure function."""
        R, x0, y0, sigma = self.prm['R'], self.prm['x0'], \
            self.prm['y0'], self.prm['sigma']  # short forms

        pressure = '4*exp(-0.5*(pow((%g*x[0] - %g)/%g, 2)) '\
            '     - 0.5*(pow((%g*x[1] - %g)/%g, 2)))' % \
            (R, x0, sigma, R, y0, sigma)
        return Expression(pressure)

    def pressure_expression(self):
        """Return Expression for pressure function."""
        # Cooler implementation
        pressure = '4*exp(-0.5*(pow((%(R)g*x[0] - %(x0)g)/%(sigma)g, 2)) '\
            '     - 0.5*(pow((%(R)g*x[1] - %(y0)g)/%(sigma)g, 2)))' % \
            vars(self.prm)
        return Expression(pressure)

    def solve(self):
        self.solver.solve()



class Solver:
    """Solve a scaled membrane problem."""
    def __init__(self, 
                 problem,
                 element_degree=1,
                 linear_solver='lu', 
                 preconditioner=None):
        self.problem = problem
        self.set_prm(element_degree, linear_solver, preconditioner)

    def set_prm(self,
                element_degree=1,
                linear_solver='lu', 
                preconditioner=None):
        self.element_degree = element_degree
        self.linear_solver = linear_solver
        self.preconditioner = preconditioner

    def setup(self):
        self.mesh = self.problem.mesh()
        self.V = V = FunctionSpace(self.mesh, 'Lagrange', 
                                   self.element_degree)

        # Define variational problem
        w = TrialFunction(V)
        v = TestFunction(V)
        f = self.problem.pressure_expression()
        self.F = inner(nabla_grad(w), nabla_grad(v))*dx - p*v*dx
        self.a = lhs(self.F)
        self.L = rhs(self.F)
        self.w = Function(V)  # solution

    def solve(self):
        A = assemble(self.a)
        b = assemble(self.L)
        bcs = self.problem.boundary_conditions(self.V)
        for bc in bcs:
            bc.apply(A, b)
        if self.linear_solver == 'lu':
            solve(A, self.w.vector(), b, 'lu')
        else:
            solve(A, self.w.vector(), b, 
                  self.linear_solver, self.preconditioner)

        self.compute_derived_quantities()

    def compute_derived_quantities(self):
        # Find maximum deflection
        self.max_w = self.w.vector().array().max()

        # Compute elastic energi: integral of T*abs(grad(D))^2
        w = self.w  # short form for nicer formula below
        E_functional = 0.5*inner(grad(w), grad(w))*dx
        self.E_w = assemble(E_functional)



def verify(self):
    problem = Problem(sigma=60)  # almost flat pressure
    problem.solve()

    w_exact = Expression('1 - x[0]*x[0] - x[1]*x[1]')
    v = pproblem.solver.V
    w_e = interpolate(w_exact, V)
    w_e_array = w_e.vector().array()
    w_array = problem.w.vector().array()
    diff_array = w_e_array - w_array
    difference = Function(V)
    difference.vector()[:] = diff_array
    import numpy
    return numpy.abs(verify_error.vector().array()).abs()


class Viz:
    def __init__(self, problem):
        self.problem = problem
        self.solver  = problem.solver

    def interactive_viz(self):
        self.viz_w = plot(self.problem.w,
                          wireframe=False,
                          title='Scaled membrane deflection',
                          rescale=False,
                          axes=True,)
        self.viz_w.elevate(-65) # tilt camera -65 degrees (latitude dir)
        self.viz_w.set_min_max(0, 0.5*self.solver.max_w)
        self.viz_w.update(self.solver.w)
        self.viz_w.write_png('membrane_deflection.png')
        self.viz_w.write_ps('membrane_deflection', format='eps')

        f = interpolate(self.problem.pressure_expression(), 
                        self.solver.V)
        self.viz_p = plot(f, title='Scaled pressure')
        self.viz_p.elevate(-65)
        self.viz_p.update(p)
        self.viz_p.write_png('pressure.png')
        self.viz_p.write_ps('pressure', format='eps')
        
        viz_m = plot(self.solver.mesh, title='Finite element mesh')

    def batch_viz(self):
        #os.system('paraview ....')
        pass



def main():
    problem = Problem(sigma=50)
    problem.solve()
    visualizer = Viz(problem)
    visualizer.interactive_viz()
    print 'Maximum real deflection is', problem.max_D
    print 'Elastic energy:', problem.energy_D
    # Should be at the end
    interactive()

if __name__ == '__main__':
    main()
