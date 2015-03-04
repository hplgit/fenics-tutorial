
from dolfin import *
import numpy

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary 

class InnerStuff(SubDomain): 
    def inside(self, x, on_boundary):
        if x[0]**2 + x[1]**2  < 0.1: 
            return True  
        return False 

mesh = Mesh("circle.xml.gz") 

W = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)

bc_func = Function(V, ("0.01*x[0] + 0.01*x[1]", "0.01*x[1]" ))
bc = DirichletBC(V, bc_func, DirichletBoundary())

subdomains = MeshFunction("uint", mesh, 0)

innerstuff = InnerStuff()
innerstuff.mark(subdomains, 1)

#plot(subdomains)

coeff_values = numpy.zeros(len(subdomains.values()))
for i in range(len(subdomains.values())):
    val = subdomains.values()[i]
    print val 
    if val == 1: 
        coeff_values[i] = 1.0 
    else:  
        coeff_values[i] = 0.001 

coeff = Function(W)
coeff.vector().set(coeff_values)

plot(coeff)
interactive()


for i in range(0, 20): 

    f = Function(V, ("0.0", "0.0" ))

    a = coeff*dot(grad(v), grad(u))*dx
    L = dot(v,f)*dx

# Compute solution
    problem = VariationalProblem(a, L, bc)
    U = problem.solve()

# Plot solution
#    plot(U)

    mesh.move(U)
    plot(mesh)


interactive()



