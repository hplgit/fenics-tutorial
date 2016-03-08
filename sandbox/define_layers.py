from dolfin import *
import sys, math, numpy

# Create mesh
degree = int(sys.argv[1])
divisions = [int(arg) for arg in sys.argv[2:]]
d = len(divisions)
domain_type = [UnitInterval, UnitSquare, UnitCube]
mesh = domain_type[d-1](*divisions)

# Define layer boundaries and k values in the layers
L = [0, 0.2, 0.6, 0.7, 1]
k = [1, 10, 100, 20]
s = len(L)-1

# Check consistency: increasing L elements
for i in range(1, s):
    if L[i] <= L[i-1]:
        raise ValueError('L=%s does not have increasing elements' % L)
        
# Check consistency (layer boundaries coincide with element
# boundaries in the x direction)
dx = 1.0/(divisions[0])
for i in range(1, s):
     D = L[i] - L[i-1]
     if abs(D/dx - round(D/dx)) > 1E-14:  # is D/dx an integer?
         raise ValueError('L=%s does not coincide '\
                          'with element boundaries' % L)

class Material(SubDomain):
    """Define material (subdomain) no. i."""
    def __init__(self, subdomain_number, subdomain_boundaries):
        self.number = subdomain_number
        self.boundaries = subdomain_boundaries
        SubDomain.__init__(self)
        
    def inside(self, x, on_boundary):
        i = self.number
        L = self.boundaries
        if L[i] <= x[0] <= L[i+1]:  # important with <=
            print 'x=%g is in subdomain %d' % (x[0],i)
            return True
        else:
            return False

cell_entity_dim = mesh.topology().dim()  # = d
subdomains = MeshFunction('uint', mesh, cell_entity_dim)
subdomains.rename('subdomains', 'markers for different materials')
# Mark subdomains with numbers i=0,1,...,s=len(L)-1
for i in range(s):
    print 'marking for i:', i
    material_i = Material(i, L)
    material_i.mark(subdomains, i)
print 'subdomains:', subdomains.values()

# Save mesh and subdomains to file
file = File('hypercube_mesh.xml.gz')
file << mesh
file = File('layers.xml.gz')
file << subdomains

# Write a module with the analytical solution:
f = open('u_layered.py', 'w')
f.write("""
import numpy
L = numpy.array(%s, float)
k = numpy.array(%s, float)
s = len(L)-1

def u_exact(x):
    # First find which subdomain x0 is located in
    for i in range(len(L)-1):
        if L[i] <= x <= L[i+1]:
            break

    # Vectorized implementation of summation:
    s2 = sum((L[1:s+1] - L[0:s])*(1.0/k[:]))
    if i == 0:
        u = (x - L[i])*(1.0/k[0])/s2
    else:
        s1 = sum((L[1:i+1] - L[0:i])*(1.0/k[0:i]))
        u = ((x - L[i])*(1.0/k[i]) + s1)/s2
    return u

if __name__ == '__main__':
    # Plot the exact solution
    from scitools.std import linspace, plot, array
    x = linspace(0, 1, 101)
    u = array([u_exact(xi) for xi in x])
    print u
    plot(x, u)
""" % (L, k))
f.close()


