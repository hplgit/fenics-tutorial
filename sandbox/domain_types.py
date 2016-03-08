"""Demonstrate mesh generating objects in DOLFIN."""

from dolfin import *

mesh = UnitInterval(20)
mesh = Interval(20, -1, 1)  # domain [-1,1]
print mesh

mesh = UnitSquare(6, 10)  # 'right' diagonal is default
viz = plot(mesh, title='UnitSquare(6, 10)')
#print 'Add labels:'
#viz.add_point_labels()
#viz.update(mesh)
# The third argument governs directions of diagonals:
mesh = UnitSquare(6, 10, 'left')
plot(mesh, title='UnitSquare(6, 10, "left")')
mesh = UnitSquare(6, 10, 'crossed')
plot(mesh, title='UnitSquare(6, 10, "crossed")')

print mesh

# Domain [0,3]x[0,2] with 6x10 divisions and left diagonals
mesh = Rectangle(0, 0, 3, 2, 6, 10, 'left')
plot(mesh, title='Rectangle(0, 0, 3, 2, 6, 10, "left")')
print mesh

# 6x10x5 boxes in the unit cube, each box gets 6 tetrahedra:
mesh = UnitCube(6, 10, 5)
plot(mesh, title='UnitCube')
print mesh

# Domain [-1,1]x[-1,0]x[-1,2] with 6x10x5 divisions
mesh = Box(-1, -1, -1, 1, 0, 2, 6, 10, 5)
plot(mesh, title='Box(-1, -1, -1, 1, 0, 2, 6, 10, 5)')
print mesh

# 10 divisions in radial directions
mesh = UnitCircle(10, 'crossed', 'maxn')  
plot(mesh, title='UnitCircle(10, "crossed", "maxn")')
mesh = UnitCircle(10, 'right', 'maxn')  
plot(mesh, title='UnitCircle(10, "right", "maxn")')
mesh = UnitCircle(10)  
plot(mesh, title='UnitCircle(10)')
print mesh

mesh = UnitSphere(10)  # 10 divisions in radial directions
plot(mesh, title='UnitSphere(10)')
print mesh


interactive()
