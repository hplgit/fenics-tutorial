# wrapper for with extensions to dolfin...
import sys, os
# make sure official dolfin is loaded before the present dolfin.py
# module in os.curdir:
print sys.prefix
print sys.version
dolfin_dir = os.path.join(sys.prefix, 'lib', 'python' + sys.version[:3],
                          'dist-packages')
print dolfin_dir
del sys.path[0]
sys.path.insert(0, dolfin_dir)
import pprint; pprint.pprint(sys.path)
from dolfin import Mesh

# add stuff here
print dir()

