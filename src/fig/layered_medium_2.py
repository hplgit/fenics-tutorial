from pysketcher import *
from numpy import exp, linspace

W = 10.
H = 10.
a = [0, 5, 10]

drawing_tool.set_coordinate_system(xmin=-1, xmax=W+1,
                                   ymin=-1, ymax=H+1,
                                   axis=False)
drawing_tool.set_linecolor('black')
drawing_tool.set_fontsize(24)

layers =      {'layer%d' % i: Line((0,a[i]), (W,a[i]))
               for i in range(len(a))}
symbols_q =   {'Omega_k%d' % i: Text(r'$\Omega_%d$: $k_%d$' % (i, i),
                                     (W/2,0.5*(a[i]+a[i+1])))
               for i in range(len(a)-1)}

sides ={'left': Line((0,0), (0,H)), 'right': Line((W,0), (W,H))}
d = sides.copy()
d.update(layers)
d.update(symbols_q)
fig = Composition(d)

fig.draw()
drawing_tool.display()
drawing_tool.savefig('tmp2')

raw_input()
