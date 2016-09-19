"""Heat conduction problems."""

from pysketcher import *

drawing_tool.set_coordinate_system(
    xmin=-0.5, xmax=1.5, ymin=-0.5, ymax=1.5, axis=False)

drawing_tool.set_linecolor('black')

domain = Rectangle((0,0), 1, 1)
subdomain = Rectangle((0.3, 0.3), 0.4, 0.4)
text1 = Text('$u=1$', (-0.2,0.5), alignment='left')
text2 = Text('$u=0$', (1.2,0.5), alignment='right')
text3 = Text(r'$\partial u/\partial n=0$', (0.5, 1.05), alignment='center')
text4 = Text(r'$\partial u/\partial n=0$', (0.5, -0.1), alignment='center')

fig1 = Composition({
    'domain': domain,
    'x=0': text1, 'x=1': text2, 'y=1': text3, 'y=0': text4,
    })

fig1.draw()
drawing_tool.display()
drawing_tool.savefig('tmp1')

drawing_tool.erase()
fig1['subdomain'] = subdomain
fig1['x=1'] = Text(r'$\partial u/\partial n = 0$',
                   (1.3,0.5), alignment='right')

fig1.draw()
drawing_tool.savefig('tmp2')

raw_input()
