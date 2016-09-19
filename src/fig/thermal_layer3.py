"""Heat conduction problems."""

from pysketcher import *

drawing_tool.set_coordinate_system(
    xmin=-0.3, xmax=1.5, ymin=-0.5, ymax=1.3, axis=False)

drawing_tool.set_linecolor('black')

text1 = Text(r'$u=U_s + A\sin wt$', (0.5,1.05), alignment='center')
text2 = Text(r'$\frac{\partial u}{\partial n}=0$', (-0.17, 0.5), alignment='left')
text3 = Text(r'$\frac{\partial u}{\partial n}=0$', (1.1, 0.5), alignment='left')
text4 = Text(r'$\frac{\partial u}{\partial n}=0$', (0.5, -0.1), alignment='center')

import numpy as np
x = np.linspace(-0.1, 1.1, 101)
y = 1.15 + 0.025*np.sin(2*np.pi/0.2*x)

fig1 = Composition({
    'domain': Rectangle((0,0), 1, 1),
    'subdomain': Rectangle((0.3, 0.3), 0.4, 0.4),
    'y=1': text1, 'x=0': text2, 'x=1': text3, 'y=0': text4,
    'sine': Curve(x,y),
    })

fig1.draw()
drawing_tool.savefig('tmp3')

fig1['subdomain'] = Rectangle((0.3,0.5), 0.4, 0.4)
fig1['y=1'] = Text(r'$u= \sin 2t$', (0.5,1.05), alignment='center')
drawing_tool.erase()
fig1.draw()
drawing_tool.savefig('tmp4')


"""
text1 = Text(r'$u= \sin 2t$', (0.08,1.05), alignment='center')
text2 = Text(r'$\frac{\partial u}{\partial n}=0$', (-0.2, 0.5), alignment='left')
text3 = Text(r'$\frac{\partial u}{\partial n}=0$', (0.2, 0.5), alignment='left')
text4 = Text(r'$\frac{\partial u}{\partial n}=0$', (0.08, -0.1), alignment='center')

x = np.linspace(-0.1, 0.167+0.1, 101)
y = 1.15 + 0.025*np.sin(2*np.pi/0.1*x)

fig2 = Composition({
    'domain': Rectangle((0,0), 0.167, 1),
    'subdomain': Rectangle((0.3*0.167, 0.3), 0.4*0.167, 0.4),
    'x=0': text1, 'x=1': text2, 'y=1': text3, 'y=0': text4,
    'sine': Curve(x,y),
    })
"""

raw_input()
