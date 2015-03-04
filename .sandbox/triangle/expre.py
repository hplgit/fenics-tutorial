from prepro2D import *
from math import pi, cos, sin

def layered_rectangle():
    h = 3.
    w = 1.
    # XXX: alternative: give name of point as part of point(x,y,name=None)
    # and then automatically make Line(X, Y,..) have 'XY' as name
    A = point(0,0)
    B = point(w,0)
    C = point(w,h)
    D = point(0,h)
    AB = Line(A, B, division=2, name='AB')
    BC = Line(B, C, division=2, name='BC')
    CD = Line(C, D, division=2, name='CD')
    DA = Line(D, A, division=2, name='DA')
    outer_boundary_parts = [AB, BC, CD, DA]

    E = point(0,h/2)
    F = point(w,h/2)
    G = point(w,3*h/4)
    H = point(0,3*h/4)
    EF = Line(E, F, division=2, name='EF')
    FG = Line(F, G, division=2, name='FG')
    GH = Line(G, H, division=2, name='GH')
    HE = Line(H, E, division=2, name='HE')
    inner_boundary_parts = [EF, FG, GH, HE]

    p, s, m = glue_boundary_parts(outer_boundary_parts)
    # Inner point
    P1 = point(w/2, h/3)
    boundary1 = Boundary(p, s, m, material=1, area=0.1, inner_point=P1)

    p, s, m = glue_boundary_parts(inner_boundary_parts)
    P2 = point(w/2, 5*h/8)
    boundary2 = Boundary(p, s, m, material=2, area=0.01, inner_point=P2)
    #boundary2 = Boundary(p, s, m, material=2, area=0.01, inner_point=(0.5,0.5))

    #print boundary1.dump()
    #print boundary2.dump()
    # Make easyviz sketch: points with text
    plot_boundaries(boundary1, boundary2, marked_points=[
        (A, 'A'), (B, 'B'), (C, 'C'), (D, 'D'), (E, 'E'), (F, 'F'),
        (G, 'G'), (H, 'H'),
        (P1, 'inner1'), (P2, 'inner2')])

    # Make TikZ picture
    latex = ''
    for b in outer_boundary_parts:
        latex += b.tikz()
    for b in inner_boundary_parts:
        latex += b.tikz()
    latex = tikz_environment(latex)
    f = open('tmp' + '.tex', 'w')
    f.write(latex)
    f.close()

    nodes, elems = triangle_output(boundary1, boundary2, maxarea=None, run_triangle=True)
    #nodes, elems = triangle_output(boundary1, maxarea=None, run_triangle=True)
    if nodes > 0:
        print '%d elements with %d nodes' % (elems, nodes)
    coordinates, connectivity, materials = load_triangle_mesh('tmp', '1')
    plot_mesh(coordinates, connectivity, materials)

def circle_in_circle(degrees=90):
    # class Point? p1 + p2, rotate(p1, 45) Vec2D from 1100, or numpy array,
    # sistnevnte har skalarmult innebygd
    A = point(0,0); B = point(2,0)
    AB = Line(A, B, division=2, name='AB')

    BC = Arc(B, center_pt=A, degrees=degrees, division=14, name='BC')

    C = BC.points[-1]
    CA = Line(C, A, 4, name='CA')

    outer_boundary_parts = [AB, BC, CA]

    D = point(1,0)
    AD = Line(A, D, division=2, name='AD')

    DE = Arc(D, center_pt=A, degrees=degrees, division=14, name='DE')

    E = DE.points[-1]
    EA = Line(E, A, division=4, name='EA')

    inner_boundary_parts = [AD, DE, EA]

    p, s, m = glue_boundary_parts(outer_boundary_parts)
    # Inner point
    rad = degrees/180.0*pi/2
    P1 = 0.75*length(B)*point(cos(rad), sin(rad))
    boundary1 = Boundary(p, s, m, material=1, area=0.01, inner_point=P1)

    p, s, m = glue_boundary_parts(inner_boundary_parts)
    P2 = 0.5*length(D)*point(cos(rad), sin(rad))
    boundary2 = Boundary(p, s, m, material=2, area=0.001, inner_point=P2)
    #boundary2 = Boundary(p, s, m, material=2, area=0.01, inner_point=(0.5,0.5))

    #print boundary1.dump()
    #print boundary2.dump()
    # Make easyviz sketch: points with text
    plot_boundaries(boundary1, boundary2, marked_points=[
        (A, 'A'), (B, 'B'), (C, 'C'), (D, 'D'), (E, 'E'),
        (P1, 'inner1'), (P2, 'inner2')])

    # Make TikZ picture
    latex = ''
    for b in outer_boundary_parts:
        latex += b.tikz()
    for b in inner_boundary_parts:
        latex += b.tikz()
    latex = tikz_environment(latex)
    f = open('tmp' + '.tex', 'w')
    f.write(latex)
    f.close()

    nodes, elems = triangle_output(boundary1, boundary2, maxarea=0.1,
                                   run_triangle=False)
    if nodes > 0:
        print '%d elements with %d nodes' % (elems, nodes)


def circle(degrees=90):
    # class Point? p1 + p2, rotate(p1, 45) Vec2D from 1100, or numpy array,
    # sistnevnte har skalarmult innebygd
    A = point(0,0); B = point(2,0)
    AB = Line(A, B, division=2, name='AB')

    BC = Arc(B, center_pt=A, degrees=degrees, division=14, name='BC')

    C = BC.points[-1]
    CA = Line(C, A, 4, name='CA')

    outer_boundary_parts = [AB, BC, CA]

    p, s, m = glue_boundary_parts(outer_boundary_parts)
    # Inner point
    rad = degrees/180.0*pi/2
    P1 = 0.75*length(B)*point(cos(rad), sin(rad))
    boundary1 = Boundary(p, s, m, material=1, area=0.01, inner_point=P1)

    plot_boundaries(boundary1, marked_points=[
        (A, 'A'), (B, 'B'), (C, 'C'), (P1, 'inner1')])

    # Make TikZ picture
    latex = ''
    for b in outer_boundary_parts:
        latex += b.tikz()
    latex = tikz_environment(latex)
    f = open('tmp' + '.tex', 'w')
    f.write(latex)
    f.close()

    nodes, elems = triangle_output(boundary1, maxarea=0.1,
                                   run_triangle=True)
    if nodes > 0:
        print '%d elements with %d nodes' % (elems, nodes)
    coordinates, connectivity, materials = load_triangle_mesh('tmp', '1')
    plot_mesh(coordinates, connectivity, materials)

import sys
#degrees = float(sys.argv[1])
#circle_in_circle(degrees)
#circle(degrees)
layered_rectangle()
