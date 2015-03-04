# From image to mesh could use matplotlib/PIL:
# http://matplotlib.sourceforge.net/users/image_tutorial.html

import math, sys, os
import numpy as np
from scitools.std import float_eq, plot, hold, figure, text, axis, title

def point(x, y):
    print x, y
    return np.array([x, y])

def length(p):
    return np.sqrt(p[0]**2 + p[1]**2)

class BoundaryPart:
    """
    Superclass for boundary parts. Subclass implement specfic
    types of parts such as Line, Arc, etc.
    """
    def __init__(self, division, boundary_marker=0, stretching=1,
                 name=None):
        """
        =================  ====================================================
        Argument           Description
        =================  ====================================================
        division           no of intervals (points-1) along boundary part
        boundary_marker    integer for numbering parts of the boundary
        stretching         real parameter for stretching points along the
                           boundary part: > 1 stretches toward the end,
                           and < 1 stretches toward the start
        name               name of the end points of the part (string),
                           typically 'CD' or 'P2P3', reflecting start point
                           'C' or 'P2' and end point 'D' or 'P3'
        =================  ====================================================
        """
        self.bc = boundary_marker
        self.np = division
        self.stretching = stretching
        self.name = name
        if isinstance(self.name, str):
            if len(self.name) == 2:
                self.name_of_start_pt = name[0]
                self.name_of_end_pt = name[1]
            elif len(self.name) == 4:
                if self.name[1].isdigit() and self.name[3].isdigit():
                    self.name_of_start_pt = name[0] + '_' + name[1]
                    self.name_of_end_pt = name[2] + '_' + name[3]

        self.points = []

    def stretch(self, t):
        """Stretch a parameter vector in [0,1] according to self.stretching."""
        if self.stretching > 0:
            return t**self.stretching      # stretching toward t=1
        else:
            return 1 - (1-t)**abs(self.stretching)  # stretching toward t=0

    def _length(self, vec):
        """Return length of 2D vector vec."""
        return math.sqrt(vec[0]**2 + vec[1]**2)

    def segments_from_indices(self, first_index, last_index):
        segments = [(i,i+1) for i in range(first_index, last_index)]
        return segments

    def generate_polygon(self):
        raise NotImplementedError(self.__class__.__name__)

    def tikz(self):
        """Write out TikZ commands for drawing self.points."""
        latex = '\n%% %s\n' % str(self)  # pretty print of object
        tikz_points = ['(%g,%g)' % (x, y) for x, y in self.points]
        latex += 'r\draw' + ' -- '.join(tikz_points)
        if hasattr(self, 'name_of_start_pt'):
            latex += '\n' + r'\path (%g,%g) node (%s) {$%s$}' % \
                     (self.points[0][0], self.points[0][1],
                      self.name_of_start_pt,
                      self.name_of_start_pt)
        if hasattr(self, 'name_of_end_pt'):
            latex += '\n' + r'\path (%g,%g) node (%s) {$%s$}' % \
                     (self.points[-1][0], self.points[-1][1],
                      self.name_of_end_pt,
                      self.name_of_end_pt)
        return latex

def tikz_environment(tikz_text, grid={}, scale=1.0):
    # grid = {'spacing':1.0, 'color': 'gray', 'lower': (0,0), 'upper': (1,1)}
    latex = r"""
\documentclass{article}
\usepackage{tikz}
\usepackage{pgflibararyarrows}
\usepackage{pgflibararysnakes}
\begin{document}
"""
    latex += r"""
\begin{tizpicture}[scale=%g]
""" % scale
    if grid:
        latex += r"""\
\draw [step=%gcm,color=%s] %s grid %s
""" % (grid.get('spacing', 0.25), grid.get('color', 'gray'),
       tuple(grid.get('lower', (0,0))), tuple(grid.get('upper', (0,0))))
    latex += tikz_text
    latex += r"""
\end{tikzpicture}

\end{document}
"""
    return latex


class UserCurve(BoundaryPart):
    def __init__(self, x, y, boundary_marker=0, stretching=1, name=None):
        assert x.size == y.size
        division = x.size - 1
        BoundaryPart.__init__(self, division, boundary_marker, stretching,
                              name)
        self.points = np.asarray([(x_, y_) for x_, y_ in zip(x, y)])
        self.segments = self.segments_from_indices(0, len(self.points)-1)
        self.markers = [self.bc]*len(self.segments)  # segment markers
        if stretching != 1:
            raise ValueError('stretching=%g != 1 is not possible for UserCurve' % stretching)

    def generate_polygon(self):
        line_vector = self.end_pt - self.start_pt
        # parameter t in [0,1]
        t_values = np.linspace(0, 1, self.np+1)
        t_values = self.stretch(t_values)
        self.points = [self.start_pt + t*line_vector for t in t_values]
        self.segments = self.segments_from_indices(0, len(self.points)-1)
        self.markers = [self.bc]*len(self.segments)  # segment markers

    def __str__(self):
        return 'UserCurve(x=[...], y=[...], boundary_marker=%s, '\
               'stretching=%s, name=%s)' % \
               (self.bc, self.stretching, self.name)

    def __repr__(self):
        return str(self)  # call self.__str__()


class Line(BoundaryPart):
    def __init__(self, start_pt, end_pt, division, boundary_marker=0,
                 stretching=1, name=None):
        BoundaryPart.__init__(self, division, boundary_marker, stretching,
                              name)
        self.start_pt = np.array(start_pt, float)
        self.end_pt = np.array(end_pt, float)
        self.generate_polygon()

    def generate_polygon(self):
        line_vector = self.end_pt - self.start_pt
        # parameter t in [0,1]
        t_values = np.linspace(0, 1, self.np+1)
        t_values = self.stretch(t_values)
        self.points = [self.start_pt + t*line_vector for t in t_values]
        self.segments = self.segments_from_indices(0, len(self.points)-1)
        self.markers = [self.bc]*len(self.segments)  # segment markers

    def __str__(self):
        return 'Line(start_pt=%s, end_pt=%s, division=%s, boundary_markers=%s, '\
               'stretching=%s, name=%s)' % \
               (tuple(self.start_pt), tuple(self.end_pt),
                self.np, self.np, self.stretching,
                self.name)

    def __repr__(self):
        return str(self)  # call self.__str__()


class EllipticalArc(BoundaryPart):
    def __init__(self, start_pt, center_pt, degrees, division,
                 other_axis=None, boundary_marker=0, stretching=1,
                 name=None):
        BoundaryPart.__init__(self, division, boundary_marker, stretching,
                              name)
        self.start_pt = np.array(start_pt, float)
        self.center_pt = np.array(center_pt, float)
        self.other_axis = other_axis
        self.degrees = float(degrees)
        self.generate_polygon()

    def generate_polygon(self):
        # local coordinate system:
        # stretching
        e1 = self.start_pt - self.center_pt
        e2 = np.array([-e1[1], e1[0]])  # ortogonal to e1
        R1 = self._length(e1)
        # Circle or ellipse?
        R2 = self.other_axis if self.other_axis is not None else R1

        cos_theta = np.dot(e1, np.array([1, 0]))/(R1*1)
        sin_theta = math.sqrt(1 - cos_theta**2)
        transformation_matrix = np.array([[ cos_theta, sin_theta],
                                          [-sin_theta, cos_theta]])

        degrees = self.degrees/360.0*2*math.pi

        # parameter t in [0,1]
        t_values = np.linspace(0, 1, self.np+1)
        t_values = self.stretch(t_values)
        self.points = []
        for t in t_values:

            phi = t*degrees
            # Generate point in local coordinate system (e1, e2)
            pt = np.array([R1*math.cos(phi), R2*math.sin(phi)])
            pt = self.center_pt + np.dot(transformation_matrix, pt)
            #print 't=', t
            #print pt
            #print np.dot(transformation_matrix, pt)
            self.points.append(pt)
            #print 'Arc:', pt, R1, R2, self.center_pt
        self.segments = self.segments_from_indices(0, len(self.points)-1)
        self.markers = [self.bc]*len(self.segments)  # segment markers

    def __str__(self):
        return 'EllipticalArc(start_pt=%s, center_pt=%s, degrees=%s, '\
               'division=%s, other_axis=%s, boundary_marker=%s, '\
               'stretching=%s, name=%s)' % \
               (tuple(self.start_pt), tuple(self.center_pt),
                self.degrees, self.np, self.other_axis,
                self.bc, self.stretching,
                self.name)

    def __repr__(self):
        return str(self)  # call self.__str__()


class Arc(EllipticalArc):
    def __init__(self, start_pt, center_pt, degrees, division,
                 boundary_marker=0, stretching=1, name=None):
        EllipticalArc.__init__(self, start_pt, center_pt, degrees,
                               division, other_axis=None,
                               boundary_marker=boundary_marker,
                               stretching=stretching, name=name)

    def __str__(self):
        return 'Arc(start_pt=%s, center_pt=%s, degrees=%s, division=%s, '\
               'boundary_marker=%s, stretching=%s, name=%s)' % \
               (tuple(self.start_pt), tuple(self.center_pt),
                self.degrees, self.np,
                self.bc, self.stretching,
                self.name)

    def __repr__(self):
        return str(self)  # call self.__str__()


def glue_boundary_parts(boundary_parts):
    points = []
    segments = []
    markers = []
    point_no = 0
    last = False
    for i in range(len(boundary_parts)):
        this_part = boundary_parts[i]
        if i == len(boundary_parts)-1:  # last part?
            next_part = boundary_parts[0]
            last = True
        else:
            next_part = boundary_parts[i+1]
        if float_eq(this_part.points[-1], next_part.points[0]):
            if not last:
                points.extend(this_part.points[:-1])  # add last from next part
            else:
                points.extend(this_part.points[:])

            segments.extend([[point_no + i1, point_no + i2] for i1, i2 in
                           this_part.segments])
            markers.extend(this_part.markers)

            if last:
                segments[-1][1] = segments[0][0]

            point_no += len(this_part.points[:-1])

    return points, segments, markers


class Boundary:
    """
    Representation of a closed 2D boundary.
    """

    def __init__(self, points, segments, markers, material=None, area=None,
                 hole=False, inner_point=None, name=None, zero_tol=1E-15):
        """
        =========== ==========================================================
        Argument    Description
        =========== ==========================================================
        points      list of ordered (x,y) points defining the boundary
        segments    list of ordered segments, connecting two consequtive
                    points along the boundary
        markers     list of boundary marker numbers for each segment
        material    material/subdomain identifier (int)
        area        measure of the largest triangle inside the boundary
        hole        True if the boundary encloses a hole in the domain
        inner_point point in the domain surrounded by the boundary (and used
                    by Triangle for generating holes, as well as area
                    and material/subdomain specifications.
        name        logical name of the boundary
        zero_tol    numbers less than zero_tol are replaced by 0
        =========== ==========================================================
        """
        self.points, self.segments, self.markers = points, segments, markers
        self.material = material
        self.area = area
        self.hole = hole
        self.inner_point = inner_point
        self.name = name
        self.points = np.asarray(self.points)
        # Without stripping off small values, triangle may stop generating mesh
        self.points[np.abs(self.points) < zero_tol] = 0
        self.x = self.points[:,0]  # all x coordinates of points
        self.y = self.points[:,1]  # all y coordinates of points
        self.point_offset = 0
        self.segment_offset = 0

    def dump(self):
        text = 'boundary "%s":' if self.name else ''
        text += 'points: %s' % ''.join(['  (%g,%g)\n' % (x, y) for x, y in self.points])
        text += 'segments: %s' % ''.join(['  %s\n' % s for s in self.segments])
        text += 'markers: %s' % ''.join(['  %s\n' % m for m in self.markers])
        text += 'area=%s, hole=%s, inner_point=%s' % \
                (self.area, self.hole, self.inner_point)
        return text

    def set_offsets(self, point_offset, segment_offset):
        self.point_offset = point_offset
        self.segment_offset = segment_offset

    def __str__(self):
        return 'Boundary: %d points, %d segments' %\
               (len(self.points.shape[0]), len(self.segments))

    def __repr__(self):
        return str(self)

    def iterpoints(self):
        """Return global point number, x and y coordinates."""
        for i in range(self.points.shape[0]):
            yield i + self.point_offset, self.x[i], self.y[i]

    def iterpoints2(self):
        """Return global point number, x and y coordinates, length."""
        n = self.points.shape[0]
        for i in range(n):
            if i < n-1:
                L = length(self.points[i+1] - self.points[i])
            else:
                L = length(self.points[i] - self.points[i-1])  # last point
            yield i + self.point_offset, self.x[i], self.y[i], L

    def itersegments(self):
        """Return global segment number, start and end point, and marker."""
        for i, s in enumerate(self.segments):
            yield i + self.segment_offset, \
                  s[0] + self.point_offset, s[1] + self.point_offset, \
                  self.markers[i]

def plot_boundaries(outer_boundary, inner_boundaries=[], marked_points=None):
    if not isinstance(inner_boundaries, (tuple,list)):
        inner_boundaries = [inner_boundaries]
    boundaries = [outer_boundary]
    boundaries.extend(inner_boundaries)

    # Find max/min of plotting area
    plot_area = [min([b.x.min() for b in boundaries]),
                 max([b.x.max() for b in boundaries]),
                 min([b.y.min() for b in boundaries]),
                 max([b.y.max() for b in boundaries])]

    aspect = (plot_area[3] - plot_area[2])/(plot_area[1] - plot_area[0])
    for b in boundaries:
        plot(b.x, b.y, daspect=[aspect,1,1], daspectratio='manual')
        hold('on')
    axis(plot_area)
    title('Specification of domain with %d boundaries' % len(boundaries))
    if marked_points:
        for pt, name in marked_points:
            text(pt[0], pt[1], name)

def triangle_output(outer_boundary, inner_boundaries=[], casename='tmp',
                    maxarea=None, run_triangle=True):
    if isinstance(inner_boundaries, Boundary):
        inner_boundaries = [inner_boundaries]  # wrap in list

    f = open(casename + '.poly', 'w')
    # Set offsets
    point_offset = len(outer_boundary.points)
    segment_offset = len(outer_boundary.segments)
    for inner_boundary in inner_boundaries:
        inner_boundary.set_offsets(point_offset, segment_offset)
        point_offset += inner_boundary.points.shape[0]
        segment_offset += len(inner_boundary.segments)


    num_vertices = point_offset
    num_segments = segment_offset
    dimension = 2
    num_attributes = 0
    num_boundary_markers = 0
    boundaries = [outer_boundary]
    boundaries.extend(inner_boundaries)

    f.write("""
#
# Declare %(num_vertices)d vertices in dimension %(dimension)d with 0 attributes and 0 boundary markers
#
%(num_vertices)d   %(dimension)d   0  0\n""" % vars())
    for bno, boundary in enumerate(boundaries):
        if bno == 0:
            f.write("""\
#
# Outer boundary with %d points
#
# point no.        x            y
#
""" % (boundary.points.shape[0]))
        else:
            f.write("""\
#
# Inner boundary with %d points
#
# point no.        x            y
""" % (boundary.points.shape[0]))

        for vertex_no, x, y in boundary.iterpoints():
            f.write("%(vertex_no)6d       %(x)12.6E %(y)12.6E\n" % vars())

    f.write("""\
#
# Declare %(num_segments)d segments and %(num_boundary_markers)d boundary markers
#
%(num_segments)d %(num_boundary_markers)d\n""" % vars())
    for bno, boundary in enumerate(boundaries):
        if bno == 0:
            f.write("""\
#
# Outer boundary with %d segments
#
# segm.no.   start-pt    end-pt        boundary marker
""" % (len(boundary.segments)))
        else:
            f.write("""\
#
# Inner boundary with %d segments
#
# segm.no.   start-pt    end-pt        boundary marker
""" % (len(boundary.segments)))

        for segment_no, start, end, marker in boundary.itersegments():
            f.write("%(segment_no)6d         %(start)4d       %(end)4d             %(marker)2d\n" % vars())

    num_holes = sum([1 for b in inner_boundaries if b.hole])
    f.write("""\
#
# There are %(num_holes)d holes
%(num_holes)d
""" % vars())
    if num_holes:
        for i, boundary in enumerate(inner_boundaries):
            if boundary.hole:
                x, y = boundary.inner_point
            f.write("""\
# Point inside hole %(i)d:
%(i)d  %(x)g %(y)g\n""" % vars())

    # Count how many boundaries that have the material and/or area set.
    # Note that we demand *both* the material number *and* the area to be set.
    num_regions = sum([1 for b in boundaries \
                       if b.area is not None and b.material is not None])
    if num_regions:
        f.write("""\
#
# Regional attributes and area constraints
#
%(num_regions)d
#
""" % vars())
        for i, boundary in enumerate(boundaries):
            if boundary.area is not None and boundary.material is not None:
                try:
                    x, y = boundary.inner_point
                    material = boundary.material
                    area = boundary.area
                except TypeError:
                    print 'boundary %s of type %s has area=%s and '\
                          'material=%, but inner_point=%s' % \
                          (boundary.name if boundary.name else '',
                           boundary.__class__.__name__,
                           boundary.area, boundary.material,
                           boundary.inner_point)
                    sys.exit(1)
                f.write("""\
# Region no. %(i)d has attribute (material) %(material)d and area %(area)g
%(i)d  %(x)g %(y)g    %(material)d  %(area)g
""" % vars())
    f.close()

    # Run triangle
    import commands
    failure, output = commands.getstatusoutput('triangle -h')
    if failure:
        print 'triangle is not properly installed on the system '\
              '(try triangle -h)'
        sys.exit(1)

    opts = '-pq '
    if num_regions:
        opts += '-A '
    if maxarea is not None:
        opts += '-a%g ' % maxarea
    elif num_regions:
        opts += '-a '

    cmd = 'triangle ' + opts + casename + '.poly'
    if not run_triangle:
        print 'Run', cmd
        return -1, -1
    else:
        print cmd

    failure, output = commands.getstatusoutput(cmd)
    #os.system(cmd)
    #output = ''
    if failure:
        print 'Failed to run\n%s\n%s' % (cmd, output)
    # Interpret output
    for line in output.splitlines():
        if 'Mesh vertices:' in line:
            num_vertices = int(line.split()[2])
        if 'Mesh triangles:' in line:
            num_triangles = int(line.split()[2])
    #if sys.platform == 'linux2':
    #    os.system('showme ' + casename + '.1')
    return num_vertices, num_triangles


def load_triangle_mesh(casename='tmp', refno=1):
    f = open('%s.%s.ele' % (casename, refno), 'r')
    num_elements, num_nodes_pr_elm, num_attributes = \
                  [int(word) for word in f.readline().split()]
    import numpy as np
    data = np.loadtxt(f, dtype=np.int)
    f.close()
    connectivity = data[:,1:num_nodes_pr_elm+1]
    materials = data[:,-1]

    f = open('%s.%s.node' % (casename, refno), 'r')
    num_vertices, dim, num_attributes, num_boundary_markers = \
                  [int(word) for word in f.readline().split()]
    import numpy as np
    data = np.loadtxt(f, dtype=np.float)
    f.close()
    coordinates = data[:,1:dim+1]

    return coordinates, connectivity, materials


def gmsh_output(outer_boundary, inner_boundaries=[], casename='tmp',
                maxarea=None, run_triangle=True):
    if isinstance(inner_boundaries, Boundary):
        inner_boundaries = [inner_boundaries]  # wrap in list

    f = open(casename + '.geo', 'w')
    # Set offsets
    point_offset = len(outer_boundary.points)
    segment_offset = len(outer_boundary.segments)
    for inner_boundary in inner_boundaries:
        inner_boundary.set_offsets(point_offset, segment_offset)
        point_offset += inner_boundary.points.shape[0]
        segment_offset += len(inner_boundary.segments)


    num_vertices = point_offset
    num_segments = segment_offset
    dimension = 2
    num_attributes = 0
    num_boundary_markers = 0
    boundaries = [outer_boundary]
    boundaries.extend(inner_boundaries)

    f.write("""
// Definition of boundary points
""")
    for bno, boundary in enumerate(boundaries):
        if bno == 0:
            f.write("""\
//
// Outer boundary with %d points
//
""" % (boundary.points.shape[0]))
        else:
            f.write("""\
//
// Inner boundary with %d points
//
""" % (boundary.points.shape[0]))

        for vertex_no, x, y, lc in boundary.iterpoints2():
            f.write("Point(%(vertex_no)d) = {%(x)g, %(y)g, 0, %(lc)g};\n" % vars())

    counter = num_segments
    for bno, boundary in enumerate(boundaries):
        if bno == 0:
            f.write("""\
//
// Outer boundary with %d Line segments
//
""" % (len(boundary.segments)))
        else:
            f.write("""\
//
// Inner boundary with %d segments
//
""" % (len(boundary.segments)))

        segments = []
        for segment_no, start, end, marker in boundary.itersegments():
            f.write("Line%(segment_no)d) = {%(start)d, %(end)d};\n" % vars())
            segments.append(segment_no)
            f.write("Physical Line%(segment_no)d) = {%(marker)d};\n" % vars())
        counter += 1
        f.write("Line Loop(%d) = {%s};\n" % (bno, ','.join(segments)))
    counter += 1
    # The first Line Loop is the exterior boundary, the next are the holes
    f.write("Plane Surface(1) = {%s}" % \
            (counter, ','.join(range(num_segments+1, counter))))


    return



def plot_mesh(vertices, cells, materials=None, plotfile='tmp.png'):
    cell_vertex_coordinates = []
    for e in xrange(cells.shape[0]):
        local_vertex_numbers = cells[e,:]
        local_coordinates = vertices[local_vertex_numbers,:]
        cell_vertex_coordinates.append(local_coordinates)
    import matplotlib.cm as cm
    import matplotlib.collections as collections
    import matplotlib.pyplot as plt
    col = collections.PolyCollection(cell_vertex_coordinates)
    if materials is not None:
        #materials[materials == 0] = 1
        #materials = np.asarray(materials, float)
        #materials = np.random.uniform(0, 1, materials.size)
        #print 'size of materials:', materials.size
        col.set_array(materials)
        #col.set_cmap(cm.jet)
        #col.set_cmap(cm.gray_r)
        col.set_cmap(cm.hot_r)
    fig = plt.figure()
    ax = fig.gca()
    ax.add_collection(col)
    xmin, xmax = vertices[:,0].min(), vertices[:,0].max()
    ymin, ymax = vertices[:,1].min(), vertices[:,1].max()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    plt.savefig(plotfile)
    plt.show()






