#!/usr/bin/env python
"""
Class for a scalar (or vector) field over a BoxGrid or UniformBoxGrid grid.
"""


from numpy import zeros, array, transpose, ndarray, linspace, meshgrid

import fenics

__all__ = ['BoxField', 'BoxGrid', 'UniformBoxGrid', 'X', 'Y', 'Z',
           'fenics_function2BoxField', 'update_from_fenics_array']

# constants for indexing the space directions:
X = X1 = 0
Y = X2 = 1
Z = X3 = 2


class UniformBoxGrid(object):
    """
    Simple uniform grid on an interval, rectangle, box, or hypercube.

    =============      ====================================================
      Attributes                           Description
    =============      ====================================================
    nsd                no of spatial dimensions in the grid
    min_coor           array of minimum coordinates
    max_coor           array of maximum coordinates
    division           array of cell divisions in the
    delta              array of grid spacings
    dirnames           names of the space directions ('x', 'y', etc.)
    shape              (nx+1, ny+1, ...); dimension of array over grid
    coor               list of coordinates; self.coor[Y][j] is the
                       the j-th coordinate in direction Y (=1)
                       X, Y, Z are predefined constants 0, 1, 2
    coorv              expanded version of coor for vectorized expressions
                       (in 2D, self.coorv[0] = self.coor[0][:,newaxis])
    tolerance          small geometric tolerance based on grid coordinates
    npoints            total number of grid points
    =============      ====================================================

    """
    def __init__(self,
                 min=(0,0),                  # minimum coordinates
                 max=(1,1),                  # maximum coordinates
                 division=(4,4),             # cell divisions
                 dirnames=('x', 'y', 'z')):  # names of the directions
        """
        Initialize a BoxGrid by giving domain range (minimum and
        maximum coordinates: min and max tuples/lists/arrays)
        and number of cells in each space direction (division tuple/list/array).
        The dirnames tuple/list holds the names of the coordinates in
        the various spatial directions.

        >>> g = UniformBoxGrid(min=0, max=1, division=10)
        >>> g = UniformBoxGrid(min=(0,-1), max=(1,1), division=(10,4))
        >>> g = UniformBoxGrid(min=(0,0,-1), max=(2,1,1), division=(2,3,5))
        """
        # Allow int/float specifications in one-dimensional grids
        # (just turn to lists for later multi-dimensional processing)
        if isinstance(min, (int,float)):
            min = [min]
        if isinstance(max, (int,float)):
            max = [max]
        if isinstance(division, (int,float)):
            division = [division]
        if isinstance(dirnames, str):
            dirnames = [dirnames]

        self.nsd = len(min)
        # strip dirnames down to right space dim (in case the default
        # with three components were unchanged by the user):
        dirnames = dirnames[:self.nsd]

        # check consistent lengths:
        for a in max, division:
            if len(a) != self.nsd:
                raise ValueError(
                    'Incompatible lengths of arguments to constructor'\
                    ' (%d != %d)' % (len(a), self.nsd))

        self.min_coor = array(min, float)
        self.max_coor = array(max, float)
        self.dirnames = dirnames
        self.division = division
        self.coor = [None]*self.nsd
        self.shape = [0]*self.nsd
        self.delta = zeros(self.nsd)

        for i in range(self.nsd):
            self.delta[i] = \
                 (self.max_coor[i] -  self.min_coor[i])/float(self.division[i])
            self.shape[i] = self.division[i] + 1  # no of grid points
            self.coor[i] = \
                 linspace(self.min_coor[i], self.max_coor[i], self.shape[i])
        self._more_init()

    def _more_init(self):
        self.shape = tuple(self.shape)
        self.coorv = meshgrid(*self.coor, indexing='ij')
        if not isinstance(self.coorv, (list,tuple)):
            # 1D grid, wrap self.coorv as list:
            self.coorv = [self.coorv]

        self.npoints = 1
        for i in range(len(self.shape)):
            self.npoints *= self.shape[i]

        self.tolerance = (max(self.max_coor) - min(self.min_coor))*1E-14

        # nicknames: xcoor, ycoor, xcoorv, ycoorv, etc
        for i in range(self.nsd):
            self.__dict__[self.dirnames[i]+'coor'] = self.coor[i]
            self.__dict__[self.dirnames[i]+'coorv'] = self.coorv[i]

        if self.nsd == 3:
            # make boundary coordinates for vectorization:
            xdummy, \
            self.ycoorv_xfixed_boundary, \
            self.zcoorv_xfixed_boundary = meshgrid(0, self.ycoor, self.zcoor,
                                                   indexing='ij')

            self.xcoorv_yfixed_boundary, \
            ydummy, \
            self.zcoorv_yfixed_boundary = meshgrid(self.xcoor, 0, self.zcoor,
                                                   indexing='ij')

            self.xcoorv_yfixed_boundary, \
            self.zcoorv_yfixed_boundary, \
            zdummy = meshgrid(self.xcoor, self.ycoor, 0, indexing='ij')

    # could have _ in all variable names and define read-only
    # access via properties

    def string2griddata(s):
        """
        Turn a text specification of a grid into a dictionary
        with the grid data.
        For example,

        >>> s = "domain=[0,10] indices=[0:11]"
        >>> data = BoxGrid.string2griddata(s)
        >>> data
        {'dirnames': ('x', 'y'), 'division': [10], 'max': [10], 'min': [0]}

        >>> s = "domain=[0.2,0.5]x[0,2E+00] indices=[0:20]x[0:100]"
        >>> data = BoxGrid.string2griddata(s)
        >>> data
        {'dirnames': ('x', 'y', 'z'),
         'division': [19, 99],
         'max': [0.5, 2],
         'min': [0.2, 0]}

        >>> s = "[0,1]x[0,2]x[-1,1.5] [0:25]x[0:10]x[0:16]"
        >>> data = BoxGrid.string2griddata(s)
        >>> data
        {'dirnames': ('x', 'y', 'z'),
         'division': [24, 9, 15],
         'max': [1.0, 2.0, 1.5],
         'min': [0.0, 0.0, -1.0]}

        The data dictionary can be used as keyword arguments to the
        class UniformBoxGrid constructor.
        """

        domain  = r'\[([^,]*),([^\]]*)\]'
        indices = r'\[([^:,]*):([^\]]*)\]'
        import re
        d = re.findall(domain, s)
        i = re.findall(indices, s)
        nsd = len(d)
        if nsd != len(i):
            raise ValueError('could not parse "%s"' % s)
        kwargs = {}
        dirnames = ('x', 'y', 'z')
        for dir in range(nsd):
            if not isinstance(d[dir], (list,tuple)) or len(d[dir]) != 2 or \
               not isinstance(i[dir], (list,tuple)) or len(i[dir]) != 2:
                raise ValueError('syntax error in "%s"' % s)

            # old syntax (nx, xmin, xmax, ny, ymin, etc.):
            #kwargs[dirnames[dir]] = (float(d[dir][0]), float(d[dir][1]))
            #kwargs['n'+dirnames[dir]] = int(i[dir][1]) - int(i[dir][0]) # no of cells!
            kwargs['min'] = [float(d[dir][0]) for dir in range(nsd)]
            kwargs['max'] = [float(d[dir][1]) for dir in range(nsd)]
            kwargs['division'] = [int(i[dir][1]) - int(i[dir][0]) \
                                  for dir in range(nsd)]
            kwargs['dirnames'] = dirnames[:nsd]
        return kwargs
    string2griddata = staticmethod(string2griddata)

    def __getitem__(self, i):
        """
        Return access to coordinate array in direction no i, or direction
        name i, or return the coordinate of a point if i is an nsd-tuple.

        >>> g = UniformBoxGrid(x=(0,1), y=(-1,1), nx=2, ny=4)  # xy grid
        >>> g[0][0] == g.min[0]   # min coor in direction 0
        True
        >>> g['x'][0] == g.min[0]   # min coor in direction 'x'
        True
        >>> g[0,4]
        (0.0, 1.0)
        >>> g = UniformBoxGrid(min=(0,-1), max=(1,1), division=(2,4), dirnames=('y', 'z'))
        >>> g[1][0] == g.min[1]
        True
        >>> g['z'][0] == g.min[1]   # min coor in direction 'z'
        True
        """
        if isinstance(i, str):
            return self.coor[self.name2dirindex(i)]
        elif isinstance(i, int):
            if self.nsd > 1:
                return self.coor[i]     # coordinate array
            else:
                return self.coor[0][i]  # coordinate itself in 1D
        elif isinstance(i, (list,tuple)):
            return tuple([self.coor[k][i[k]] for k in range(len(i))])
        else:
            raise TypeError('i must be str, int, tuple')


    def __setitem__(self, i, value):
        raise AttributeError('subscript assignment is not valid for '\
                             '%s instances' % self.__class__.__name__)

    def ncells(self, i):
        """Return no of cells in direction i."""
        # i has the meaning as in __getitem__. May be removed if not much used
        return len(self.coor[i])-1

    def name2dirindex(self, name):
        """
        Return direction index corresponding to direction name.
        In an xyz-grid, 'x' is 0, 'y' is 1, and 'z' is 2.
        In an yz-grid, 'x' is not defined, 'y' is 0, and 'z' is 1.
        """
        try:
            return self.dirnames.index(name)
        except ValueError:
            print name, 'is not defined'
            return None

    def dirindex2name(self, i):
        """Inverse of name2dirindex."""
        try:
            return self.dirnames[i]
        except IndexError:
            print i, 'is not a valid index'
            return None

    def ok(self):
        return True  # constructor init only => always ok

    def __len__(self):
        """Total number of grid points."""
        n = 1
        for dir in self.coor:
            n *= len(dir)
        return n

    def __repr__(self):
        s = self.__class__.__name__ + \
            '(min=%s, max=%s, division=%s, dirnames=%s)' % \
            (self.min_coor.tolist(),
             self.max_coor.tolist(),
             self.division, self.dirnames)
        return s

    def __str__(self):
        """Pretty print, using the syntax of init_fromstring."""
        domain = 'x'.join(['[%g,%g]' % (min_, max_) \
                           for min_, max_ in
                           zip(self.min_coor, self.max_coor)])
        indices = 'x'.join(['[0:%d]' % div for div in self.division])
        return 'domain=%s  indices=%s' % (domain, indices)

    def interpolator(self, point_values):
        """
        Given a self.nsd dimension array point_values with
        values at each grid point, this method returns a function
        for interpolating the scalar field defined by point_values
        at an arbitrary point.

        2D Example:
        given a filled array point_values[i,j], compute
        interpolator = grid.interpolator(point_values)
        v = interpolator(0.1243, 9.231)  # interpolate point_values

        >>> g=UniformBoxGrid(x=(0,2), nx=2, y=(-1,1), ny=2)
        >>> g
        UniformBoxGrid(x=(0,2), nx=2, y=(-1,1), ny=2)
        >>> def f(x,y): return 2+2*x-y

        >>> f=g.vectorized_eval(f)
        >>> f
        array([[ 3.,  2.,  1.],
               [ 5.,  4.,  3.],
               [ 7.,  6.,  5.]])
        >>> i=g.interpolator(f)
        >>> i(0.1,0.234)        # interpolate (not a grid point)
        1.9660000000000002
        >>> f(0.1,0.234)        # exact answer
        1.9660000000000002
        """
        args = self.coor
        args.append(point_values)
        # make use of wrap2callable, which applies ScientificPython
        return wrap2callable(args)

    def vectorized_eval(self, f):
        """
        Evaluate a function f (of the space directions) over a grid.
        f is supposed to be vectorized.

        >>> g = BoxGrid(x=(0,1), y=(0,1), nx=3, ny=3)
        >>> # f(x,y) = sin(x)*exp(x-y):
        >>> a = g.vectorized_eval(lambda x,y: sin(x)*exp(y-x))
        >>> print a
        [[ 0.          0.          0.          0.        ]
         [ 0.23444524  0.3271947   0.45663698  0.63728825]
         [ 0.31748164  0.44308133  0.6183698   0.86300458]
         [ 0.30955988  0.43202561  0.60294031  0.84147098]]

        >>> # f(x,y) = 2: (requires special consideration)
        >>> a = g.vectorized_eval(lambda x,y: zeros(g.shape)+2)
        >>> print a
        [[ 2.  2.  2.  2.]
         [ 2.  2.  2.  2.]
         [ 2.  2.  2.  2.]
         [ 2.  2.  2.  2.]]
        """
        a = f(*self.coorv)

        # check if f is really vectorized:
        try:
            msg = 'calling %s, which is supposed to be vectorized' % f.__name__
        except AttributeError:  # if __name__ is missing
            msg = 'calling a function, which is supposed to be vectorized'
        try:
            self.compatible(a)
        except (IndexError,TypeError) as e:
            print 'e=',e, type(e), e.__class__.__name__
            raise e.__class__('BoxGrid.vectorized_eval(f):\n%s, BUT:\n%s' % \
                              (msg, e))
        return a

    def init_fromstring(s):
        data = UniformBoxGrid.string2griddata(s)
        return UniformBoxGrid(**data)
    init_fromstring = staticmethod(init_fromstring)

    def compatible(self, data_array, name_of_data_array=''):
        """
        Check that data_array is a NumPy array with dimensions
        compatible with the grid.
        """
        if not isinstance(data_array, ndarray):
            raise TypeError('data %s is %s, not NumPy array' % \
                            (name_of_data_array, type(data_array)))
        else:
            if data_array.shape != self.shape:
                raise IndexError("data %s of shape %s is not "\
                                 "compatible with the grid's shape %s" % \
                                 (name_of_data_array, data_array.shape,
                                  self.shape))
        return True # if we haven't raised any exceptions

    def iter(self, domain_part='all', vectorized_version=True):
        """
        Return iterator over grid points.
        domain_part = 'all':  all grid points
        domain_part = 'interior':  interior grid points
        domain_part = 'all_boundary':  all boundary points
        domain_part = 'interior_boundary':  interior boundary points
        domain_part = 'corners':  all corner points
        domain_part = 'all_edges':  all points along edges in 3D grids
        domain_part = 'interior_edges':  interior points along edges

        vectorized_version is true if the iterator returns slice
        objects for the index slice in each direction.
        vectorized_version is false if the iterator visits each point
        at a time (scalar version).
        """
        self.iterator_domain = domain_part
        self.vectorized_iter = vectorized_version
        return self

    def __iter__(self):
        # Idea: set up slices for the various self.iterator_domain
        # values. In scalar mode, make a loop over the slices and
        # yield the scalar value. In vectorized mode, return the
        # appropriate slices.

        self._slices = []  # elements meant to be slice objects

        if self.iterator_domain == 'all':
            self._slices.append([])
            for i in range(self.nsd):
                self._slices[-1].append((i, slice(0, len(self.coor[i]), 1)))

        elif self.iterator_domain == 'interior':
            self._slices.append([])
            for i in range(self.nsd):
                self._slices[-1].append((i, slice(1, len(self.coor[i])-1, 1)))

        elif self.iterator_domain == 'all_boundary':
            for i in range(self.nsd):
                self._slices.append([])
                # boundary i fixed at 0:
                for j in range(self.nsd):
                    if j != i:
                        self._slices[-1].\
                           append((j, slice(0, len(self.coor[j]), 1)))
                    else:
                        self._slices[-1].append((i, slice(0, 1, 1)))
                # boundary i fixed at its max value:
                for j in range(self.nsd):
                    if j != i:
                        self._slices[-1].\
                           append((j, slice(0, len(self.coor[j]), 1)))
                    else:
                        n = len(self.coor[i])
                        self._slices[-1].append((i, slice(n-1, n, 1)))

        elif self.iterator_domain == 'interior_boundary':
            for i in range(self.nsd):
                self._slices.append([])
                # boundary i fixed at 0:
                for j in range(self.nsd):
                    if j != i:
                        self._slices[-1].\
                           append((j, slice(1, len(self.coor[j])-1, 1)))
                    else:
                        self._slices[-1].append((i, slice(0, 1, 1)))
                # boundary i fixed at its max value:
                for j in range(self.nsd):
                    if j != i:
                        self._slices[-1].\
                           append((j, slice(1, len(self.coor[j])-1, 1)))
                    else:
                        n = len(self.coor[i])
                        self._slices[-1].append((i, slice(n-1, n, 1)))

        elif self.iterator_domain == 'corners':
            if self.nsd == 1:
                for i0 in (0, len(self.coor[0])-1):
                    self._slices.append([])
                    self._slices[-1].append((0, slice(i0, i0+1, 1)))
            elif self.nsd == 2:
                for i0 in (0, len(self.coor[0])-1):
                    for i1 in (0, len(self.coor[1])-1):
                        self._slices.append([])
                        self._slices[-1].append((0, slice(i0, i0+1, 1)))
                        self._slices[-1].append((0, slice(i1, i1+1, 1)))
            elif self.nsd == 3:
                for i0 in (0, len(self.coor[0])-1):
                    for i1 in (0, len(self.coor[1])-1):
                        for i2 in (0, len(self.coor[2])-1):
                            self._slices.append([])
                            self._slices[-1].append((0, slice(i0, i0+1, 1)))
                            self._slices[-1].append((0, slice(i1, i1+1, 1)))
                            self._slices[-1].append((0, slice(i2, i2+1, 1)))

        elif self.iterator_domain == 'all_edges':
            print 'iterator over "all_edges" is not implemented'
        elif self.iterator_domain == 'interior_edges':
            print 'iterator over "interior_edges" is not implemented'
        else:
            raise ValueError('iterator over "%s" is not impl.' % \
                             self.iterator_domain)

#    "def __next__(self):"
        """
        If vectorized mode:
        Return list of slice instances, where the i-th element in the
        list represents the slice for the index in the i-th space
        direction (0,...,nsd-1).

        If scalar mode:
        Return list of indices (in multi-D) or the index (in 1D).
        """
        if self.vectorized_iter:
            for s in self._slices:
                yield [slice_in_dir for dir, slice_in_dir in s]
        else:
            # scalar version
            for s in self._slices:
                slices = [slice_in_dir for dir, slice_in_dir in s]
                if len(slices) == 1:
                    for i in xrange(slices[0].start, slices[0].stop):
                        yield i
                elif len(slices) == 2:
                    for i in xrange(slices[0].start, slices[0].stop):
                        for j in xrange(slices[1].start, slices[1].stop):
                            yield [i, j]
                elif len(slices) == 3:
                    for i in xrange(slices[0].start, slices[0].stop):
                        for j in xrange(slices[1].start, slices[1].stop):
                            for k in xrange(slices[2].start, slices[2].stop):
                                yield [i, j, k]


    def locate_cell(self, point):
        """
        Given a point (x, (x,y), (x,y,z)), locate the cell in which
        the point is located, and return
        1) the (i,j,k) vertex index
        of the "lower-left" grid point in this cell,
        2) the distances (dx, (dx,dy), or (dx,dy,dz)) from this point to
        the given point,
        3) a boolean list if point matches the
        coordinates of any grid lines. If a point matches
        the last grid point in a direction, the cell index is
        set to the max index such that the (i,j,k) index can be used
        directly for look up in an array of values. The corresponding
        element in the distance array is then set 0.
        4) the indices of the nearest grid point.

        The method only works for uniform grid spacing.
        Used for interpolation.

        >>> g1 = UniformBoxGrid(min=0, max=1, division=4)
        >>> cell_index, distance, match, nearest = g1.locate_cell(0.7)
        >>> print cell_index
        [2]
        >>> print distance
        [ 0.2]
        >>> print match
        [False]
        >>> print nearest
        [3]
        >>>
        >>> g1.locate_cell(0.5)
        ([2], array([ 0.]), [True], [2])
        >>>
        >>> g2 = UniformBoxGrid.init_fromstring('[-1,1]x[-1,2] [0:3]x[0:4]')
        >>> print g2.coor
        [array([-1.        , -0.33333333,  0.33333333,  1.        ]), array([-1.  , -0.25,  0.5 ,  1.25,  2.  ])]
        >>> g2.locate_cell((0.2,0.2))
        ([1, 1], array([ 0.53333333,  0.45      ]), [False, False], [2, 2])
        >>> g2.locate_cell((1,2))
        ([3, 4], array([ 0.,  0.]), [True, True], [3, 4])
        >>>
        >>>
        >>>
        """
        if isinstance(point, (int,float)):
            point = [point]
        nsd = len(point)
        if nsd != self.nsd:
            raise ValueError('point=%s has wrong dimension (this is a %dD grid!)' % \
                             (point, self.nsd))
        #index = zeros(nsd, int)
        index = [0]*nsd
        distance = zeros(nsd)
        grid_point = [False]*nsd
        nearest_point = [0]*nsd
        for i, coor in enumerate(point):
            # is point inside the domain?
            if coor < self.min_coor[i] or coor > self.max_coor[i]:
                raise ValueError(
                    'locate_cell: point=%s is outside the domain [%s,%s]' % \
                    point, self.min_coor[i], self.max_coor[i])
            index[i] = int((coor - self.min_coor[i])//self.delta[i])  # (need integer division)
            distance[i] = coor - (self.min_coor[i] + index[i]*self.delta[i])
            if distance[i] > self.delta[i]/2:
                nearest_point[i] = index[i] + 1
            else:
                nearest_point[i] = index[i]
            if abs(distance[i]) < self.tolerance:
                grid_point[i] = True
                nearest_point[i] = index[i]
            if (abs(distance[i] - self.delta[i])) < self.tolerance:
                # last cell, update index such that it coincides with the point
                grid_point[i] = True
                index[i] += 1
                nearest_point[i] = index[i]
                distance[i] = 0.0

        return index, distance, grid_point, nearest_point

    def interpolate(v0, v1, x0, x1, x):
        return v0 + (v1-v0)/float(x1-x0)*(x-x0)

    def gridline_slice(self, start_coor, direction=0, end_coor=None):
        """
        Compute start and end indices of a line through the grid,
        and return a tuple that can be used as slice for the
        grid points along the line.

        The line must be in x, y or z direction (direction=0,1 or 2).
        If end_coor=None, the line ends where the grid ends.
        start_coor holds the coordinates of the start of the line.
        If start_coor does not coincide with one of the grid points,
        the line is snapped onto the grid (i.e., the line coincides with
        a grid line).

        Return: tuple with indices and slice describing the grid point
        indices that make up the line, plus a boolean "snapped" which is
        True if the original line did not coincide with any grid line,
        meaning that the returned line was snapped onto to the grid.

        >>> g2 = UniformBoxGrid.init_fromstring('[-1,1]x[-1,2] [0:3]x[0:4]')
        >>> print g2.coor
        [array([-1.        , -0.33333333,  0.33333333,  1.        ]),
         array([-1.  , -0.25,  0.5 ,  1.25,  2.  ])]

        >>> g2.gridline_slice((-1, 0.5), 0)
        ((slice(0, 4, 1), 2), False)

        >>> g2.gridline_slice((-0.9, 0.4), 0)
        ((slice(0, 4, 1), 2), True)

        >>> g2.gridline_slice((-0.2, -1), 1)
        ((1, slice(0, 5, 1)), True)

        """

        start_cell, start_distance, start_match, start_nearest = \
                    self.locate_cell(start_coor)
        # If snapping the line onto to the grid is not desired, the
        # start_cell and start_match lists must be used for interpolation
        # (i.e., interpolation is needed in the directions i where
        # start_match[i] is False).

        start_snapped = start_nearest[:]
        if end_coor is None:
            end_snapped = start_snapped[:]
            end_snapped[direction] = self.division[direction] # highest legal index
        else:
            end_cell, end_distance, end_match, end_nearest = \
                      self.locate_cell(end_coor)
            end_snapped = end_nearest[:]
        # recall that upper index limit must be +1 in a slice:
        line_slice = start_snapped[:]
        line_slice[direction] = \
            slice(start_snapped[direction], end_snapped[direction]+1, 1)
        # note that if all start_match are true, then the plane
        # was not snapped
        return tuple(line_slice), not array(start_match).all()


    def gridplane_slice(self, value, constant_coor=0):
        """
        Compute a slice for a plane through the grid,
        defined by coor[constant_coor]=value.

        Return a tuple that can be used as slice, plus a boolean
        parameter "snapped" reflecting if the plane was snapped
        onto a grid plane (i.e., value did not correspond to
        an existing grid plane).
        """
        start_coor = self.min_coor.copy()
        start_coor[constant_coor] = value
        start_cell, start_distance, start_match, start_nearest = \
                    self.locate_cell(start_coor)
        start_snapped = [0]*self.nsd
        start_snapped[constant_coor] = start_nearest[constant_coor]
        # recall that upper index limit must be +1 in a slice:
        end_snapped = [self.division[i] for i in range(self.nsd)]
        end_snapped[constant_coor] = start_snapped[constant_coor]
        plane_slice = [slice(start_snapped[i], end_snapped[i]+1, 1) \
                       for i in range(self.nsd)]
        plane_slice[constant_coor] = start_nearest[constant_coor]
        return tuple(plane_slice), not start_match[constant_coor]



class BoxGrid(UniformBoxGrid):
    """
    Extension of class UniformBoxGrid to non-uniform box grids.
    The coordinate vectors (in each space direction) can have
    arbitrarily spaced coordinate values.

    The coor argument must be a list of nsd (number of
    space dimension) components, each component contains the
    grid coordinates in that space direction (stored as an array).
    """
    def __init__(self, coor, dirnames=('x', 'y', 'z')):

        UniformBoxGrid.__init__(self,
                                min=[a[0] for a in coor],
                                max=[a[-1] for a in coor],
                                division=[len(a)-1 for a in coor],
                                dirnames=dirnames)
        # override:
        self.coor = coor

    def __repr__(self):
        s = self.__class__.__name__ + '(coor=%s)' % self.coor
        return s

    def locate_cell(self, point):
        raise NotImplementedError('Cannot locate point in cells in non-uniform grids')


def _test(g, points=None):
    result = 'g=%s' % str(g)
    def fv(*args):
        # vectorized evaluation function
        return zeros(g.shape)+2
    def fs(*args):
        # scalar version
        return 2
    fv_arr = g.vectorized_eval(fv)
    fs_arr = zeros(g.shape)

    coor = [0.0]*g.nsd
    itparts = ['all', 'interior', 'all_boundary', 'interior_boundary',
               'corners']
    if g.nsd == 3:
        itparts += ['all_edges', 'interior_edges']

    for domain_part in itparts:
        result += '\niterator over "%s"\n' % domain_part
        for i in g.iter(domain_part, vectorized_version=False):
            if isinstance(i, int):  i = [i]  # wrap index as list (if 1D)
            for k in range(g.nsd):
                coor[k] = g.coor[k][i[k]]
            result += '%s %s\n' % (i, coor)
            if domain_part == 'all':  # fs_arr shape corresponds to all points
                fs_arr[i] = fs(*coor)
        result += 'vectorized iterator over "%s":\n' % domain_part
        for slices in g.iter(domain_part, vectorized_version=True):
            if domain_part == 'all':
                fs_arr[slices] = fv(*g.coor)
            # else: more complicated
            for slice_ in slices:
                result += 'slice: %s values: %s\n' % (slice_, fs_arr[slice_])
    # boundary slices...
    return result


class Field(object):
    """
    General base class for all grids. Holds a connection to a
    grid, a name of the field, and optionally a list of the
    independent variables and a string with a description of the
    field.
    """
    def __init__(self, grid, name,
                 independent_variables=None,
                 description=None,
                 **kwargs):
        self.grid = grid

        self.name = name
        self.independent_variables = independent_variables
        if self.independent_variables is None:
            # copy grid.dirnames as independent variables:
            self.independent_variables = self.grid.dirnames

        # metainformation:
        self.meta = {'description': description,}
        self.meta.update(kwargs)  # user can add more meta information


class BoxField(Field):
    """
    Field over a BoxGrid or UniformBoxGrid grid.

    =============      =============================================
      Attributes                       Description
    =============      =============================================
    grid               reference to the underlying grid instance
    values             array holding field values at the grid points
    =============      =============================================

    """
    def __init__(self, grid, name, vector=0, **kwargs):
        """
        Initialize scalar or vector field over a BoxGrid/UniformBoxGrid.

        =============      ===============================================
          Arguments                          Description
        =============      ===============================================
        *grid*             grid instance
        *name*             name of the field
        *vector*           scalar field if 0, otherwise the no of vector
                           components (spatial dimensions of vector field)
        *values*           (*kwargs*) optional array with field values
        =============      ===============================================

        Here is an example::

        >>> g = UniformBoxGrid(min=[0,0], max=[1.,1.], division=[3, 4])

        >>> print g
        domain=[0,1]x[0,1]  indices=[0:3]x[0:4]

        >>> u = BoxField(g, 'u')
        >>> u.values = u.grid.vectorized_eval(lambda x,y: x + y)

        >>> i = 1; j = 2
        >>> print 'u(%g, %g)=%g' % (g.coor[X][i], g.coor[Y][j], u.values[i,j])
        u(0.333333, 0.5)=0.833333

        >>> # visualize:
        >>> from scitools.std import surf
        >>> surf(u.grid.coorv[X], u.grid.coorv[Y], u.values)

        ``u.grid.coorv`` is a list of coordinate arrays that are
        suitable for Matlab-style visualization of 2D scalar fields.
        Also note how one can access the coordinates and u value at
        a point (i,j) in the grid.
        """
        Field.__init__(self, grid, name, **kwargs)

        if vector > 0:
            # for a vector field we add a "dimension" in values for
            # the various vector components (first index):
            self.required_shape = [vector]
            self.required_shape += list(self.grid.shape)
        else:
            self.required_shape = self.grid.shape

        if 'values' in kwargs:
            values = kwargs['values']
            self.set_values(values)
        else:
            # create array of scalar field grid point values:
            self.values = zeros(self.required_shape)

        # doesn't  work: self.__getitem__ = self.values.__getitem__
        #self.__setitem__ = self.values.__setitem__

    def copy_values(self, values):
        """Take a copy of the values array and reshape it if necessary."""
        self.set_values(values.copy())

    def set_values(self, values):
        """Attach the values array to this BoxField object."""
        if values.shape == self.required_shape:
            self.values = values  # field data are provided
        else:
            try:
                values.shape = self.required_shape
                self.values = values
            except ValueError:
                raise ValueError(
                    'values array is incompatible with grid size; '\
                    'shape is %s while required shape is %s' % \
                    (values.shape, self.required_shape))

    def update(self):
        """Update the self.values array (if grid has been changed)."""
        if self.grid.shape != self.values.shape:
            self.values = zeros(self.grid.shape)

    # these are slower than u_ = u.values; u_[i] since an additional
    # function call is required compared to NumPy array indexing:
    def __getitem__(self, i):  return self.values[i]
    def __setitem__(self, i, v):  self.values[i] = v

    def __str__(self):
        if len(self.values.shape) > self.grid.nsd:
            s = 'Vector field with %d components' % self.values.shape[-1]
        else:
            s = 'Scalar field'
        s += ', over ' + str(self.grid)
        return s

    def gridline(self, start_coor, direction=0, end_coor=None,
                 snap=True):
        """
        Return a coordinate array and corresponding field values
        along a line starting with start_coor, in the given
        direction, and ending in end_coor (default: grid boundary).
        Two more boolean values are also returned: fixed_coor
        (the value of the fixed coordinates, which may be different
        from those in start_coor if snap=True) and snapped (True if
        the line really had to be snapped onto the grid, i.e.,
        fixed_coor differs from coordinates in start_coor.

        If snap is True, the line is snapped onto the grid, otherwise
        values along the line must be interpolated (not yet implemented).

        >>> g2 = UniformBoxGrid.init_fromstring('[-1,1]x[-1,2] [0:3]x[0:4]')
        >>> print g2
        UniformBoxGrid(min=[-1. -1.], max=[ 1.  2.],
        division=[3, 4], dirnames=('x', 'y'))
        >>> print g2.coor
        [array([-1.        , -0.33333333,  0.33333333,  1.        ]),
        array([-1.  , -0.25,  0.5 ,  1.25,  2.  ])]

        >>> u = BoxField(g2, 'u')
        >>> u.values = u.grid.vectorized_eval(lambda x,y: x + y)
        >>> xc, uc, fixed_coor, snapped = u.gridline((-1,0.5), 0)
        >>> print xc
        [-1.         -0.33333333  0.33333333  1.        ]
        >>> print uc
        [-0.5         0.16666667  0.83333333  1.5       ]
        >>> print fixed_coor, snapped
        [0.5] False
        >>> #plot(xc, uc, title='u(x, y=%g)' % fixed_coor)
        """
        if not snap:
            raise NotImplementedError('Use snap=True, no interpolation impl.')

        slice_index, snapped = \
             self.grid.gridline_slice(start_coor, direction, end_coor)
        fixed_coor = [self.grid[s][i] for s,i in enumerate(slice_index) \
                      if not isinstance(i, slice)]
        if len(fixed_coor) == 1:
            fixed_coor = fixed_coor[0]  # avoid returning list of length 1
        return self.grid.coor[direction][slice_index[direction].start:\
                                         slice_index[direction].stop], \
               self.values[slice_index], fixed_coor, snapped

    def gridplane(self, value, constant_coor=0, snap=True):
        """
        Return two one-dimensional coordinate arrays and
        corresponding field values over a plane where one coordinate,
        constant_coor, is fixed at a value.

        If snap is True, the plane is snapped onto a grid plane such
        that the points in the plane coincide with the grid points.
        Otherwise, the returned values must be interpolated (not yet impl.).
        """
        if not snap:
            raise NotImplementedError('Use snap=True, no interpolation impl.')

        slice_index, snapped = self.grid.gridplane_slice(value, constant_coor)
        if constant_coor == 0:
            x = self.grid.coor[1]
            y = self.grid.coor[2]
        elif constant_coor == 1:
            x = self.grid.coor[0]
            y = self.grid.coor[2]
        elif constant_coor == 2:
            x = self.grid.coor[0]
            y = self.grid.coor[1]
        fixed_coor = self.grid.coor[constant_coor][slice_index[constant_coor]]
        return x, y, self.values[slice_index], fixed_coor, snapped

def _rank12rankd_mesh(a, shape):
    """
    Given rank 1 array a with values in a mesh with the no of points
    described by shape, transform the array to the right "mesh array"
    with the same shape.
    """
    shape = list(shape)
    shape.reverse()
    if len(a.shape) == 1:
        return a.reshape(shape).transpose()
    else:
        raise ValueError('array a cannot be multi-dimensional (not %s), ' \
                         'break it up into one-dimensional components' \
                         % a.shape)

def fenics_mesh2UniformBoxGrid(fenics_mesh, division=None):
    """
    Turn a regular, structured DOLFIN finite element mesh into
    a UniformBoxGrid object. (Application: plotting with scitools.)
    Standard DOLFIN numbering numbers the nodes along the x[0] axis,
    then x[1] axis, and so on.
    """
    if hasattr(fenics_mesh, 'structured_data'):
        coor = fenics_mesh.structured_data
        min_coor = [c[0]  for c in coor]
        max_coor = [c[-1] for c in coor]
        division = [len(c)-1 for c in coor]
    else:
        if division is None:
            raise ValueError('division must be given when fenics_mesh does not have a strutured_data attribute')
        else:
            coor = fenics_mesh.coordinates() # numpy array
            min_coor = coor[0]
            max_coor = coor[-1]

    return UniformBoxGrid(min=min_coor, max=max_coor,
                          division=division)

def fenics_mesh2BoxGrid(fenics_mesh, division=None):
    """
    Turn a structured DOLFIN finite element mesh into
    a BoxGrid object.
    Standard DOLFIN numbering numbers the nodes along the x[0] axis,
    then x[1] axis, and so on.
    """
    if hasattr(fenics_mesh, 'structured_data'):
        coor = fenics_mesh.structured_data
        return BoxGrid(coor)
    else:
        if division is None:
            raise ValueError('division must be given when fenics_mesh does not have a strutured_data attribute')
        else:
            c = fenics_mesh.coordinates() # numpy array
            shape = [n+1 for n in division]  # shape for points in each dir.

            c2 = [c[:,i] for i in range(c.shape[1])]  # split x,y,z components
            for i in range(c.shape[1]):
                c2[i] = _rank12rankd_mesh(c2[i], shape)
            # extract coordinates in the different directions
            coor = []
            if len(c2) == 1:
                coor = [c2[0][:]]
            elif len(c2) == 2:
                coor = [c2[0][:,0], c2[1][0,:]]
            elif len(c2) == 3:
                coor = [c2[0][:,0,0], c2[1][0,:,0], c2[2][0,0,:]]
            return BoxGrid(coor)


def fenics_function2BoxField(fenics_function, fenics_mesh,
                             division=None, uniform_mesh=True):
    """
    Turn a DOLFIN P1 finite element field over a structured mesh into
    a BoxField object.
    Standard DOLFIN numbering numbers the nodes along the x[0] axis,
    then x[1] axis, and so on.

    If the DOLFIN function employs elements of degree > 1, one should
    project or interpolate the field onto a field with elements of
    degree=1.
    """
    if fenics_function.ufl_element().degree() != 1:
        raise TypeError("""\
The fenics_function2BoxField function works with degree=1 elements
only. The DOLFIN function (fenics_function) has finite elements of type
%s
i.e., the degree=%d != 1. Project or interpolate this function
onto a space of P1 elements, i.e.,

V2 = FunctionSpace(mesh, 'CG', 1)
u2 = project(u, V2)
# or
u2 = interpolate(u, V2)

""" % (str(fenics_function.ufl_element()), fenics_function.ufl_element().degree()))

    import dolfin
    if dolfin.__version__[:3] == "1.0":
        nodal_values = fenics_function.vector().array().copy()
    else:
        #map = fenics_function.function_space().dofmap().vertex_to_dof_map(fenics_mesh)
        d2v = fenics.dof_to_vertex_map(fenics_function.function_space())
        nodal_values = fenics_function.vector().array().copy()
        nodal_values[d2v] = fenics_function.vector().array().copy()

    if uniform_mesh:
        grid = fenics_mesh2UniformBoxGrid(fenics_mesh, division)
    else:
        grid = fenics_mesh2BoxGrid(fenics_mesh, division)

    if nodal_values.size > grid.npoints:
        # vector field, treat each component separately
        ncomponents = int(nodal_values.size/grid.npoints)
        try:
            nodal_values.shape = (ncomponents, grid.npoints)
        except ValueError as e:
            raise ValueError('Vector field (nodal_values) has length %d, there are %d grid points, and this does not match with %d components' % (nodal_values.size, grid.npoints, ncomponents))
        vector_field = [_rank12rankd_mesh(nodal_values[i,:].copy(),
                                          grid.shape) \
                        for i in range(ncomponents)]
        nodal_values = array(vector_field)
        bf = BoxField(grid, name=fenics_function.name(),
                      vector=ncomponents, values=nodal_values)
    else:
        try:
            nodal_values = _rank12rankd_mesh(nodal_values, grid.shape)
        except ValueError as e:
            raise ValueError('DOLFIN function has vector of size %s while the provided mesh has %d points and shape %s' % (nodal_values.size, grid.npoints, grid.shape))
        bf = BoxField(grid, name=fenics_function.name(),
                      vector=0, values=nodal_values)
    return bf

def update_from_fenics_array(fenics_array, box_field):
    """
    Update the values in a BoxField object box_field with a new
    DOLFIN array (fenics_array). The array must be reshaped and
    transposed in the right way
    (therefore box_field.copy_values(fenics_array) will not work).
    """
    nodal_values = fenics_array.copy()
    if len(nodal_values.shape) > 1:
        raise NotImplementedError # no support for vector valued functions yet
                                  # the problem is in _rank12rankd_mesh
    try:
        nodal_values = _rank12rankd_mesh(nodal_values, box_field.grid.shape)
    except ValueError as e:
        raise ValueError(
            'DOLFIN function has vector of size %s while '
            'the provided mesh demands %s' %
            (nodal_values.size, grid.shape))
    box_field.set_values(nodal_values)
    return box_field

def _str_equal(a, b):
    """Return '' if a==b, else string with indication of differences."""
    if a == b:
        return ''
    else:
        r = [] # resulting string with indication of differences
        for i, chars in enumerate(zip(a, b)):
            a_, b_ = chars
            if a_ == b_:
                r.append(a_)
            else:
                r.append('[')
                r.append(a_)
                r.append('|')
                r.append(b_)
                r.append(']')
        return ''.join(r)

def test_2Dmesh():
    g1 = UniformBoxGrid(min=0, max=1, division=4)
    expected = """\
g=domain=[0,1]  indices=[0:4]
iterator over "all"
[0] [0.0]
[1] [0.25]
[2] [0.5]
[3] [0.75]
[4] [1.0]
vectorized iterator over "all":
slice: slice(0, 5, 1) values: [ 2.  2.  2.  2.  2.]

iterator over "interior"
[1] [0.25]
[2] [0.5]
[3] [0.75]
vectorized iterator over "interior":
slice: slice(1, 4, 1) values: [ 2.  2.  2.]

iterator over "all_boundary"
[0, 4] [0.0]
vectorized iterator over "all_boundary":
slice: slice(0, 1, 1) values: [ 2.]
slice: slice(4, 5, 1) values: [ 2.]

iterator over "interior_boundary"
[0, 4] [0.0]
vectorized iterator over "interior_boundary":
slice: slice(0, 1, 1) values: [ 2.]
slice: slice(4, 5, 1) values: [ 2.]

iterator over "corners"
[0] [0.0]
[4] [1.0]
vectorized iterator over "corners":
slice: slice(0, 1, 1) values: [ 2.]
slice: slice(4, 5, 1) values: [ 2.]

iterator over "all_edges" is not implemented
iterator over "all_edges" is not implemented
iterator over "interior_edges" is not implemented
iterator over "interior_edges" is not implemented
"""
    computed = _test(g1, [0.7, 0.5])
    msg = _str_equal(expected, computed)
    print 'msg: [%s]' % msg
    success = not msg
    assert success, msg

    # Demonstrate functionality with structured mesh class
    spec = '[0,1]x[-1,2] with indices [0:3]x[0:2]'
    g2 = UniformBoxGrid.init_fromstring(spec)
    _test(g2, [(0.2,0.2), (1,2)])
    g3 = UniformBoxGrid(min=(0,0,-1), max=(1,1,1), division=(4,1,2))
    _test(g3)
    print 'g3=\n%s' % str(g3)
    print 'g3[Z]=', g3[Z]
    print 'g3[Z][1] =', g3[Z][1]
    print 'dx, dy, dz spacings:', g3.delta
    g4 = UniformBoxGrid(min=(0,-1), max=(1,1),
                        division=(4,2), dirnames=('y','z'))
    _test(g4)
    print 'g4["y"][-1]:', g4["y"][-1]

def _test3():
    from numpy import sin, zeros, exp
    # check vectorization evaluation:
    g = UniformBoxGrid(min=(0,0), max=(1,1), division=(3,3))
    try:
        g.vectorized_eval(lambda x,y: 2)
    except TypeError as msg:
        # fine, expect to arrive here
        print '*** Expected error in this test:', msg
    try:
        g.vectorized_eval(lambda x,y: zeros((2,2))+2)
    except IndexError as msg:
        # fine, expect to arrive here
        print '*** Expected error in this test:', msg

    a = g.vectorized_eval(lambda x,y: sin(x)*exp(y-x))
    print a
    a = g.vectorized_eval(lambda x,y: zeros(g.shape)+2)
    print a

def _test_field(g):
    print 'grid: %s' % g

    # function: 1 + x + y + z
    def f(*args):
        sum = 1.0
        for x in args:
            sum = sum + x
        return sum

    u = BoxField(g, 'u')
    v = BoxField(g, 'v', vector=g.nsd)

    u.values = u.grid.vectorized_eval(f)  # fill field values

    if   g.nsd == 1:
        v[:,X] = u.values + 1  # 1st component
    elif g.nsd == 2:
        v[:,:,X] = u.values + 1  # 1st component
        v[:,:,Y] = u.values + 2  # 2nd component
    elif g.nsd == 3:
        v[:,:,:,X] = u.values + 1  # 1st component
        v[:,:,:,Y] = u.values + 2  # 2nd component
        v[:,:,:,Z] = u.values + 3  # 3rd component

    # write out field values at the mid point of the grid
    # (convert g.shape to NumPy array and divide by 2 to find
    # approximately the mid point)
    midptindex = tuple(array(g.shape,int)/2)
    ptcoor = g[midptindex]
    # tuples with just one item does not work as indices
    print 'mid point %s has indices %s' % (ptcoor, midptindex)
    print 'f%s=%g' % (ptcoor, f(*ptcoor))
    print 'u at %s: %g' % (midptindex, u[midptindex])
    v_index = list(midptindex); v_index.append(slice(g.nsd))
    print 'v at %s: %s' % (midptindex, v[v_index])

    # test extraction of lines:
    if u.grid.nsd >= 2:
        direction = u.grid.nsd-1
        coor, u_coor, fixed_coor, snapped = \
              u.gridline(u.grid.min_coor, direction)
        if snapped: print 'Error: snapped line'
        print 'line in x[%d]-direction, starting at %s' % \
              (direction, u.grid.min_coor)
        print coor
        print u_coor

        direction = 0
        point = u.grid.min_coor.copy()
        point[direction+1] = u.grid.max_coor[direction+1]
        coor, u_coor, fixed_coor, snapped = \
              u.gridline(u.grid.min_coor, direction)
        if snapped: print 'Error: snapped line'
        print 'line in x[%d]-direction, starting at %s' % \
              (direction, point)
        print coor
        print u_coor

    if u.grid.nsd == 3:
        y_center = (u.grid.max_coor[1] + u.grid.min_coor[1])/2.0
        xc, yc, uc, fixed_coor, snapped = \
            u.gridplane(value=y_center, constant_coor=1)
        print 'Plane y=%g:' % fixed_coor,
        if snapped: print ' (snapped from y=%g)' % y_center
        else: print
        print xc
        print yc
        print uc


def _test4():
    g1 = UniformBoxGrid(min=0, max=1, division=4)
    _testbox(g1)
    spec = '[0,1]x[-1,2] with indices [0:4]x[0:3]'
    g2 = UniformBoxGrid.init_fromstring(spec)
    _testbox(g2)
    g3 = UniformBoxGrid(min=(0,0,-1), max=(1,1,1), division=(4,3,2))
    _testbox(g3)

if __name__ == '__main__':
    test_2Dmesh()
