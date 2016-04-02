# This module provides interpolation for functions defined on a grid.
#
# Written by Konrad Hinsen <konrad.hinsen@cnrs-orleans.fr>
#

"""
Interpolation of functions defined on a grid
"""

from Scientific import N
from Scientific.indexing import index_expression
from Scientific._interpolation import _interpolate
import operator

#
# General interpolating functions.
#
class InterpolatingFunction:

    """X{Function} defined by values on a X{grid} using X{interpolation}

    An interpolating function of M{n} variables with M{m}-dimensional values
    is defined by an M{(n+m)}-dimensional array of values and M{n}
    one-dimensional arrays that define the variables values
    corresponding to the grid points. The grid does not have to be
    equidistant.

    An InterpolatingFunction object has attributes C{real} and C{imag}
    like a complex function (even if its values are real).
    """

    def __init__(self, axes, values, default = None, period = None):
        """
        @param axes: a sequence of one-dimensional arrays, one for each
            variable, specifying the values of the variables at
            the grid points in ascending order
        @type axes: sequence of N.array

        @param values: the function values on the grid
        @type values: N.array

        @param default: the value of the function outside the grid. A value
            of C{None} means that the function is undefined outside
            the grid and that any attempt to evaluate it there
            raises an exception.
        @type default: number or C{None}

        @param period: the period for each of the variables, or C{None} for
            variables in which the function is not periodic.
        @type period: sequence of numbers or C{None}
        """
        if len(axes) > len(values.shape):
            raise ValueError('Inconsistent arguments')
        for i, axis in enumerate(axes):
            if len(axis.shape) != 1:
                raise ValueError("Axes must be 1D arrays")
            if len(axis) != values.shape[i]:
                raise ValueError("Axes must match value array")
            if N.logical_or.reduce(axis[1:]-axis[:-1] <= 0.):
                raise ValueError("Axis values must be distinct and "
                                 "in ascending order")
        self.axes = list(axes)
        self.shape = sum([axis.shape for axis in self.axes], ())
        self.values = values
        self.default = default
        if period is None:
            period = len(self.axes)*[None]
        self.period = period
        if len(self.period) != len(self.axes):
            raise ValueError('Inconsistent arguments')
        for a, p in zip(self.axes, self.period):
            if p is not None and a[0]+p <= a[-1]:
                raise ValueError('Period too short')

    def __call__(self, *points):
        """
        @returns: the function value obtained by linear interpolation
        @rtype: number
        @raise TypeError: if the number of arguments (C{len(points)})
            does not match the number of variables of the function
        @raise ValueError: if the evaluation point is outside of the
            domain of definition and no default value is defined
        """
        if len(points) != len(self.axes):
            raise TypeError('Wrong number of arguments')
        if len(points) == 1:
            # Fast Pyrex implementation for the important special case
            # of a function of one variable with all arrays of type double.
            period = self.period[0]
            if period is None: period = 0.
            try:
                return _interpolate(points[0], self.axes[0],
                                    self.values, period)
            except:
                # Run the Python version if anything goes wrong
                pass
        try:
            neighbours = map(_lookup, points, self.axes, self.period)
        except ValueError, text:
            if self.default is not None:
                return self.default
            else:
                raise ValueError(text)
        slices = sum([item[0] for item in neighbours], ())
        values = self.values[slices]
        for item in neighbours:
            weight = item[1]
            values = (1.-weight)*values[0]+weight*values[1]
        return values

    def __len__(self):
        """
        @returns: number of variables
        @rtype: C{int}
        """
        return len(self.axes[0])

    def __getitem__(self, i):
        """
        @param i: any indexing expression possible for C{N.array}
            that does not use C{N.NewAxis}
        @type i: indexing expression
        @returns: an InterpolatingFunction whose number of variables
            is reduced, or a number if no variable is left
        @rtype: L{InterpolatingFunction} or number
        @raise TypeError: if i is not an allowed index expression
        """
        if isinstance(i, int):
            if len(self.axes) == 1:
                return (self.axes[0][i], self.values[i])
            else:
                return self._constructor(self.axes[1:], self.values[i])
        elif isinstance(i, slice):
            axes = [self.axes[0][i]] + self.axes[1:]
            return self._constructor(axes, self.values[i])
        elif isinstance(i, tuple):
            axes = []
            rest = self.axes[:]
            for item in i:
                if not isinstance(item, int):
                    axes.append(rest[0][item])
                del rest[0]
            axes = axes + rest
            return self._constructor(axes, self.values[i])
        else:
            raise TypeError("illegal index type")

    def __getslice__(self, i, j):
        """
        @param i: lower slice index
        @type i: C{int}
        @param j: upper slice index
        @type j: C{int}
        @returns: an InterpolatingFunction whose number of variables
            is reduced by one, or a number if no variable is left
        @rtype: L{InterpolatingFunction} or number
        """
        axes = [self.axes[0][i:j]] + self.axes[1:]
        return self._constructor(axes, self.values[i:j])

    def __getattr__(self, attr):
        if attr == 'real':
            values = self.values
            try:
                values = values.real
            except ValueError:
                pass
            default = self.default
            try:
                default = default.real
            except:
                pass
            return self._constructor(self.axes, values, default. self.period)
        elif attr == 'imag':
            try:
                values = self.values.imag
            except ValueError:
                values = 0*self.values
            default = self.default
            try:
                default = self.default.imag
            except:
                try:
                    default = 0*self.default
                except:
                    default = None
            return self._constructor(self.axes, values, default, self.period)
        else:
            raise AttributeError(attr)

    def selectInterval(self, first, last, variable=0):
        """
        @param first: lower limit of an axis interval
        @type first: C{float}
        @param last: upper limit of an axis interval
        @type last: C{float}
        @param variable: the index of the variable of the function
            along which the interval restriction is applied
        @type variable: C{int}
        @returns: a new InterpolatingFunction whose grid is restricted
        @rtype: L{InterpolatingFunction}
        """
        x = self.axes[variable]
        c = N.logical_and(N.greater_equal(x, first),
                          N.less_equal(x, last))
        i_axes = self.axes[:variable] + [N.compress(c, x)] + \
                 self.axes[variable+1:]
        i_values = N.compress(c, self.values, variable)
        return self._constructor(i_axes, i_values, None, None)

    def derivative(self, variable = 0):
        """
        @param variable: the index of the variable of the function
            with respect to which the X{derivative} is taken
        @type variable: C{int}
        @returns: a new InterpolatingFunction containing the numerical
            derivative
        @rtype: L{InterpolatingFunction}
        """
        diffaxis = self.axes[variable]
        ai = index_expression[::] + \
             (len(self.values.shape)-variable-1) * index_expression[N.NewAxis]
        period = self.period[variable]
        if period is None:
            ui = variable*index_expression[::] + \
                 index_expression[1::] + index_expression[...]
            li = variable*index_expression[::] + \
                 index_expression[:-1:] + index_expression[...]
            d_values = (self.values[ui]-self.values[li]) / \
                       (diffaxis[1:]-diffaxis[:-1])[ai]
            diffaxis = 0.5*(diffaxis[1:]+diffaxis[:-1])
        else:
            u = N.take(self.values, range(1, len(diffaxis))+[0], axis=variable)
            l = self.values
            ua = N.concatenate((diffaxis[1:], period+diffaxis[0:1]))
            la = diffaxis
            d_values = (u-l)/(ua-la)[ai]
            diffaxis = 0.5*(ua+la)
        d_axes = self.axes[:variable]+[diffaxis]+self.axes[variable+1:]
        d_default = None
        if self.default is not None:
            d_default = 0.
        return self._constructor(d_axes, d_values, d_default, self.period)

    def integral(self, variable = 0):
        """
        @param variable: the index of the variable of the function
            with respect to which the X{integration} is performed
        @type variable: C{int}
        @returns: a new InterpolatingFunction containing the numerical
            X{integral}. The integration constant is defined such that
            the integral at the first grid point is zero.
        @rtype: L{InterpolatingFunction}
        """
        if self.period[variable] is not None:
            raise ValueError('Integration over periodic variables not defined')
        intaxis = self.axes[variable]
        ui = variable*index_expression[::] + \
             index_expression[1::] + index_expression[...]
        li = variable*index_expression[::] + \
             index_expression[:-1:] + index_expression[...]
        uai = index_expression[1::] + (len(self.values.shape)-variable-1) * \
              index_expression[N.NewAxis]
        lai = index_expression[:-1:] + (len(self.values.shape)-variable-1) * \
              index_expression[N.NewAxis]
        i_values = 0.5*N.add.accumulate((self.values[ui]
                                               +self.values[li])* \
                                              (intaxis[uai]-intaxis[lai]),
                                              variable)
        s = list(self.values.shape)
        s[variable] = 1
        z = N.zeros(tuple(s))
        return self._constructor(self.axes,
                                 N.concatenate((z, i_values), variable),
                                 None)

    def definiteIntegral(self, variable = 0):
        """
        @param variable: the index of the variable of the function
            with respect to which the X{integration} is performed
        @type variable: C{int}
        @returns: a new InterpolatingFunction containing the numerical
            X{integral}. The integration constant is defined such that
            the integral at the first grid point is zero. If the original
            function has only one free variable, the definite integral
            is a number
        @rtype: L{InterpolatingFunction} or number
        """
        if self.period[variable] is not None:
            raise ValueError('Integration over periodic variables not defined')
        intaxis = self.axes[variable]
        ui = variable*index_expression[::] + \
             index_expression[1::] + index_expression[...]
        li = variable*index_expression[::] + \
             index_expression[:-1:] + index_expression[...]
        uai = index_expression[1::] + (len(self.values.shape)-variable-1) * \
              index_expression[N.NewAxis]
        lai = index_expression[:-1:] + (len(self.values.shape)-variable-1) * \
              index_expression[N.NewAxis]
        i_values = 0.5*N.add.reduce((self.values[ui]+self.values[li]) * \
                   (intaxis[uai]-intaxis[lai]), variable)
        if len(self.axes) == 1:
            return i_values
        else:
            i_axes = self.axes[:variable] + self.axes[variable+1:]
            return self._constructor(i_axes, i_values, None)

    def __abs__(self):
        values = abs(self.values)
        try:
            default = abs(self.default)
        except:
            default = self.default
        return self._constructor(self.axes, values, default)

    def _mathfunc(self, function):
        if self.default is None:
            default = None
        else:
            default = function(self.default)
        return self._constructor(self.axes, function(self.values), default)

    def exp(self):
        return self._mathfunc(N.exp)

    def log(self):
        return self._mathfunc(N.log)

    def sqrt(self):
        return self._mathfunc(N.sqrt)

    def sin(self):
        return self._mathfunc(N.sin)

    def cos(self):
        return self._mathfunc(N.cos)

    def tan(self):
        return self._mathfunc(N.tan)

    def sinh(self):
        return self._mathfunc(N.sinh)

    def cosh(self):
        return self._mathfunc(N.cosh)

    def tanh(self):
        return self._mathfunc(N.tanh)

    def arcsin(self):
        return self._mathfunc(N.arcsin)

    def arccos(self):
        return self._mathfunc(N.arccos)

    def arctan(self):
        return self._mathfunc(N.arctan)

InterpolatingFunction._constructor = InterpolatingFunction

#
# Interpolating function on data in netCDF file
#
class NetCDFInterpolatingFunction(InterpolatingFunction):

    """Function defined by values on a grid in a X{netCDF} file

    A subclass of L{InterpolatingFunction}.
    """

    def __init__(self, filename, axesnames, variablename, default = None,
                 period = None):
        """
        @param filename: the name of the netCDF file
        @type filename: C{str}

        @param axesnames: the names of the netCDF variables that contain the
            axes information
        @type axesnames: sequence of C{str}

        @param variablename: the name of the netCDF variable that contains
            the data values
        @type variablename: C{str}

        @param default: the value of the function outside the grid. A value
            of C{None} means that the function is undefined outside
            the grid and that any attempt to evaluate it there
            raises an exception.
        @type default: number or C{None}

        @param period: the period for each of the variables, or C{None} for
            variables in which the function is not periodic.
        @type period: sequence of numbers or C{None}
        """
        from Scientific.IO.NetCDF import NetCDFFile
        self.file = NetCDFFile(filename, 'r')
        self.axes = [self.file.variables[n] for n in axesnames]
        for a in self.axes:
            if len(a.dimensions) != 1:
                raise ValueError("axes must be 1d arrays")
        self.values = self.file.variables[variablename]
        if tuple(v.dimensions[0] for v in self.axes) != self.values.dimensions:
            raise ValueError("axes and values have incompatible dimensions")
        self.default = default
        self.shape = ()
        for axis in self.axes:
            self.shape = self.shape + axis.shape
        if period is None:
            period = len(self.axes)*[None]
        self.period = period
        if len(self.period) != len(self.axes):
            raise ValueError('Inconsistent arguments')
        for a, p in zip(self.axes, self.period):
            if p is not None and a[0]+p <= a[-1]:
                raise ValueError('Period too short')

NetCDFInterpolatingFunction._constructor = InterpolatingFunction


# Helper functions

def _lookup(point, axis, period):
    if period is None:
        j = int(N.int_sum(N.less_equal(axis, point)))
        if j == len(axis):
            if N.fabs(point - axis[j-1]) < 1.e-9:
                return index_expression[j-2:j:1], 1.
            else:
                j = 0
        if j == 0:
            raise ValueError('Point outside grid of values')
        i = j-1
        weight = (point-axis[i])/(axis[j]-axis[i])
        return index_expression[i:j+1:1], weight
    else:
        point = axis[0] + (point-axis[0]) % period
        j = int(N.int_sum(N.less_equal(axis, point)))
        i = j-1
        if j == len(axis):
            weight = (point-axis[i])/(axis[0]+period-axis[i])
            return index_expression[0:i+1:i], 1.-weight
        else:
            weight = (point-axis[i])/(axis[j]-axis[i])
            return index_expression[i:j+1:1], weight

def _combinations(axes):
    if len(axes) == 1:
        return map(lambda x: (x,), axes[0])
    else:
        rest = _combinations(axes[1:])
        l = []
        for x in axes[0]:
            for y in rest:
                l.append((x,)+y)
        return l


# Test code

if __name__ == '__main__':

##     axis = N.arange(0,1.1,0.1)
##     values = N.sqrt(axis)
##     s = InterpolatingFunction((axis,), values)
##     print s(0.22), N.sqrt(0.22)
##     sd = s.derivative()
##     print sd(0.35), 0.5/N.sqrt(0.35)
##     si = s.integral()
##     print si(0.42), (0.42**1.5)/1.5
##     print s.definiteIntegral()
##     values = N.sin(axis[:,N.NewAxis])*N.cos(axis)
##     sc = InterpolatingFunction((axis,axis),values)
##     print sc(0.23, 0.77), N.sin(0.23)*N.cos(0.77)

    axis = N.arange(20)*(2.*N.pi)/20.
    values = N.sin(axis)
    s = InterpolatingFunction((axis,), values, period=(2.*N.pi,))
    c = s.derivative()
    for x in N.arange(0., 15., 1.):
        print x
        print N.sin(x), s(x)
        print N.cos(x), c(x)
