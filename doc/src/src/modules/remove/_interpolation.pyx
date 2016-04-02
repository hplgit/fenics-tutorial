# Implementation of function interpolation in Cython
#
# Written by Konrad Hinsen
#

import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double fmod(double x, double y)

def _interpolate(double x, np.ndarray axis, np.ndarray values, double period):
    cdef double *axis_d, *values_d
    cdef int axis_s, values_s
    cdef int npoints
    cdef int i1, i2, i
    cdef double w1, w2

    assert axis.descr.type_num == np.NPY_DOUBLE \
           and values.descr.type_num == np.NPY_DOUBLE
    npoints = axis.shape[0]
    axis_s = axis.strides[0]/sizeof(double)
    axis_d = <double *>axis.data
    values_s = values.strides[0]/sizeof(double)
    values_d = <double *>values.data

    if period > 0.: # periodic
        x = axis_d[0] + fmod(x-axis_d[0], period)
    else:
        if x < axis_d[0]-1.e-9 or x > axis_d[(npoints-1)*axis_s]+1.e-9:
            raise ValueError('Point outside grid of values')
    i1 = 0
    i2 = npoints-1
    while i2-i1 > 1:
        i = (i1+i2)/2
        if axis_d[i*axis_s] > x:
            i2 = i
        else:
            i1 = i
    if period > 0. and x > axis_d[i2*axis_s]:
        i1 = npoints-1
        i2 = 0
        w2 = (x-axis_d[i1*axis_s])/(axis_d[i2*axis_s]+period-axis_d[i1*axis_s])
    elif period > 0. and x < axis_d[i1*axis_s]:
        i1 = npoints-1
        i2 = 0
        w2 = (x-axis_d[i1*axis_s]+period) \
              / (axis_d[i2*axis_s]-axis_d[i1*axis_s]+period)
    else:
        w2 = (x-axis_d[i1*axis_s])/(axis_d[i2*axis_s]-axis_d[i1*axis_s])
    w1 = 1.-w2
    return w1*values_d[i1*values_s]+w2*values_d[i2*values_s]
