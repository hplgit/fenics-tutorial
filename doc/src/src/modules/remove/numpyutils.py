"""

Functionality of this module that extends Numerical Python
==========================================================

 - solve_tridiag_linear_system:
           returns the solution of a tridiagonal linear system
 - wrap2callable:
           tool for turning constants, discrete data, string
           formulas, function objects, or plain functions
           into an object that behaves as a function
 - NumPy_array_iterator:
           allows iterating over all array elements using
           a single, standard for loop (for value, index in iterator),
           has some additional features compared with numpy.ndenumerate
 - asarray_cpwarn:
           as ``numpy.asarray(a)``, but a warning or exception is issued if
           the array a is copied
 - meshgrid:
           extended version of ``numpy.meshgrid`` to 1D, 2D and 3D grids,
           with sparse or dense coordinate arrays and matrix or grid
           indexing
 - ndgrid:
           same as calling ``meshgrid`` with indexing='ij' (matrix indexing)
 - float_eq:
           ``operator ==`` for float operands with tolerance,
           ``float_eq(a,b,tol)`` means ``abs(a-b) < tol``
           works for both scalar and array arguments
           (similar functions for other operations exists:
           ``float_le``, ``float_lt``, ``float_ge``, ``float_gt``,
           ``float_ne``)
 - cut_noise:
           set all small (noise) elements of an array to zero
 - matrix_rank:
           compute the rank of a matrix
 - orth:
           compute an orthonormal basis from a matrix (taken from
           ``scipy.linalg`` to avoid ``scipy`` dependence)
 - null:
           compute the null space of a matrix
 - norm_L2, norm_l2, norm_L1, norm_l1, norm_inf:
           discrete and continuous norms for multi-dimensional arrays
           viewed as vectors
 - approximate_Jacobian:
           compute finite difference approximation of the Jacobian
           of a nonlinear vector function
 - compute_historgram:
           return x and y arrays of a histogram, given a vector of samples
 - seq:
           ``seq(a,b,s, [type])`` computes numbers from ``a`` up to and
           including ``b`` in steps of s and (default) type ``float_``;
 - iseq:
           as ``seq``, but integer counters are computed
           (``iseq`` is an alternative to range where the
           upper limit is included in the sequence - this can
           be important for direct mapping of indices between
           mathematics and Python code);
"""

if __name__.find('numpyutils') != -1:
    from numpy import *

#else if name is some other module name:
# this file is included in numpytools.py (through a preprocessing step)
# and the code below then relies on previously imported Numerical Python
# modules (Numeric, numpy, numarray)

import operator
from .FloatComparison import float_eq, float_ne, float_lt, float_le, \
     float_gt, float_ge
import collections
from functools import reduce
from Heaviside import *
import numpy

def meshgrid(x=None, y=None, z=None, sparse=False, indexing='xy',
             memoryorder=None):
    """Now just a call to numpy.meshgrid (for numpy version >= 1.7)."""
    # In the past, numpy.meshgrid only worked for 2D problems and this
    # was the generalization that is now incorporated in numpy v1.7.
    args = []
    if x is not None:
        args.append(x)
    if y is not None:
        args.append(y)
    if z is not None:
        args.append(z)
    return numpy.meshgrid(*args, indexing=indexing, sparse=sparse)

def meshgrid_scitools(x=None, y=None, z=None, sparse=False, indexing='xy',
                      memoryorder=None):
    """
    Original scitools.std.meshgrid:

    Extension of ``numpy.meshgrid`` to 1D, 2D and 3D problems, and also
    support of both "matrix" and "grid" numbering.
    (See below how it relates to ``meshgrid``, ``ogrid``, and ``mgrid``
    in ``numpy``.)

    This extended version makes 1D/2D/3D coordinate arrays for
    vectorized evaluations of 1D/2D/3D scalar/vector fields over
    1D/2D/3D grids, given one-dimensional coordinate arrays x, y,
    and/or, z.

    >>> x=linspace(0,1,3)        # coordinates along x axis
    >>> y=linspace(0,1,2)        # coordinates along y axis
    >>> xv, yv = meshgrid(x,y)   # extend x and y for a 2D xy grid
    >>> xv
    array([[ 0. ,  0.5,  1. ],
           [ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]])
    >>> xv, yv = meshgrid(x,y, sparse=True)  # make sparse output arrays
    >>> xv
    array([[ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.],
           [ 1.]])

    >>> # 2D slice of a 3D grid, with z=const:
    >>> z=5
    >>> xv, yv, zc = meshgrid(x,y,z)
    >>> xv
    array([[ 0. ,  0.5,  1. ],
           [ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]])
    >>> zc
    5

    >>> # 2D slice of a 3D grid, with x=const:
    >>> meshgrid(2,y,x)
    (2, array([[ 0.,  1.],
           [ 0.,  1.],
           [ 0.,  1.]]), array([[ 0. ,  0. ],
           [ 0.5,  0.5],
           [ 1. ,  1. ]]))
    >>> meshgrid(0,1,5, sparse=True)  # just a 3D point
    (0, 1, 5)
    >>> meshgrid(y)      # 1D grid; y is just returned
    array([ 0.,  1.])
    >>> meshgrid(x,y, indexing='ij')  # change to matrix/Matlab-style indexing
    (array([[ 0. ,  0. ],
           [ 0.5,  0.5],
           [ 1. ,  1. ]]), array([[ 0.,  1.],
           [ 0.,  1.],
           [ 0.,  1.]]))

    For use with ``contour`` plotting commands, one needs the
    ``indexing='ij'`` argument, or equivalently one can call
    ``ndgrid(x,y)`` (which means ``meshgrid(x,x, indexing='ij'``).

    Why does SciTools has its own meshgrid function when numpy has
    three similar functions, ``mgrid``, ``ogrid``, and ``meshgrid``?
    The ``meshgrid`` function in numpy is limited to two dimensions
    only, while the SciTools version can also work with 3D and 1D
    grids. In addition, the numpy version of ``meshgrid`` has no
    option for generating sparse grids to conserve memory, like we
    have in SciTools by specifying the ``sparse`` argument.

    Moreover, the numpy functions ``mgrid`` and ``ogrid`` do provide
    support for, respectively, full and sparse n-dimensional
    meshgrids, however, these functions use slices to generate the
    meshgrids rather than one-dimensional coordinate arrays such as in
    Matlab. With slices, the user does not have the option to generate
    meshgrid with, e.g., irregular spacings, like::

    >>> x = array([-1,-0.5,1,4,5], float)
    >>> y = array([0,-2,-5], float)
    >>> xv, yv = meshgrid(x, y, sparse=False)

    >>> xv
    array([[-1. , -0.5,  1. ,  4. ,  5. ],
           [-1. , -0.5,  1. ,  4. ,  5. ],
           [-1. , -0.5,  1. ,  4. ,  5. ]])

    >>> yv
    array([[ 0.,  0.,  0.,  0.,  0.],
           [-2., -2., -2., -2., -2.],
           [-5., -5., -5., -5., -5.]])


    In addition to the reasons mentioned above, the ``meshgrid``
    function in numpy supports only Cartesian indexing, i.e., x and y,
    not matrix indexing, i.e., rows and columns (on the other hand,
    ``mgrid`` and ``ogrid`` support only matrix indexing). The
    ``meshgrid`` function in SciTools supports both indexing
    conventions through the ``indexing`` keyword argument. Giving the
    string ``'ij'`` returns a meshgrid with matrix indexing, while
    ``'xy'`` returns a meshgrid with Cartesian indexing. The
    difference is illustrated by the following code snippet::

      nx = 10
      ny = 15

      x = linspace(-2,2,nx)
      y = linspace(-2,2,ny)

      xv, yv = meshgrid(x, y, sparse=False, indexing='ij')
      for i in range(nx):
          for j in range(ny):
              # treat xv[i,j], yv[i,j]

      xv, yv = meshgrid(x, y, sparse=False, indexing='xy')
      for i in range(nx):
          for j in range(ny):
              # treat xv[j,i], yv[j,i]

    It is not entirely true that matrix indexing is not supported by the
    ``meshgrid`` function in numpy because we can just switch the order of
    the first two input and output arguments::

    >>> yv, xv = numpy.meshgrid(y, x)
    >>> # same as:
    >>> xv, yv = meshgrid(x, y, indexing='ij')

    However, we think it is clearer to have the logical "x, y"
    sequence on the left-hand side and instead adjust a keyword argument.
    """

    import types
    def fixed(coor):
        return isinstance(coor, (float, complex, int, type(None)))

    if not fixed(x):
        x = asarray(x)
    if not fixed(y):
        y = asarray(y)
    if not fixed(z):
        z = asarray(z)

    def arr1D(coor):
        try:
            if len(coor.shape) == 1:
                return True
            else:
                return False
        except AttributeError:
            return False

    # if two of the arguments are fixed, we have a 1D grid, and
    # the third argument can be reused as is:

    if arr1D(x) and fixed(y) and fixed(z):
        return x
    if fixed(x) and arr1D(y) and fixed(z):
        return y
    if fixed(x) and fixed(y) and arr1D(z):
        return z

    # if x,y,z are identical, make copies:
    try:
        if y is x: y = x.copy()
        if z is x: z = x.copy()
        if z is y: z = y.copy()
    except AttributeError:  # x, y, or z not numpy array
        pass

    if memoryorder is not None:
        import warnings
        msg = "Keyword argument 'memoryorder' is deprecated and will be " \
              "removed in the future. Please use the 'indexing' keyword " \
              "argument instead."
        warnings.warn(msg, DeprecationWarning)
        if memoryorder == 'xyz':
            indexing = 'ij'
        else:
            indexing = 'xy'

    # If the keyword argument sparse is set to False, the full N-D matrix
    # (not only the 1-D vector) should be returned. The mult_fact variable
    # should then be updated as necessary.
    mult_fact = 1

    # if only one argument is fixed, we have a 2D grid:
    if arr1D(x) and arr1D(y) and fixed(z):
        if indexing == 'ij':
            if not sparse:
                mult_fact = ones((len(x),len(y)))
            if z is None:
                return x[:,newaxis]*mult_fact, y[newaxis,:]*mult_fact
            else:
                return x[:,newaxis]*mult_fact, y[newaxis,:]*mult_fact, z
        else:
            if not sparse:
                mult_fact = ones((len(y),len(x)))
            if z is None:
                return x[newaxis,:]*mult_fact, y[:,newaxis]*mult_fact
            else:
                return x[newaxis,:]*mult_fact, y[:,newaxis]*mult_fact, z

    if arr1D(x) and fixed(y) and arr1D(z):
        if indexing == 'ij':
            if not sparse:
                mult_fact = ones((len(x),len(z)))
            if y is None:
                return x[:,newaxis]*mult_fact, z[newaxis,:]*mult_fact
            else:
                return x[:,newaxis]*mult_fact, y, z[newaxis,:]*mult_fact
        else:
            if not sparse:
                mult_fact = ones((len(z),len(x)))
            if y is None:
                return x[newaxis,:]*mult_fact, z[:,newaxis]*mult_fact
            else:
                return x[newaxis,:]*mult_fact, y, z[:,newaxis]*mult_fact

    if fixed(x) and arr1D(y) and arr1D(z):
        if indexing == 'ij':
            if not sparse:
                mult_fact = ones((len(y),len(z)))
            if x is None:
                return y[:,newaxis]*mult_fact, z[newaxis,:]*mult_fact
            else:
                return x, y[:,newaxis]*mult_fact, z[newaxis,:]*mult_fact
        else:
            if not sparse:
                mult_fact = ones((len(z),len(y)))
            if x is None:
                return y[newaxis,:]*mult_fact, z[:,newaxis]*mult_fact
            else:
                return x, y[newaxis,:]*mult_fact, z[:,newaxis]*mult_fact

    # or maybe we have a full 3D grid:
    if arr1D(x) and arr1D(y) and arr1D(z):
        if indexing == 'ij':
            if not sparse:
                mult_fact = ones((len(x),len(y),len(z)))
            return x[:,newaxis,newaxis]*mult_fact, \
                   y[newaxis,:,newaxis]*mult_fact, \
                   z[newaxis,newaxis,:]*mult_fact
        else:
            if not sparse:
                mult_fact = ones((len(y),len(x),len(z)))
            return x[newaxis,:,newaxis]*mult_fact, \
                   y[:,newaxis,newaxis]*mult_fact, \
                   z[newaxis,newaxis,:]*mult_fact

    # at this stage we assume that we just have scalars:
    l = []
    if x is not None:
        l.append(x)
    if y is not None:
        l.append(y)
    if z is not None:
        l.append(z)
    if len(l) == 1:
        return l[0]
    else:
        return tuple(l)


def ndgrid(*args,**kwargs):
    """
    Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
    ``meshgrid`` for documentation).
    """
    kwargs['indexing'] = 'ij'
    return meshgrid(*args,**kwargs)

def approximate_Jacobian(f, x, f_args, h=1.0E-4):
    """
    Compute approximate Jacobian of f(x, *f_args) at x.
    Method: forward finite difference approximation with step h.
    """
    x = np.asarray(x)
    f0 = np.asarray(f(x, *f_args))  # f may return list/tuple
    J = np.zeros((len(x), len(f0)))
    dx = np.zeros_like(x)
    for i in range(len(x)):
       dx[i] = h
       J[i] = (np.asarray(f(x+dx, *f_args)) - f0)/h
       dx[i] = 0.0
    return J.transpose()

def length(a):
    """Return the length of the largest dimension of array a."""
    return max(a.shape)

def cut_noise(a, tol=1E-10):
    """
    Set elements in array a to zero if the absolute value is
    less than tol.
    """
    a[abs(a) < tol] = 0
    return a


def Gram_Schmidt1(vecs, row_wise_storage=True):
    """
    Apply the Gram-Schmidt orthogonalization algorithm to a set
    of vectors. vecs is a two-dimensional array where the vectors
    are stored row-wise, or vecs may be a list of vectors, where
    each vector can be a list or a one-dimensional array.
    An array basis is returned, where basis[i,:] (row_wise_storage
    is True) or basis[:,i] (row_wise_storage is False) is the i-th
    orthonormal vector in the basis.

    This function does not handle null vectors, see Gram_Schmidt
    for a (slower) function that does.
    """
    from numpy.linalg import inv
    from math import sqrt

    vecs = asarray(vecs)  # transform to array if list of vectors
    m, n = vecs.shape
    basis = array(transpose(vecs))
    eye = identity(n).astype(float)

    basis[:,0] /= sqrt(dot(basis[:,0], basis[:,0]))
    for i in range(1, m):
	v = basis[:,i]/sqrt(dot(basis[:,i], basis[:,i]))
    	U = basis[:,:i]
	P = eye - dot(U, dot(inv(dot(transpose(U), U)), transpose(U)))
	basis[:, i] = dot(P, v)
	basis[:, i] /= sqrt(dot(basis[:, i], basis[:, i]))

    return transpose(basis) if row_wise_storage else basis


def Gram_Schmidt(vecs, row_wise_storage=True, tol=1E-10,
                 normalize=False, remove_null_vectors=False,
                 remove_noise=False):
    """
    Apply the Gram-Schmidt orthogonalization algorithm to a set
    of vectors. vecs is a two-dimensional array where the vectors
    are stored row-wise, or vecs may be a list of vectors, where
    each vector can be a list or a one-dimensional array.

    The argument tol is a tolerance for null vectors (the absolute
    value of all elements must be less than tol to have a null
    vector).

    If normalize is True, the orthogonal vectors are normalized to form
    an orthonormal basis.

    If remove_null_vectors is True, all null vectors are removed from
    the resulting basis.

    If remove_noise is True, all elements whose absolute values are
    less than tol are set to zero.

    An array basis is returned, where basis[i,:] (row_wise_storage
    is True) or basis[:,i] (row_wise_storage is False) is the i-th
    orthogonal vector in the basis.

    This function handles null vectors, see Gram_Schmidt1
    for a (faster) function that does not.
    """
    # The algorithm below views vecs as a matrix A with the vectors
    # stored as columns:
    vecs = asarray(vecs)  # transform to array if list of vectors
    if row_wise_storage:
        A = transpose(vecs).copy()
    else:
        A = vecs.copy()

    m, n = A.shape
    V = zeros((m,n))

    for j in xrange(n):
        v0 = A[:,j]
        v = v0.copy()
        for i in xrange(j):
            vi = V[:,i]

            if (abs(vi) > tol).any():
                v -= (vdot(v0,vi)/vdot(vi,vi))*vi
        V[:,j] = v

    if remove_null_vectors:
        indices = [i for i in xrange(n) if (abs(V[:,i]) < tol).all()]
        V = V[ix_(range(m), indices)]

    if normalize:
        for j in xrange(V.shape[1]):
            V[:,j] /= linalg.norm(V[:,j])

    if remove_noise:
        V = cut_noise(V, tol)

    return transpose(V) if row_wise_storage else V


def matrix_rank(A):
    """
    Returns the rank of a matrix A (rank means an estimate of
    the number of linearly independent rows or columns).
    """
    A = asarray(A)
    u, s, v = svd(A)
    maxabs = norm(x)
    maxdim = max(A.shape)
    tol = maxabs*maxdim*1E-13
    r = s > tol
    return sum(r)


def orth(A):
    """
    (Plain copy from scipy.linalg.orth - this one here applies numpy.svd
    and avoids the need for having scipy installed.)

    Construct an orthonormal basis for the range of A using SVD.

    @param A: array, shape (M, N)
    @return:
        Q : array, shape (M, K)
        Orthonormal basis for the range of A.
        K = effective rank of A, as determined by automatic cutoff

    see also svd (singular value decomposition of a matrix in scipy.linalg)
    """
    u,s,vh = svd(A)
    M,N = A.shape
    tol = max(M,N)*numpy.amax(s)*eps
    num = numpy.sum(s > tol,dtype=int)
    Q = u[:,:num]
    return Q


def null(A, tol=1e-10, row_wise_storage=True):
    """
    Return the null space of a matrix A.
    If row_wise_storage is True, a two-dimensional array where the
    vectors that span the null space are stored as rows, otherwise
    they are stored as columns.

    Code by Bastian Weber based on code by Robert Kern and Ryan Krauss.
    """
    n, m = A.shape
    if n > m :
        return transpose(null(transpose(A), tol))

    u, s, vh = linalg.svd(A)
    s = append(s, zeros(m))[0:m]
    null_mask = (s <= tol)
    null_space = compress(null_mask, vh, axis=0)
    null_space = conjugate(null_space)  # in case of complex values
    if row_wise_storage:
        return null_space
    else:
        return transpose(null_space)


# the norm_* functions also work for arrays with dimensions larger than 2,
# in contrast to (most of) the numpy.linalg.norm functions

def norm_l2(u):
    """
    Standard l2 norm of a multi-dimensional array u viewed as a vector.
    """
    return linalg.norm(u.ravel())

def norm_L2(u):
    """
    L2 norm of a multi-dimensional array u viewed as a vector
    (norm is norm_l2/n, where n is length of u (no of elements)).

    If u holds function values and the norm of u is supposed to
    approximate an integral (L2 norm) of the function, this (and
    not norm_l2) is the right norm function to use.
    """
    return norm_l2(u)/sqrt(float(u.size))

def norm_l1(u):
    """
    l1 norm of a multi-dimensional array u viewed as a vector:
    ``linalg.norm(u.ravel(),1)``.
    """
    #return sum(abs(u.ravel()))
    return linalg.norm(u.ravel(),1)

def norm_L1(u):
    """
    L1 norm of a multi-dimensional array u viewed as a vector:
    ``norm_l1(u)/float(u.size)``.

    If *u* holds function values and the norm of u is supposed to
    approximate an integral (L1 norm) of the function, this (and
    not norm_l1) is the right norm function to use.
    """
    return norm_l1(u)/float(u.size)

def norm_inf(u):
    """Infinity/max norm of a multi-dimensional array u viewed as a vector."""
    #return abs(u.ravel()).max()
    return linalg.norm(u.ravel(), inf)


def solve_tridiag_linear_system(A, b):
    """
    Solve an n times n tridiagonal linear system of the form::

     A[0,1]*x[0] + A[0,2]*x[1]                                        = 0
     A[1,0]*x[0] + A[1,1]*x[1] + A[1,2]*x[2]                          = 0
     ...
     ...
              A[k,0]*x[k-1] + A[k,1]*x[k] + A[k,2]*x[k+1]             = 0
     ...
                  A[n-2,0]*x[n-3] + A[n-2,1]*x[n-2] + A[n-2,2]*x[n-1] = 0
     ...
                                    A[n-1,0]*x[n-2] + A[n-1,1]*x[n-1] = 0

    The diagonal of the coefficent matrix is stored in A[:,1],
    the subdiagonal is stored in A[1:,0], and the superdiagonal
    is stored in A[:-1,2].
    """

    #The storage is not memory friendly in Python/C (diagonals stored
    #columnwise in A), but if A is sent to F77 for high-performance
    #computing, a copy is taken and the F77 routine works with the
    #same algorithm and hence optimal (columnwise traversal)
    #Fortran storage.

    c, d = factorize_tridiag_matrix(A)
    return solve_tridiag_factored_system(b, A, c, d)


def factorize_tridiag_matrix(A):
    """
    Perform the factorization step only in solving a tridiagonal
    linear system. See the function solve_tridiag_linear_system
    for how the matrix *A* is stored.
    Two arrays, *c* and *d*, are returned, and these represent,
    together with superdiagonal *A[:-1,2]*, the factorized form of
    *A*. To solve a system with ``solve_tridiag_factored_system``,
    *A*, *c*, and *d* must be passed as arguments.
    """
    n = len(b)
    # scratch arrays:
    d = zeros(n, 'd');  c = zeros(n, 'd');  m = zeros(n, 'd')

    d[0] = A[0,1]
    c[0] = b[0]

    for k in iseq(start=1, stop=n-1, inc=1):
        m[k] = A[k,0]/d[k-1]
        d[k] = A[k,1] - m[k]*A[k-1,2]
        c[k] = b[k] - m[k]*c[k-1]
    return c, d


def solve_tridiag_factored_system(b, A, c, d):
    """
    The backsubsitution part of solving a tridiagonal linear system.
    The right-hand side is b, while *A*, *c*, and *d* represent the
    factored matrix (see the factorize_tridiag_matrix function).
    The solution x to A*x=b is returned.
    """
    n = len(b)
    x = zeros(n, 'd')  # solution

    # back substitution:
    x[n-1] = c[n-1]/d[n-1]
    for k in iseq(start=n-2, stop=0, inc=-1):
        x[k] = (c[k] - A[k,2]*x[k+1])/d[k]
    return x



try:
    import Pmw
    class NumPy2BltVector(Pmw.Blt.Vector):
        """
        Copy a numpy array to a BLT vector:
        # a: some numpy array
        b = NumPy2BltVector(a)  # b is BLT vector
        g = Pmw.Blt.Graph(someframe)
        # send b to g for plotting
        """
        def __init__(self, array):
            Pmw.Blt.Vector.__init__(self, len(array))
            self.set(tuple(array))  # copy elements
except:
    class NumPy2BltVector:
        def __init__(self, array):
            raise ImportError("Python is not working properly with BLT")

try:
    from scitools.StringFunction import StringFunction
except:
    pass  # wrap2callable may not work


class WrapNo2Callable:
    """Turn a number (constant) into a callable function."""
    def __init__(self, constant):
        self.constant = constant
        self._array_shape = None

    def __call__(self, *args):
        """
        >>> w = WrapNo2Callable(4.4)
        >>> w(99)
        4.4000000000000004
        >>> # try vectorized computations:
        >>> x = linspace(1, 4, 4)
        >>> y = linspace(1, 2, 2)
        >>> xv = x[:,NewAxis]; yv = y[NewAxis,:]
        >>> xv + yv
        array([[ 2.,  3.],
               [ 3.,  4.],
               [ 4.,  5.],
               [ 5.,  6.]])
        >>> w(xv, yv)
        array([[ 4.4,  4.4],
               [ 4.4,  4.4],
               [ 4.4,  4.4],
               [ 4.4,  4.4]])

        If you want to call such a function object with space-time
        arguments and vectorized expressions, make sure the time
        argument is not the first argument. That is,
        w(xv, yv, t) is fine, but w(t, xv, yv) will return 4.4,
        not the desired array!
        """
        if isinstance(args[0], (float, int, complex)):
            # scalar version:
            # (operator.isNumberType(args[0]) cannot be used as it is
            # true also for numpy arrays
            return self.constant
        else: # assume numpy array
            if self._array_shape is None:
                self._set_array_shape()
            else:
                r = self.constant*ones(self._array_shape, 'd')
                # could store r (allocated once) and just return reference
                return r

    def _set_array_shape(self, arg):
        # vectorized version:
        r = arg.copy()
        # to get right dimension of the return array,
        # compute with args in a simple formula (sum of args)
        for a in args[1:]:
            r = r + a  # in-place r+= won't work
            # (handles x,y,t - the last t just adds a constant)
            # an argument sequence t, x, y  will fail (1st arg
            # is not a numpy array)
        self._array_shape = r.shape

    # The problem with this class is that, in the vectorized version,
    # the array shape is determined in the first call, i.e., later
    # calls may return an array with the wrong shape if the shape of
    # the input arguments change! Sometimes, when called along boundaries
    # of grids, the shape may change so the next implementation is
    # slower and safer.

class WrapNo2Callable:
    """Turn a number (constant) into a callable function."""
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, *args):
        """
        >>> w = WrapNo2Callable(4.4)
        >>> w(99)
        4.4000000000000004
        >>> # try vectorized computations:
        >>> x = linspace(1, 4, 4)
        >>> y = linspace(1, 2, 2)
        >>> xv = x[:,NewAxis]; yv = y[NewAxis,:]
        >>> xv + yv
        array([[ 2.,  3.],
               [ 3.,  4.],
               [ 4.,  5.],
               [ 5.,  6.]])
        >>> w(xv, yv)
        array([[ 4.4,  4.4],
               [ 4.4,  4.4],
               [ 4.4,  4.4],
               [ 4.4,  4.4]])

        If you want to call such a function object with space-time
        arguments and vectorized expressions, make sure the time
        argument is not the first argument. That is,
        w(xv, yv, t) is fine, but w(t, xv, yv) will return 4.4,
        not the desired array!

        """
        if isinstance(args[0], (float, int, complex)):
            # scalar version:
            return self.constant
        else:
            # vectorized version:
            r = args[0].copy()
            # to get right dimension of the return array,
            # compute with args in a simple formula (sum of args)
            for a in args[1:]:
                r = r + a  # in-place r+= won't work
                # (handles x,y,t - the last t just adds a constant)
            r[:] = self.constant
            return r


class WrapDiscreteData2Callable:
    """
    Turn discrete data on a uniform grid into a callable function,
    i.e., equip the data with an interpolation function.

    >>> x = linspace(0, 1, 11)
    >>> y = 1+2*x
    >>> f = WrapDiscreteData2Callable((x,y))
    >>> # or just use the wrap2callable generic function:
    >>> f = wrap2callable((x,y))
    >>> f(0.5)   # evaluate f(x) by interpolation
    1.5
    >>> f(0.5, 0.1)  # discrete data with extra time prm: f(x,t)
    1.5
    """
    def __init__(self, data):
        self.data = data  # (x,y,f) data for an f(x,y) function

        from scitools.misc import import_module
        InterpolatingFunction = import_module(
            'Scientific.Functions.Interpolation', 'InterpolatingFunction')
        import Scientific
        v = Scientific.__version__
        target = '2.9.1'
        if v < target:
            raise ImportError(
                'ScientificPython is in (old) version %s, need %s' \
                % (v, target))

        self.interpolating_function = \
             InterpolatingFunction(self.data[:-1], self.data[-1])
        self.ndims = len(self.data[:-1])  # no of spatial dim.

    def __call__(self, *args):
        # allow more arguments (typically time) after spatial pos.:
        args = args[:self.ndims]
        # args can be tuple of scalars (point) or tuple of vectors
        if isinstance(args[0], (float, int, complex)):
            return self.interpolating_function(*args)
        else:
            # args is tuple of vectors; Interpolation must work
            # with one point at a time:
            r = [self.interpolating_function(*a) for a in zip(*args)]
            return array(r)  # wrap in numpy array


def wrap2callable(f, **kwargs):
    """
    Allow constants, string formulas, discrete data points,
    user-defined functions and (callable) classes to be wrapped
    in a new callable function. That is, all the mentioned data
    structures can be used as a function, usually of space and/or
    time.
    (kwargs is used for string formulas)

    >>> f1 = wrap2callable(2.0)
    >>> f1(0.5)
    2.0
    >>> f2 = wrap2callable('1+2*x')
    >>> f2(0.5)
    2.0
    >>> f3 = wrap2callable('1+2*t', independent_variable='t')
    >>> f3(0.5)
    2.0
    >>> f4 = wrap2callable('a+b*t')
    >>> f4(0.5)
    Traceback (most recent call last):
    ...
    NameError: name 'a' is not defined
    >>> f4 = wrap2callable('a+b*t', independent_variable='t', a=1, b=2)
    >>> f4(0.5)
    2.0

    >>> x = linspace(0, 1, 3); y=1+2*x
    >>> f5 = wrap2callable((x,y))
    >>> f5(0.5)
    2.0
    >>> def myfunc(x):  return 1+2*x
    >>> f6 = wrap2callable(myfunc)
    >>> f6(0.5)
    2.0
    >>> f7 = wrap2callable(lambda x: 1+2*x)
    >>> f7(0.5)
    2.0
    >>> class MyClass:
            'Representation of a function f(x; a, b) =a + b*x'
            def __init__(self, a=1, b=1):
                self.a = a;  self.b = b
            def __call__(self, x):
                return self.a + self.b*x
    >>> myclass = MyClass(a=1, b=2)
    >>> f8 = wrap2callable(myclass)
    >>> f8(0.5)
    2.0
    >>> # 3D functions:
    >>> f9 = wrap2callable('1+2*x+3*y+4*z', independent_variables=('x','y','z'))
    >>> f9(0.5,1/3.,0.25)
    4.0
    >>> # discrete 3D data:
    >>> y = linspace(0, 1, 3); z = linspace(-1, 0.5, 16)
    >>> xv = reshape(x, (len(x),1,1))
    >>> yv = reshape(y, (1,len(y),1))
    >>> zv = reshape(z, (1,1,len(z)))
    >>> def myfunc3(x,y,z):  return 1+2*x+3*y+4*z

    >>> values = myfunc3(xv, yv, zv)
    >>> f10 = wrap2callable((x, y, z, values))
    >>> f10(0.5, 1/3., 0.25)
    4.0

    One can also check what the object is wrapped as and do more
    specific operations, e.g.,

    >>> f9.__class__.__name__
    'StringFunction'
    >>> str(f9)     # look at function formula
    '1+2*x+3*y+4*z'
    >>> f8.__class__.__name__
    'MyClass'
    >>> f8.a, f8.b  # access MyClass-specific data
    (1, 2)

    Troubleshooting regarding string functions:
    If you use a string formula with a numpy array, you typically get
    error messages like::

       TypeError: only rank-0 arrays can be converted to Python scalars.

    You must then make the right import (numpy is recommended)::

       from Numeric/numarray/numpy/scitools.numpytools import *

    in the calling code and supply the keyword argument::

       globals=globals()

    to wrap2callable. See also the documentation of class StringFunction
    for more information.
    """
    if isinstance(f, str):
        return StringFunction(f, **kwargs)
        # this is a considerable optimization (up to a factor of 3),
        # but then the additional info in the StringFunction instance
        # is lost in the calling code:
        # return StringFunction(f, **kwargs).__call__
    elif isinstance(f, (float, int, complex)):
        return WrapNo2Callable(f)
    elif isinstance(f, (list,tuple)):
        return WrapDiscreteData2Callable(f)
    elif callable(f):
        return f
    else:
        raise TypeError('f of type %s is not callable' % type(f))


# problem: setitem in ArrayGen does not support multiple indices
# relying on inherited __setitem__ works fine

def NumPy_array_iterator(a, **kwargs):
    """
    Iterate over all elements in a numpy array a.
    Two return values: a generator function and the code of this function.
    The ``numpy.ndenumerate`` iterator performs the same iteration over
    an array, but ``NumPy_array_iterator`` has some additional features
    (especially handy for coding finite difference stencils, see next
    paragraph).

    The keyword arguments specify offsets in the start and stop value
    of the index in each dimension. Legal argument names are
    ``offset0_start``, ``offset0_stop``, ``offset1_start``,
    ``offset1_stop``, etc.  Also ``offset_start`` and ``offset_stop``
    are legal keyword arguments, these imply the same offset value for
    all dimensions.

    Another keyword argument is ``no_value``, which can be True or False.
    If the value is True, the iterator returns the indices as a tuple,
    otherwise (default) the iterator returns a two-tuple consisting of
    the value of the array and the corresponding indices (as a tuple).

    Examples::

    >>> q = linspace(1, 2*3*4, 2*3*4);  q.shape = (2,3,4)
    >>> it, code = NumPy_array_iterator(q)

    >>> print code  # generator function with 3 nested loops:
    def nested_loops(a):
        for i0 in xrange(0, a.shape[0]-0):
            for i1 in xrange(0, a.shape[1]-0):
                for i2 in xrange(0, a.shape[2]-0):
                    yield a[i0, i1, i2], (i0, i1, i2)

    >>> type(it)
    <type 'function'>
    >>> for value, index in it(q):
    ...     print 'a%s = %g' % (index, value)
    ...
    a(0, 0, 0) = 1
    a(0, 0, 1) = 2
    a(0, 0, 2) = 3
    a(0, 0, 3) = 4
    a(0, 1, 0) = 5
    a(0, 1, 1) = 6
    a(0, 1, 2) = 7
    a(0, 1, 3) = 8
    a(0, 2, 0) = 9
    a(0, 2, 1) = 10
    a(0, 2, 2) = 11
    a(0, 2, 3) = 12
    a(1, 0, 0) = 13
    a(1, 0, 1) = 14
    a(1, 0, 2) = 15
    a(1, 0, 3) = 16
    a(1, 1, 0) = 17
    a(1, 1, 1) = 18
    a(1, 1, 2) = 19
    a(1, 1, 3) = 20
    a(1, 2, 0) = 21
    a(1, 2, 1) = 22
    a(1, 2, 2) = 23
    a(1, 2, 3) = 24

    Here is the version where only the indices and no the values
    are returned by the iterator::

    >>> q = linspace(1, 1*3, 3);  q.shape = (1,3)
    >>> it, code = NumPy_array_iterator(q, no_value=True)

    >>> print code
    def nested_loops(a):
        for i0 in xrange(0, a.shape[0]-0):
            for i1 in xrange(0, a.shape[1]-0):
                yield i0, i1

    >>> for i,j in it(q):
    ...   print i,j
    0 0
    0 1
    0 2


    Now let us try some offsets::

    >>> it, code = NumPy_array_iterator(q, offset1_stop=1, offset_start=1)

    >>> print code
    def nested_loops(a):
        for i0 in xrange(1, a.shape[0]-0):
            for i1 in xrange(1, a.shape[1]-1):
                for i2 in xrange(1, a.shape[2]-0):
                    yield a[i0, i1, i2], (i0, i1, i2)

    >>> # note: the offsets appear in the xrange arguments
    >>> for value, index in it(q):
    ...     print 'a%s = %g' % (index, value)
    ...
    a(1, 1, 1) = 18
    a(1, 1, 2) = 19
    a(1, 1, 3) = 20

    """
    # build the code of the generator function in a text string
    # (since the number of nested loops needed to iterate over all
    # elements are parameterized through len(a.shape))
    dims = list(range(len(a.shape)))
    offset_code1 = ['offset%d_start=0' % d for d in dims]
    offset_code2 = ['offset%d_stop=0'  % d for d in dims]
    for d in range(len(a.shape)):
        key1 = 'offset%d_start' % d
        key2 = 'offset%d_stop' % d
        if key1 in kwargs:
            offset_code1.append(key1 + '=' + str(kwargs[key1]))
        if key2 in kwargs:
            offset_code2.append(key2 + '=' + str(kwargs[key2]))

    for key in kwargs:
        if key == 'offset_start':
            offset_code1.extend(['offset%d_start=%d' % (d, kwargs[key]) \
                            for d in range(len(a.shape))])
        if key == 'offset_stop':
            offset_code2.extend(['offset%d_stop=%d' % (d, kwargs[key]) \
                            for d in range(len(a.shape))])

    no_value = kwargs.get('no_value', False)

    for line in offset_code1:
        exec(line)
    for line in offset_code2:
        exec(line)
    code = 'def nested_loops(a):\n'
    indentation = ' '*4
    indent = '' + indentation
    for dim in range(len(a.shape)):
        code += indent + \
        'for i%d in xrange(%d, a.shape[%d]-%d):\n' \
                % (dim, eval('offset%d_start' % dim),
                   dim, eval('offset%d_stop' % dim))
        indent += indentation
    index = ', '.join(['i%d' % d for d in range(len(a.shape))])
    if no_value:
        code += indent + 'yield ' + index
    else:
        code += indent + 'yield ' + 'a[%s]' % index + ', (' + index + ')'
    exec(code)
    return nested_loops, code

def compute_histogram(samples, nbins=50, piecewise_constant=True):
    """
    Given a numpy array samples with random samples, this function
    returns the (x,y) arrays in a plot-ready version of the histogram.
    If piecewise_constant is True, the (x,y) arrays gives a piecewise
    constant curve when plotted, otherwise the (x,y) arrays gives a
    piecewise linear curve where the x coordinates coincide with the
    center points in each bin. The function makes use of
    numpy.lib.function_base.histogram with some additional code
    (for a piecewise curve or displaced x values to the centes of
    the bins).
    """
    import sys
    if 'numpy' in sys.modules:
        y0, bin_edges = histogram(samples, bins=nbins, normed=True)
    h = bin_edges[1] - bin_edges[0]  # bin width
    if piecewise_constant:
        x = zeros(2*len(bin_edges), type(bin_edges[0]))
        y = x.copy()
        x[0] = bin_edges[0]
        y[0] = 0
        for i in range(len(bin_edges)-1):
            x[2*i+1] = bin_edges[i]
            x[2*i+2] = bin_edges[i+1]
            y[2*i+1] = y0[i]
            y[2*i+2] = y0[i]
        x[-1] = bin_edges[-1]
        y[-1] = 0
    else:
        x = zeros(len(bin_edges)-1, type(bin_edges[0]))
        y = y0.copy()
        for i in range(len(x)):
            x[i] = (bin_edges[i] + bin_edges[i+1])/2.0
    return x, y

def _test_factorial(n=80):
    import timeit
    cpu = {}
    for version in ['math', 'scipy', 'plain iterative',
                    'plain recursive', 'lambda list comprehension',
                    'lambda recursive', 'lambda functional',
                    'reduce']:
        t = timeit.Timer('factorial(%s, method="%s")' % (n, version),
                         'from scitools.std import factorial')
        #cpu[version] = t.timeit(100000)
        cpu[version] = t.timeit(10000)
    cpu_min = min(list(cpu.values()))
    cpu = {version: cpu[version]/cpu_min for version in cpu}
    for version in cpu:
        print '%-25s %5.1f' % (version, cpu[version])

def factorial(n, method='math'):
    """
    Compute the factorial n!.
    Different implementations are available (see source code for
    implementation details).

    Note: The math module in Python 2.6 features a factorial
    function, making the present function redundant (except that
    the various pure Python implementations can be of interest
    for comparison).

    Here is an efficiency comparison of the methods (computing 80!
    via python -c 'import scitools.numpyutils as n; n._test_factorial()')


    ==========================   =====================
            Method                Normalized CPU time
    ==========================   =====================
    math                          1.0
    reduce                        1.3
    plain iterative               1.3
    lambda list comprehension     3.5
    scipy                         4.3
    lambda functional             4.5
    lambda recursive              4.6
    plain recursive              13.1
    ==========================   =====================

    """
    if not isinstance(n, (int, float)):
        raise TypeError('factorial(n): n must be integer not %s' % type(n))

    if n == 0 or n == 1:
        return 1

    if method == 'math':
        import math
        return math.factorial(n)
    elif method == 'scipy':
        try:
            import scipy.misc
            return scipy.misc.factorial(n)
        except ImportError:
            print 'numpyutils.factorial: scipy is not available'
            print 'default method="reduce" is used instead'
            return reduce(operator.mul, range(2, n+1))
            # or return factorial(n)
    elif method == 'plain iterative':
        f = 1
        for i in range(1, n+1):
            f *= i
        return f
    elif method == 'plain recursive':
        if n == 1:
            return 1
        else:
            return n*factorial(n-1, method)
    elif method == 'lambda recursive':
        fc = lambda n: n and fc(n-1)*n or 1
        return fc(n)
    elif method == 'lambda functional':
        fc = lambda n: n<=0 or \
             reduce(lambda a,b: a*b, range(1,n+1))
        return fc(n)
    elif method == 'lambda list comprehension':
        fc = lambda n: [j for j in [1] for i in range(2,n+1) \
                        for j in [j*i]] [-1]
        return fc(n)
    elif method == 'reduce':
        return reduce(operator.mul, range(2, n+1))
    else:
        raise ValueError('factorial: method="%s" is not supported' % method)


def asarray_cpwarn(a, dtype=None, message='warning', comment=''):
    """
    As asarray, but a warning or exception is issued if the
    a array is copied.
    """
    a_new = asarray(a, dtype)
    # must drop numpy's order argument since it conflicts
    # with Numeric's savespace

    # did we copy?
    if a_new is not a:
        # we do not return the identical array, i.e., copy has taken place
        msg = '%s  copy of array %s, from %s to %s' % \
              (comment, a.shape, type(a), type(a_new))
        if message == 'warning':
            print 'Warning: %s' % msg
        elif message == 'exception':
            raise TypeError(msg)
    return a_new


def seq(min=0.0, max=None, inc=1.0, type=float,
        return_type='NumPyArray'):
    """
    Generate numbers from min to (and including!) max,
    with increment of inc. Safe alternative to arange.
    The return_type string governs the type of the returned
    sequence of numbers ('NumPyArray', 'list', or 'tuple').
    """
    if max is None: # allow sequence(3) to be 0., 1., 2., 3.
        # take 1st arg as max, min as 0, and inc=1
        max = min; min = 0.0; inc = 1.0
    r = arange(min, max + inc/2.0, inc, type)
    if return_type == 'NumPyArray' or return_type == ndarray:
        return r
    elif return_type == 'list':
        return r.tolist()
    elif return_type == 'tuple':
        return tuple(r.tolist())


def iseq(start=0, stop=None, inc=1):
    """
    Generate integers from start to (and including) stop,
    with increment of inc. Alternative to range/xrange.
    """
    if stop is None: # allow isequence(3) to be 0, 1, 2, 3
        # take 1st arg as stop, start as 0, and inc=1
        stop = start; start = 0; inc = 1
    return xrange(start, stop+inc, inc)

sequence = seq  # backward compatibility
isequence = iseq  # backward compatibility


def arr(shape=None, element_type=float,
        interval=None,
        data=None, copy=True,
        file_=None,
        order='C'):
    """
    Compact and flexible interface for creating numpy arrays,
    including several consistency and error checks.

     - *shape*: length of each dimension, tuple or int
     - *data*: list, tuple, or numpy array with data elements
     - *copy*: copy data if true, share data if false, boolean
     - *element_type*: float, int, int16, float64, bool, etc.
     - *interval*: make elements from a to b (shape gives no of elms), tuple or list
     - *file_*: filename or file object containing array data, string
     - *order*: 'Fortran' or 'C' storage, string
     - return value: created Numerical Python array

    The array can be created in four ways:

      1. as zeros (just shape specified),

      2. as uniformly spaced coordinates in an interval [a,b]

      3. as a copy of or reference to (depending on copy=True,False resp.)
         a list, tuple, or numpy array (provided as the data argument),

      4. from data in a file (for one- or two-dimensional real-valued arrays).

    The function calls the underlying numpy functions zeros, array and
    linspace (see the numpy manual for the functionality of these
    functions).  In case of data in a file, the first line determines
    the number of columns in the array. The file format is just rows
    and columns with numbers, no decorations (square brackets, commas,
    etc.) are allowed.

    >>> arr((3,4))
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.]])

    >>> arr(4, element_type=int) + 4  # integer array
    array([4, 4, 4, 4])

    >>> arr(3, interval=[0,2])
    array([ 0.,  1.,  2.])

    >>> somelist=[[0,1],[5,5]]
    >>> a = arr(data=somelist)
    >>> a  # a has always float elements by default
    array([[ 0.,  1.],
           [ 5.,  5.]])
    >>> a = arr(data=somelist, element_type=int)
    >>> a
    array([[0, 1],
           [5, 5]])
    >>> b = a + 1

    >>> c = arr(data=b, copy=False)  # let c share data with b
    >>> b is c
    True
    >>> id(b) == id(c)
    True

    >>> # make a file with array data:
    >>> f = open('tmp.dat', 'w')
    >>> f.write('''\
    ... 1 3
    ... 2 6
    ... 3 12
    ... 3.5 20
    ... ''')
    >>> f.close()
    >>> # read array data from file:
    >>> a = arr(file_='tmp.dat')
    >>> a
    array([[  1. ,   3. ],
           [  2. ,   6. ],
           [  3. ,  12. ],
           [  3.5,  20. ]])
    """
    if data is None and file_ is None and shape is None:
        return None

    if data is not None:

        if not isinstance(data, collections.Sequence):
            raise TypeError('arr: data argument is not a sequence type')

        if isinstance(shape, (list,tuple)):
            # check that shape and data are compatible:
            if reduce(operator.mul, shape) != size(data):
                raise ValueError(
                    'arr: shape=%s is not compatible with %d '\
                    'elements in the provided data' % (shape, size(data)))
        elif isinstance(shape, int):
            if shape != size(data):
                raise ValueError(
                    'arr: shape=%d is not compatible with %d '\
                    'elements in the provided data' % (shape, size(data)))
        elif shape is None:
            if isinstance(data, (list,tuple)) and copy == False:
                # cannot share data (data is list/tuple)
                copy = True
            return array(data, dtype=element_type, copy=copy, order=order)
        else:
            raise TypeError(
                'shape is %s, must be list/tuple or int' % type(shape))
    elif file_ is not None:
        if isinstance(file_, basestring):
            file_ = open(file_, 'r')
        # skip blank lines:
        while True:
            line1 = file_.readline().strip()
            if line1 != '':
                break
        ncolumns = len(line1.split())
        file_.seek(0)
        # we assume that array data in file has element_type=float:
        if not (element_type == float or element_type == 'd'):
            raise ValueError('element_type must be float_/"%s", not "%s"' % \
                             ('d', element_type))

        d = array([float(word) for word in file_.read().split()])
        if isinstance(file_, basestring):
            f.close()
        # shape array d:
        if ncolumns > 1:
            suggested_shape = (int(len(d)/ncolumns), ncolumns)
            total_size = suggested_shape[0]*suggested_shape[1]
            if total_size != len(d):
                raise ValueError(
                    'found %d array entries in file "%s", but first line\n'\
                    'contains %d elements - no shape is compatible with\n'\
                    'these values' % (len(d), file, ncolumns))
            d.shape = suggested_shape
        if shape is not None:
            if shape != d.shape:
                raise ValueError(
                    'shape=%s is not compatible with shape %s found in "%s"' % \
                    (shape, d.shape, file))
        return d

    elif interval is not None and shape is not None:
        if not isinstance(shape, int):
            raise TypeError('For array values in an interval, '\
                            'shape must be an integer')
        if not isinstance(interval, (list,tuple)):
            raise TypeError('interval must be list or tuple, not %s' % \
                            type(interval))
        if len(interval) != 2:
            raise ValueError('interval must be a 2-tuple (or list)')

        try:
            return linspace(interval[0], interval[1], shape)
        except MemoryError as e:
            # print more information (size of data):
            print e, 'of size %s' % shape

    else:
        # no data, no file, just make zeros

        if not isinstance(shape, (tuple, int, list)):
            raise TypeError('arr: shape (1st arg) must be tuple or int')
        if shape is None:
            raise ValueError(
                'arr: either shape, data, or from_function must be specified')

        try:
            return zeros(shape, dtype=element_type, order=order)
        except MemoryError as e:
            # print more information (size of data):
            print e, 'of size %s' % shape

def _test():
    _test_FloatComparison()
    # test norm functions for multi-dimensional arrays:
    a = array(list(range(27)))
    a.shape = (3,3,3)
    functions = [norm_l2, norm_L2, norm_l1, norm_L1, norm_inf]
    results = [78.7464284904401239, 15.1547572288924073, 351, 13, 26]
    for f, r in zip(functions, results):
        if not float_eq(f(a), r):
            print '%s failed: result=%g, not %g' % (f.__name__, f(a), r)

    # Gram-Schmidt:
    A = array([[1,2,3], [3,4,5], [6,4,1]], float)
    V1 = Gram_Schmidt(A, normalize=True)
    V2 = Gram_Schmidt1(A)
    if not float_eq(V1, V2):
        print 'The two Gram_Schmidt versions did not give equal results'
        print 'Gram_Schmidt:\n', V1
        print 'Gram_Schmidt1:\n', V2

    # Null space:
    K = array([[1,2,3], [1,2,3], [0,0,0], [-1, -2, -3]], float)
    #K = random.random(3*7).reshape(7,3) # does not work...
    print 'K=\n', K
    print 'null(K)=\n', null(K)
    r = K*null(K)
    print 'K*null(K):', r


if __name__ == '__main__':
    from numpy import *
    _test()
