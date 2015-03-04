from dolfin import *
import time, sys
import numpy as np

def box(x, y, z, nx=None, ny=None, nz=None):
    """
    Return a mesh over a box-shaped domain.

    ============  =================================================
    Name          Description
    ============  =================================================
    x, y, z       interval [,] or coordinate arrays
    nx, ny, nz    integers reflecting the division of intervals
    ============  =================================================

    In the x, y, and z directions one can either specify an interval
    to be uniformly partitioned, with a given number of
    divisions (nx, ny, or nz), or one can specify a coordinate array
    for non-uniformly partitioned structured meshes.

    Examples::

        # Unit cube
        mesh = box(x=[0, 1], y=[0, 1], z=[0, 1], nx=10, ny=12, nz=5)
        # Specified graded mesh in x direction
        x = [0, 0.1, 0.2, 0.5, 1, 2]  # implies nx=len(x)
        mesh = box(x=x, y=[-1, 1], z=[-1, 1], ny=12, nz=5)
    """
    for arg in x, y, z:
        if not isinstance(arg, (list,tuple,np.ndarray)):
            raise TypeError('box: x, y, z must be list, tuple or numpy '\
                            'array, not %s' % type(arg))
    if len(x) == 2:
        if nx is None:
            raise ValueError('box: interval in x %s, no nx set' % x)
        x = np.linspace(x[0], x[1], nx+1)
    else:
        nx = len(x)-1
    if len(y) == 2:
        if nx is None:
            raise ValueError('box: interval in y %s, no ny set' % y)
        y = np.linspace(y[0], y[1], ny+1)
    else:
        ny = len(y)-1
    if len(z) == 2:
        if nz is None:
            raise ValueError('box: interval in z %s, no nz set' % z)
        z = np.linspace(z[0], z[1], nz+1)
    else:
        nz = len(z)-1

    editor = MeshEditor()
    mesh = Mesh()
    tdim = gdim = 3
    editor.open(mesh, 'tetrahedron', tdim, gdim)

    editor.init_vertices((nx+1)*(ny+1)*(nz+1))
    vertex = 0
    for iz in xrange(nz+1):
        for iy in xrange(ny+1):
            for ix in xrange(nx+1):
                editor.add_vertex(vertex, x[ix], y[iy], z[iz])
                vertex += 1

    editor.init_cells(6*nx*ny*nz)
    cell = 0
    for iz in xrange(nz):
        for iy in xrange(ny):
            for ix in xrange(nx):
                v0 = iz*(nx+1)*(ny+1) + iy*(nx+1) + ix
                v1 = v0 + 1;
                v2 = v0 + (nx+1)
                v3 = v1 + (nx+1)
                v4 = v0 + (nx+1)*(ny+1)
                v5 = v1 + (nx+1)*(ny+1)
                v6 = v2 + (nx+1)*(ny+1)
                v7 = v3 + (nx+1)*(ny+1)

                editor.add_cell(cell, v0, v1, v3, v7);  cell += 1
                editor.add_cell(cell, v0, v1, v7, v5);  cell += 1
                editor.add_cell(cell, v0, v5, v7, v4);  cell += 1
                editor.add_cell(cell, v0, v3, v2, v7);  cell += 1
                editor.add_cell(cell, v0, v6, v4, v7);  cell += 1
                editor.add_cell(cell, v0, v2, v6, v7);  cell += 1

    editor.close()
    mesh.structured_mesh = (x, y, z)
    return mesh


def rectangle(x, y, nx=None, ny=None, diagonal='right'):
    """
    Return a mesh over a domain with the shape of a rectangle.

    ============  =================================================
    Name          Description
    ============  =================================================
    x, y          interval [,] or coordinate arrays
    nx, ny        integers reflecting the division of intervals
    diagonal      string specifying the direction of diagonals
                  ('left', 'right', 'left/right', 'right/left',
                  'crossed')
    ============  =================================================

    In the x and y directions one can either specify an interval
    to be uniformly partitioned, with a given number of
    divisions (nx or ny), or one can specify a coordinate array
    for non-uniformly partitioned structured meshes.

    Examples::

        # Unit square
        mesh = box(x=[0, 1], y=[0, 1], nx=10, ny=12)
        # Specified graded mesh in y direction
        y = [0, 0.1, 0.2, 0.5, 1, 2]  # implies nx=len(x)
        mesh = box(x=[0, 3], y=y, nx=12)
    """

    for arg in x, y:
        if not isinstance(arg, (list,tuple,np.ndarray)):
            raise TypeError('box: x, y, z must be list, tuple or numpy '\
                            'array, not %s' % type(arg))
    if len(x) == 2:
        if nx is None:
            raise ValueError('box: interval in x %s, no nx set' % x)
        x = np.linspace(x[0], x[1], nx+1)
    else:
        nx = len(x)-1
    if len(y) == 2:
        if nx is None:
            raise ValueError('box: interval in y %s, no ny set' % y)
        y = np.linspace(y[0], y[1], ny+1)
    else:
        ny = len(y)-1

    valid_diagonals = 'left', 'right', 'left/right', 'right/left', 'crossed'
    if not diagonal in valid_diagonals:
        raise ValueError('rectangle: wrong value of diagonal="%s", not in %s' \
                         % (diagonal, ', '.join(valid_diagonals)))

    editor = MeshEditor()
    mesh = Mesh()
    tdim = gdim = 2
    editor.open(mesh, 'triangle', tdim, gdim)

    if diagonal == 'crossed':
        editor.init_vertices((nx+1)*(ny+1) + nx*ny)
        editor.init_cells(4*nx*ny)
    else:
        editor.init_vertices((nx+1)*(ny+1))
        editor.init_cells(2*nx*ny)

    vertex = 0
    for iy in xrange(ny+1):
        for ix in xrange(nx+1):
            editor.add_vertex(vertex, x[ix], y[iy])
            vertex += 1
    if diagonal == 'crossed':
        for iy in xrange(ny):
            for ix in xrange(nx):
                x_mid = 0.5*(x[ix+1] + x[ix])
                y_mid = 0.5*(y[iy+1] + y[iy])
                editor.add_vertex(vertex, x_mid, y_mid)
                vertex += 1

    cell = 0
    if diagonal == 'crossed':
        for iy in xrange(ny):
            for ix in xrange(nx):
                v0 = iy*(nx+1) + ix
                v1 = v0 + 1
                v2 = v0 + (nx+1)
                v3 = v1 + (nx+1)
                vmid = (nx+1)*(ny+1) + iy*nx + ix

                # Note that v0 < v1 < v2 < v3 < vmid.
                editor.add_cell(cell, v0, v1, vmid);  cell += 1
                editor.add_cell(cell, v0, v2, vmid);  cell += 1
                editor.add_cell(cell, v1, v3, vmid);  cell += 1
                editor.add_cell(cell, v2, v3, vmid);  cell += 1

    else:
        local_diagonal = diagonal
        # Set up alternating diagonal
        for iy in xrange(ny):
            if diagonal == "right/left":
                if iy % 2 == 0:
                    local_diagonal = "right"
                else:
                    local_diagonal = "left"

            if diagonal == "left/right":
                if iy % 2 == 0:
                    local_diagonal = "left"
                else:
                    local_diagonal = "right"
            for ix in xrange(nx):
                v0 = iy*(nx + 1) + ix
                v1 = v0 + 1
                v2 = v0 + nx+1
                v3 = v1 + nx+1

                if local_diagonal == "left":
                    editor.add_cell(cell, v0, v1, v2);  cell += 1
                    editor.add_cell(cell, v1, v2, v3);  cell += 1
                    if diagonal == "right/left" or diagonal == "left/right":
                        local_diagonal = "right"
                else:
                    editor.add_cell(cell, v0, v1, v3);  cell += 1
                    editor.add_cell(cell, v0, v2, v3);  cell += 1
                    if diagonal == "right/left" or diagonal == "left/right":
                        local_diagonal = "left"

    editor.close()
    mesh.structured_mesh = (x, y)
    return mesh

def interval(x, nx=None):
    if len(x) > 2:
        nx = len(x)-1
    elif len(x) == 2:
        if x is None:
            raise ValueError('interval: nx must be specified')

    mesh = Interval(nx, x[0], x[-1])

    if len(x) > 2:
        mesh.coordinates()[:] = np.asarray(x).reshape(nx+1, 1)

    mesh.structured_mesh = (x,)
    return mesh


def remove_cells(mesh, cell_markers=[]):
    num_cells = mesh.num_cells()
    cell_markers = np.asarray(cell_markers)
    gt = cell_markers > num_cells-1
    if gt.any():
        raise ValueError('remove_cells: cell_markers object has illegal values > %d (%s)' % (num_cells-1, cell_markers[gt]))

    editor = MeshEditor()
    new_mesh = Mesh()
    tdim = mesh.topology().dim()
    gdim = mesh.geometry().dim()
    elem_tp = ['interval', 'triangle', 'tetrahedron']
    editor.open(new_mesh, elem_tp[tdim-1], tdim, gdim)

    # Find the cell numbers that will survive the removal
    survivors = range(num_cells)
    for m in cell_markers:
        survivors.remove(m)

    # Find a mapping from old to new vertex numbers
    new_mesh_cells = mesh.cells()[survivors]
    import sets
    indices = sets.Set(new_mesh_cells.flatten()) # remove duplicate vertex numbers
    orig_vertices = list(indices)
    old2new = dict(zip(orig_vertices, range(len(orig_vertices))))

    coor = mesh.coordinates()  # for efficiency
    cell = mesh.cells()
    editor.init_vertices(len(old2new))
    for i in old2new:
        editor.add_vertex(old2new[i], Point(*coor[i]))
    editor.init_cells(len(survivors))
    for c, c_old in enumerate(survivors):
        #new_vertex_numbers = np.array([old2new[i] for i in cell[c_old]])
        new_vertex_numbers = [old2new[i] for i in cell[c_old]]
        editor.add_cell(c, *new_vertex_numbers)
    editor.close()
    return new_mesh

def verify():
    mesh = box(x=[0, 0.5, 1], y=[-1, 1], z=[0, 1], ny=1, nz=2)
    v_coor_box = mesh.coordinates().copy()
    # if not .copy() it seems that other meshes are using the
    # same data, because v_coor_box just contains garbage below...
    r_coor_box = np.array([[ 0. , -1. ,  0. ],
                           [ 0.5, -1. ,  0. ],
                           [ 1. , -1. ,  0. ],
                           [ 0. ,  1. ,  0. ],
                           [ 0.5,  1. ,  0. ],
                           [ 1. ,  1. ,  0. ],
                           [ 0. , -1. ,  0.5],
                           [ 0.5, -1. ,  0.5],
                           [ 1. , -1. ,  0.5],
                           [ 0. ,  1. ,  0.5],
                           [ 0.5,  1. ,  0.5],
                           [ 1. ,  1. ,  0.5],
                           [ 0. , -1. ,  1. ],
                           [ 0.5, -1. ,  1. ],
                           [ 1. , -1. ,  1. ],
                           [ 0. ,  1. ,  1. ],
                           [ 0.5,  1. ,  1. ],
                           [ 1. ,  1. ,  1. ]])

    v_cell_box = mesh.cells().copy()
    r_cell_box = np.array([[ 0,  1,  4, 10],
                           [ 0,  1,  7, 10],
                           [ 0,  6,  7, 10],
                           [ 0,  3,  4, 10],
                           [ 0,  6,  9, 10],
                           [ 0,  3,  9, 10],
                           [ 1,  2,  5, 11],
                           [ 1,  2,  8, 11],
                           [ 1,  7,  8, 11],
                           [ 1,  4,  5, 11],
                           [ 1,  7, 10, 11],
                           [ 1,  4, 10, 11],
                           [ 6,  7, 10, 16],
                           [ 6,  7, 13, 16],
                           [ 6, 12, 13, 16],
                           [ 6,  9, 10, 16],
                           [ 6, 12, 15, 16],
                           [ 6,  9, 15, 16],
                           [ 7,  8, 11, 17],
                           [ 7,  8, 14, 17],
                           [ 7, 13, 14, 17],
                           [ 7, 10, 11, 17],
                           [ 7, 13, 16, 17],
                           [ 7, 10, 16, 17]], dtype=np.uint32)


    mesh = rectangle(y=[0, 0.5, 1], x=[-1, 1], nx=2)
    v_coor_rect = mesh.coordinates().copy()
    r_coor_rect = np.array([[-1. ,  0. ],
                            [ 0. ,  0. ],
                            [ 1. ,  0. ],
                            [-1. ,  0.5],
                            [ 0. ,  0.5],
                            [ 1. ,  0.5],
                            [-1. ,  1. ],
                            [ 0. ,  1. ],
                            [ 1. ,  1. ]])
    v_cell_rect = mesh.cells().copy()
    r_cell_rect = np.array([[0, 1, 4],
                            [0, 3, 4],
                            [1, 2, 5],
                            [1, 4, 5],
                            [3, 4, 7],
                            [3, 6, 7],
                            [4, 5, 8],
                            [4, 7, 8]], dtype=np.uint32)
    mesh = remove_cells(mesh, cell_markers=[0, 1])
    v_coor_rect_remv = mesh.coordinates().copy()
    r_coor_rect_remv = np.array([[ 0. ,  0. ],
                                 [ 1. ,  0. ],
                                 [-1. ,  0.5],
                                 [ 0. ,  0.5],
                                 [ 1. ,  0.5],
                                 [-1. ,  1. ],
                                 [ 0. ,  1. ],
                                 [ 1. ,  1. ]])
    v_cell_rect_remv = mesh.cells().copy()
    r_cell_rect_remv = np.array([[0, 1, 4],
                                 [0, 3, 4],
                                 [2, 3, 6],
                                 [2, 5, 6],
                                 [3, 4, 7],
                                 [3, 6, 7]], dtype=np.uint32)

    mesh = interval(x=[0, 0.1, 0.2, 1])
    v_coor_intv = mesh.coordinates().copy()
    r_coor_intv = np.array([[ 0. ],
                            [ 0.1],
                            [ 0.2],
                            [ 1. ]])

    v_cell_intv = mesh.cells().copy()
    r_cell_intv = np.array([[0, 1],
                            [1, 2],
                            [2, 3]], dtype=np.uint32)

    assert np.allclose(v_coor_box, r_coor_box)
    assert (v_cell_box == r_cell_box).all()
    assert np.allclose(v_coor_rect, r_coor_rect)
    assert (v_cell_rect == r_cell_rect).all()
    assert np.allclose(v_coor_rect_remv, r_coor_rect_remv)
    assert (v_cell_rect_remv == r_cell_rect_remv).all()
    assert np.allclose(v_coor_intv, r_coor_intv)
    assert (v_cell_intv == r_cell_intv).all()

def plot_mesh(coordinates, connectivity, materials=None,
              plotfile='tmp.png', numbering=False):
    if materials is None:
        # Same material number (1) for all cells
        materials = np.ones(connectivity.shape[0])

    vertices = []
    for e in xrange(connectivity.shape[0]):
        local_vertex_numbers = connectivity[e,:]
        local_coordinates = coordinates[local_vertex_numbers,:]
        vertices.append(local_coordinates)
    # Vectorized version
    print 'vertices, manual loop:'
    print vertices
    vertices = coordinates[connectivity]
    print 'vertices, vectorized:'
    print vertices
    import matplotlib.cm as cm
    import matplotlib.collections as collections
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap as lc
    col = collections.PolyCollection(vertices, facecolors='yellow',)
    #materials[materials == 0] = 1
    #materials = np.asarray(materials, float)
    #materials = np.random.uniform(0, 1, materials.size)
    #print 'size of materials:', materials.size
    col.set_array(materials)
    graywhite = lc(['white', 'gray'])
    col.set_cmap(cm.hot)
    col.set_cmap(cm.jet)
    col.set_cmap(graywhite)
    fig = plt.figure()
    ax = fig.gca()
    ax.add_collection(col)
    if numbering:
        # Vertex numbering
        for i in range(coordinates.shape[0]):
            ax.text(coordinates[i,0], coordinates[i,1], str(i),
                    bbox=dict(facecolor='red', alpha=0.5))
        # Element numbering
        for i in range(connectivity.shape[0]):
            v0, v1, v2 = connectivity[i,:]
            centroid = (coordinates[v0,:] + coordinates[v1,:] + \
                        coordinates[v2,:])/3.0
            ax.text(centroid[0], centroid[1], i)

    xmin, xmax = coordinates[:,0].min(), coordinates[:,0].max()
    ymin, ymax = coordinates[:,1].min(), coordinates[:,1].max()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    plt.savefig(plotfile)
    plt.show()

n = int(sys.argv[1])
t0 = time.clock()
mesh = box(x=[0, 1], y=[0, 1], z=[0, 1], nx=n, ny=n, nz=n)
# 1M vertices in 8 sec on MacBook Air 11''
t1 = time.clock()
print mesh
print 'generated in %s seconds' % (t1-t0)
#file = File('mesh.xml')
#file << mesh
if mesh.num_cells() <= 6*4**3:
    plot(mesh)

t0 = time.clock()
n = 4
mesh2 = rectangle(x=[0, 0.1, 0.2, 0.5, 1], y=[0, 1], ny=n,
                  diagonal='left/right')
# 1M vertices in 8 sec on MacBook Air 11''
t1 = time.clock()
print mesh2
print 'generated in %s seconds' % (t1-t0)
#file = File('mesh2.xml')
#file << mesh2

cell_markers = [1,2,3,4,20,28,29,30,31]
mesh21 = remove_cells(mesh2, cell_markers)
print 'mesh21:', mesh21

mesh3 = interval(x=[0, 0.1, 0.2, 1])

print '----------------------'
verify()

if mesh2.num_cells() <= 6*4**3:
    plot(mesh2)
    plot(mesh21)
    plot_mesh(mesh2.coordinates(), mesh2.cells(), materials=None,
              plotfile='tmp.png', numbering=True)
    interactive()


