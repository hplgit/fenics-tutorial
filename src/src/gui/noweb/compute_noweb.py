def define_pool():
    from parampool.utils import fenicsxml2pool
    from parampool.pool.Pool import Pool
    pool = Pool()
    pool.subpool('Main menu')
    pool.add_data_item(name='element degree', default=1,
                       str2type=int)
    pool.add_data_item(name='Nx', default=10,
                       str2type=int)
    pool.add_data_item(name='Ny', default=10,
                       str2type=int)
    pool.add_data_item(name='f', default='-6.0',
                       str2type=str)
    pool.add_data_item(name='u0',
                       default='1 + x[0]*x[0] + 2*x[1]*x[1]',
                       str2type=str)
    # Subpool with built-in FEniCS parameters
    pool = fenicsxml2pool('prm.xml', pool)
    pool.update()
    return pool

def compute(pool):
    # Load pool into DOLFIN's parameters data structure
    from parampool.utils import set_dolfin_prm
    import dolfin
    pool.traverse(set_dolfin_prm, user_data=dolfin.parameters)
    # Load user's parameters
    Nx = pool.get_value('Nx')
    Ny = pool.get_value('Ny')
    degree = pool.get_value('element degree')
    f_str = pool.get_value('f')
    u0_str = pool.get_value('u0')
    f = dolfin.Expression(f_str)
    u0 = dolfin.Expression(u0_str)

    from poisson_solver import solver
    u = solver(f, u0, Nx, Ny, degree)
    #dolfin.plot(u, title='Solution', interactive=True)

    from poisson_iterative import gradient
    grad_u = gradient(u)
    grad_u_x, grad_u_y = grad_u.split(deepcopy=True)

    # Make VTK file, offer for download
    vtkfile = dolfin.File('poisson2D.pvd')
    vtkfile << u
    vtkfile << grad_u

    # Make Matplotlib plots and inline them in the HTML code
    from poisson_bcs import structured_mesh
    u_box = structured_mesh(u, (Nx, Ny))
    u_ = u_box.values
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    cv = u_box.grid.coorv  # vectorized mesh coordinates
    ax.plot_surface(cv[0], cv[1], u_, cmap=cm.coolwarm,
                    rstride=1, cstride=1)
    plt.title('Surface plot of solution')
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = figfile.getvalue()  # get the bytes in PNG file
    import base64
    figdata_png = base64.b64encode(figdata_png)

    # Return HTML code for presenting the results
    html = """
<p>Maximum value of u: %g</p>
<p><a href="posson2D.pvd">VTK file with data</a></p>
<p><img src="data:image/png;base64,%s" width="600"></p>
""" % (u.vector().array().max(), figdata_png)
    return html

def compute_noweb(pool):
    # Load pool into DOLFIN's parameters data structure
    from parampool.utils import set_dolfin_prm
    import dolfin
    pool.traverse(set_dolfin_prm, user_data=dolfin.parameters)
    # Load user's parameters
    Nx = pool.get_value('Nx')
    Ny = pool.get_value('Ny')
    degree = pool.get_value('element degree')
    f_str = pool.get_value('f')
    u0_str = pool.get_value('u0')
    f = dolfin.Expression(f_str)
    u0 = dolfin.Expression(u0_str)

    from poisson_solver import solver
    u = solver(f, u0, Nx, Ny, degree)
    #dolfin.plot(u, title='Solution', interactive=True)

    from poisson_iterative import gradient
    grad_u = gradient(u)
    grad_u_x, grad_u_y = grad_u.split(deepcopy=True)

    # Make VTK file, offer for download
    vtkfile = dolfin.File('poisson2D.pvd')
    vtkfile << u
    vtkfile << grad_u

    dolfin.plot(u)
    dolfin.plot(u.function_space().mesh())
    dolfin.plot(grad_u)
    dolfin.interactive()
