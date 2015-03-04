from prepro2D import load_triangle_mesh, plot_mesh
coordinates, connectivity, materials = load_triangle_mesh('tmp', '1')
plot_mesh(coordinates, connectivity, materials)
