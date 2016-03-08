from membrane2_class import Problem, Viz

for sigma in 0.1, 1, 10:
    for A in 1, 100, 1000:
        print '\n\n ****************** sigma:', sigma
        dir = 'case-%g' % sigma
        os.mkdir(dir)
        os.chdir(dir)
        problem = Problem(sigma=sigma)
        problem.solve()
        visualizer = Viz(problem)
        visualizer.batch_viz()
        os.chdir(os.pardir)

