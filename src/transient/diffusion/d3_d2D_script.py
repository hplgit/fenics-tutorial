#!/usr/bin/env

import subprocess, numpy, re

def iterative_solver_data(output):
    """
    Extract information on iterative solver performance from
    output string from a FEniCS program.
    """
    time_points = []
    num_iter = []
    for line in output.splitlines():

        if line.find('Solving linear system of size') == 0:
            size = int(line.split()[5])

        # Convergence output differ from backend to backend,
        # so a regex is needed to capture the no of iterations
        pattern = 'converged in (\d+) iterations'
        m = re.search(pattern, line)
        if m:
            num_iter.append(int(m.group(1)))

        if 'time = ' in line:
            time_points.append(float(line.split()[2]))
        if 'nonzero initial guess' in line:
            nzig = True if line.split()[-1] == 'True' else False
    return numpy.array(time_points), numpy.array(num_iter), size, nzig


def system(cmd):
    if os.name == 'posix':
        outfile = '.tmp.tmp1'
        failure = os.system(cmd + ' | tee ' + outfile)
        f = open(outfile, 'r'); output = f.read(); f.close()
        os.remove(outfile)
    else:
        # Windows: grab output and don't display anything in the terminal window
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        output, errors = p.communicate()
        return iterative_solver_data(output)

cmd = 'python d3_d2D.py zero'
t1, iter0, size, nzig1 = system(cmd)

cmd = 'python d3_d2D.py last'
t2, iterl, size, nzig2 = system(cmd)

cmd = 'python d3_d2D.py random'
t3, iterr, size, nzig3 = system(cmd)

cmd = 'python d3_d2D.py default'
t4, iterd, size, nzig4 = system(cmd)

cmd = 'python d3_d2D.py extrapolate'
t5, itere, size, nzig5 = system(cmd)

print 'zero as guess:          ', iter0, nzig1
print 'last solution as guess: ', iterl, nzig2
print 'random as guess:        ', iterr, nzig3
print 'default (zero) as guess:', iterd, nzig4
print 'extrapolated guess:     ', itere, nzig5

from scitools.std import plot
