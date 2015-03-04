"""
Temperature variations in the ground. 
Class version of diffusion123D_sin.py.
NOTE: go on with class Problem, but use the rest of the code
from the newer sin_daD.py.
"""

class Problem:
    def __init__(self):
        """Set default values of problem parameters."""
        self.d = 2
        self.degree = 1
        self.physics = dict(T_R = 0, T_A=0, omega=7.27E-5)
        self.numerics = dict(num_steps_per_period=14, num_periods=5)

    def update(self):
        """Update problem parameters that depend on others."""
        omega = self.physics['omega']
        period = 2*pi/omega
        self.numerics['dt'] = period/self.numerics['num_steps_per_period']
        self.numerics['tstop'] = period*self.numerics['num_periods']

    def T_0(self):
        return Expression(
            'T_R + T_A*sin(omega*t)',
            'T_R': self.physics['T_R'],
            'T_A': self.physics['T_A'],
            'omega': self.physics['omega'])
    
    def exact_solution(self, x, t):
        pass

[[[[ not ready
