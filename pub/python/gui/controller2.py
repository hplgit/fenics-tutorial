import os
from compute import compute as compute_function

# Pool object (must be imported before model)
from compute import define_pool as pool_function
pool = pool_function()

# Can define other default values in a file: --poolfile name
#from parampool.pool.UI import set_defaults_from_file
#pool = set_defaults_from_file(pool)
# Can override default values on the command line
from parampool.pool.UI import set_values_from_command_line
pool = set_values_from_command_line(pool)

from parampool.pool.UI import write_poolfile
write_poolfile(pool, '.tmp_pool.dat')

result = compute_function(pool)
