import os
from compute import compute as compute_function

# Pool object (must be imported before model)
from compute import define_pool as pool_function
pool = pool_function()

# Can define other default values in a file: --poolfile name
from parampool.pool.UI import set_defaults_from_file
pool = set_defaults_from_file(pool)
# Can override default values on the command line
from parampool.pool.UI import set_values_from_command_line
pool = set_values_from_command_line(pool)

from flask import Flask, render_template, request
from model import Compute

# Application object
app = Flask(__name__)

# Path to the web application
@app.route('/', methods=['GET', 'POST'])
def index():
    form = Compute(request.form)
    if request.method == 'POST': # and form.validate():

        # Send data to Pool object
        for field in form:
            if field.name not in request.files:
                name = field.description
                value = field.data
                data_item = pool.set_value(name, value)

        result = compute(pool)

    else:
        result = None

    return render_template("view.html", form=form, result=result)


def compute(pool):
    """
    Generic function for calling compute_function with values
    taken from the pool object.
    Return the output from the compute_function.
    """

    # compute_function must have only one positional argument
    # named pool
    import inspect
    arg_names = inspect.getargspec(compute_function).args
    if len(arg_names) == 1 and arg_names[0] == "pool":
        result = compute_function(pool)
    else:
        raise TypeError('%s(%s) can only have one argument named "pool"'
                        % (compute_function.__name__, ', '.join(arg_names)))
    return result

if __name__ == '__main__':
    app.run(debug=True)

    # Dump pool to file for use with --poolfile .tmp_pool.dat
    from parampool.pool.UI import write_poolfile
    write_poolfile(pool, '.tmp_pool.dat')
