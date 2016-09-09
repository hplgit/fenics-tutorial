import wtforms as wtf
from parampool.html5.flask.fields import HTML5FloatField, FloatRangeField, IntegerRangeField
import flask.ext.wtf.html5 as html5

class Compute(wtf.Form):

    Main_menu__element_degree      = wtf.TextField(
        label=u'element degree',
        description=u'/Main menu/element degree',
        default='1',
        validators=[wtf.validators.InputRequired()])

    Main_menu__Nx                  = wtf.TextField(
        label=u'Nx',
        description=u'/Main menu/Nx',
        default='10',
        validators=[wtf.validators.InputRequired()])

    Main_menu__Ny                  = wtf.TextField(
        label=u'Ny',
        description=u'/Main menu/Ny',
        default='10',
        validators=[wtf.validators.InputRequired()])

    Main_menu__f                   = wtf.TextField(
        label=u'f',
        description=u'/Main menu/f',
        default='-6.0',
        validators=[wtf.validators.InputRequired()])

    Main_menu__u0                  = wtf.TextField(
        label=u'u0',
        description=u'/Main menu/u0',
        default='1 + x[0]*x[0] + 2*x[1]*x[1]',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__allow_extrapolation = wtf.BooleanField(
        label=u'allow_extrapolation',
        description=u'/Main menu/dolfin/allow_extrapolation',
        default=False)

    Main_menu__dolfin__dof_ordering_library = wtf.TextField(
        label=u'dof_ordering_library',
        description=u'/Main menu/dolfin/dof_ordering_library',
        default='SCOTCH',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__ghost_mode  = wtf.TextField(
        label=u'ghost_mode',
        description=u'/Main menu/dolfin/ghost_mode',
        default='none',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__graph_coloring_library = wtf.TextField(
        label=u'graph_coloring_library',
        description=u'/Main menu/dolfin/graph_coloring_library',
        default='Boost',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__linear_algebra_backend = wtf.TextField(
        label=u'linear_algebra_backend',
        description=u'/Main menu/dolfin/linear_algebra_backend',
        default='PETSc',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__mesh_partitioner = wtf.TextField(
        label=u'mesh_partitioner',
        description=u'/Main menu/dolfin/mesh_partitioner',
        default='SCOTCH',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__num_threads = html5.IntegerField(
        label=u'num_threads',
        description=u'/Main menu/dolfin/num_threads',
        default=0,
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__partitioning_approach = wtf.TextField(
        label=u'partitioning_approach',
        description=u'/Main menu/dolfin/partitioning_approach',
        default='PARTITION',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__print_mpi_thread_support_level = wtf.BooleanField(
        label=u'print_mpi_thread_support_level',
        description=u'/Main menu/dolfin/print_mpi_thread_support_level',
        default=False)

    Main_menu__dolfin__refinement_algorithm = wtf.TextField(
        label=u'refinement_algorithm',
        description=u'/Main menu/dolfin/refinement_algorithm',
        default='plaza',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__relative_line_width = HTML5FloatField(
        label=u'relative_line_width',
        description=u'/Main menu/dolfin/relative_line_width',
        default=0.025,
        validators=[wtf.validators.InputRequired()],
        step='any',
        #min=-1000, max=%(maxvalue)g, step=0.001,
        )

    Main_menu__dolfin__reorder_cells_gps = wtf.BooleanField(
        label=u'reorder_cells_gps',
        description=u'/Main menu/dolfin/reorder_cells_gps',
        default=False)

    Main_menu__dolfin__reorder_dofs_serial = wtf.BooleanField(
        label=u'reorder_dofs_serial',
        description=u'/Main menu/dolfin/reorder_dofs_serial',
        default=True)

    Main_menu__dolfin__reorder_vertices_gps = wtf.BooleanField(
        label=u'reorder_vertices_gps',
        description=u'/Main menu/dolfin/reorder_vertices_gps',
        default=False)

    Main_menu__dolfin__std_out_all_processes = wtf.BooleanField(
        label=u'std_out_all_processes',
        description=u'/Main menu/dolfin/std_out_all_processes',
        default=True)

    Main_menu__dolfin__timer_prefix = wtf.TextField(
        label=u'timer_prefix',
        description=u'/Main menu/dolfin/timer_prefix',
        default='emptystring',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__use_petsc_signal_handler = wtf.BooleanField(
        label=u'use_petsc_signal_handler',
        description=u'/Main menu/dolfin/use_petsc_signal_handler',
        default=False)

    Main_menu__dolfin__warn_on_xml_file_size = html5.IntegerField(
        label=u'warn_on_xml_file_size',
        description=u'/Main menu/dolfin/warn_on_xml_file_size',
        default=100,
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__form_compiler__cache_dir = wtf.TextField(
        label=u'cache_dir',
        description=u'/Main menu/dolfin/form_compiler/cache_dir',
        default='emptystring',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__form_compiler__convert_exceptions_to_warnings = wtf.BooleanField(
        label=u'convert_exceptions_to_warnings',
        description=u'/Main menu/dolfin/form_compiler/convert_exceptions_to_warnings',
        default=False)

    Main_menu__dolfin__form_compiler__cpp_optimize = wtf.BooleanField(
        label=u'cpp_optimize',
        description=u'/Main menu/dolfin/form_compiler/cpp_optimize',
        default=True)

    Main_menu__dolfin__form_compiler__cpp_optimize_flags = wtf.TextField(
        label=u'cpp_optimize_flags',
        description=u'/Main menu/dolfin/form_compiler/cpp_optimize_flags',
        default='-O2',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__form_compiler__epsilon = HTML5FloatField(
        label=u'epsilon',
        description=u'/Main menu/dolfin/form_compiler/epsilon',
        default=1e-14,
        validators=[wtf.validators.InputRequired()],
        step='any',
        #min=-1000, max=%(maxvalue)g, step=0.001,
        )

    Main_menu__dolfin__form_compiler__error_control = wtf.BooleanField(
        label=u'error_control',
        description=u'/Main menu/dolfin/form_compiler/error_control',
        default=False)

    Main_menu__dolfin__form_compiler__form_postfix = wtf.BooleanField(
        label=u'form_postfix',
        description=u'/Main menu/dolfin/form_compiler/form_postfix',
        default=True)

    Main_menu__dolfin__form_compiler__format = wtf.TextField(
        label=u'format',
        description=u'/Main menu/dolfin/form_compiler/format',
        default='ufc',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__form_compiler__log_level = html5.IntegerField(
        label=u'log_level',
        description=u'/Main menu/dolfin/form_compiler/log_level',
        default=25,
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__form_compiler__log_prefix = wtf.TextField(
        label=u'log_prefix',
        description=u'/Main menu/dolfin/form_compiler/log_prefix',
        default='emptystring',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__form_compiler__name = wtf.TextField(
        label=u'name',
        description=u'/Main menu/dolfin/form_compiler/name',
        default='ffc',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__form_compiler__noevaluate_basis_derivatives = wtf.BooleanField(
        label=u'no-evaluate_basis_derivatives',
        description=u'/Main menu/dolfin/form_compiler/no-evaluate_basis_derivatives',
        default=True)

    Main_menu__dolfin__form_compiler__optimize = wtf.BooleanField(
        label=u'optimize',
        description=u'/Main menu/dolfin/form_compiler/optimize',
        default=False)

    Main_menu__dolfin__form_compiler__output_dir = wtf.TextField(
        label=u'output_dir',
        description=u'/Main menu/dolfin/form_compiler/output_dir',
        default='.',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__form_compiler__precision = html5.IntegerField(
        label=u'precision',
        description=u'/Main menu/dolfin/form_compiler/precision',
        default=15,
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__form_compiler__quadrature_degree = html5.IntegerField(
        label=u'quadrature_degree',
        description=u'/Main menu/dolfin/form_compiler/quadrature_degree',
        default=-1,
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__form_compiler__quadrature_rule = wtf.TextField(
        label=u'quadrature_rule',
        description=u'/Main menu/dolfin/form_compiler/quadrature_rule',
        default='auto',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__form_compiler__representation = wtf.TextField(
        label=u'representation',
        description=u'/Main menu/dolfin/form_compiler/representation',
        default='auto',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__form_compiler__restrict_keyword = wtf.TextField(
        label=u'restrict_keyword',
        description=u'/Main menu/dolfin/form_compiler/restrict_keyword',
        default='emptystring',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__form_compiler__split = wtf.BooleanField(
        label=u'split',
        description=u'/Main menu/dolfin/form_compiler/split',
        default=False)

    Main_menu__dolfin__krylov_solver__absolute_tolerance = HTML5FloatField(
        label=u'absolute_tolerance',
        description=u'/Main menu/dolfin/krylov_solver/absolute_tolerance',
        default=1e-05,
        validators=[wtf.validators.InputRequired()],
        step='any',
        #min=-1000, max=%(maxvalue)g, step=0.001,
        )

    Main_menu__dolfin__krylov_solver__divergence_limit = HTML5FloatField(
        label=u'divergence_limit',
        description=u'/Main menu/dolfin/krylov_solver/divergence_limit',
        default=10000,
        validators=[wtf.validators.InputRequired()],
        step='any',
        #min=-1000, max=%(maxvalue)g, step=0.001,
        )

    Main_menu__dolfin__krylov_solver__error_on_nonconvergence = wtf.BooleanField(
        label=u'error_on_nonconvergence',
        description=u'/Main menu/dolfin/krylov_solver/error_on_nonconvergence',
        default=True)

    Main_menu__dolfin__krylov_solver__maximum_iterations = html5.IntegerField(
        label=u'maximum_iterations',
        description=u'/Main menu/dolfin/krylov_solver/maximum_iterations',
        default=1000,
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__krylov_solver__monitor_convergence = wtf.BooleanField(
        label=u'monitor_convergence',
        description=u'/Main menu/dolfin/krylov_solver/monitor_convergence',
        default=False)

    Main_menu__dolfin__krylov_solver__nonzero_initial_guess = wtf.BooleanField(
        label=u'nonzero_initial_guess',
        description=u'/Main menu/dolfin/krylov_solver/nonzero_initial_guess',
        default=False)

    Main_menu__dolfin__krylov_solver__relative_tolerance = HTML5FloatField(
        label=u'relative_tolerance',
        description=u'/Main menu/dolfin/krylov_solver/relative_tolerance',
        default=0.001,
        validators=[wtf.validators.InputRequired()],
        step='any',
        #min=-1000, max=%(maxvalue)g, step=0.001,
        )

    Main_menu__dolfin__krylov_solver__report = wtf.BooleanField(
        label=u'report',
        description=u'/Main menu/dolfin/krylov_solver/report',
        default=True)

    Main_menu__dolfin__krylov_solver__gmres__restart = html5.IntegerField(
        label=u'restart',
        description=u'/Main menu/dolfin/krylov_solver/gmres/restart',
        default=30,
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__krylov_solver__preconditioner__report = wtf.BooleanField(
        label=u'report',
        description=u'/Main menu/dolfin/krylov_solver/preconditioner/report',
        default=False)

    Main_menu__dolfin__krylov_solver__preconditioner__shift_nonzero = HTML5FloatField(
        label=u'shift_nonzero',
        description=u'/Main menu/dolfin/krylov_solver/preconditioner/shift_nonzero',
        default=0,
        validators=[wtf.validators.InputRequired()],
        step='any',
        #min=-1000, max=%(maxvalue)g, step=0.001,
        )

    Main_menu__dolfin__krylov_solver__preconditioner__structure = wtf.TextField(
        label=u'structure',
        description=u'/Main menu/dolfin/krylov_solver/preconditioner/structure',
        default='different_nonzero_pattern',
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__krylov_solver__preconditioner__ilu__fill_level = html5.IntegerField(
        label=u'fill_level',
        description=u'/Main menu/dolfin/krylov_solver/preconditioner/ilu/fill_level',
        default=0,
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__krylov_solver__preconditioner__schwarz__overlap = html5.IntegerField(
        label=u'overlap',
        description=u'/Main menu/dolfin/krylov_solver/preconditioner/schwarz/overlap',
        default=1,
        validators=[wtf.validators.InputRequired()])

    Main_menu__dolfin__lu_solver__report = wtf.BooleanField(
        label=u'report',
        description=u'/Main menu/dolfin/lu_solver/report',
        default=True)

    Main_menu__dolfin__lu_solver__reuse_factorization = wtf.BooleanField(
        label=u'reuse_factorization',
        description=u'/Main menu/dolfin/lu_solver/reuse_factorization',
        default=False)

    Main_menu__dolfin__lu_solver__same_nonzero_pattern = wtf.BooleanField(
        label=u'same_nonzero_pattern',
        description=u'/Main menu/dolfin/lu_solver/same_nonzero_pattern',
        default=False)

    Main_menu__dolfin__lu_solver__symmetric = wtf.BooleanField(
        label=u'symmetric',
        description=u'/Main menu/dolfin/lu_solver/symmetric',
        default=False)

    Main_menu__dolfin__lu_solver__verbose = wtf.BooleanField(
        label=u'verbose',
        description=u'/Main menu/dolfin/lu_solver/verbose',
        default=False)
