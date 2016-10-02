from parampool.generator.flask import generate
from compute import compute, define_pool

generate(compute, pool_function=define_pool, MathJax=True)
