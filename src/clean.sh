#!/bin/sh
find . \( -name 'hypercube_mesh.xml*' -o -name 'layers.xml*' -o -name 'subdomains.xml*' -o -name 'mesh.xml*' -o -name '*.pyc' -o -name 'u_layered.py res*' -o -name '*.vtu' -o -name '*.png' -o -name '*.eps' -o -name '*.pvd' -o -name '*.res tmp.tar.gz' -o -name '*~' \) -print -exec rm -rf {} \;
