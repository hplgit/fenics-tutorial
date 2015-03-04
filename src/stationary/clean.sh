#!/bin/sh
find . \( -name '*.pvd' -o -name '*.vtu' -o -name '*.png' -o -name '*.eps' -o -name '*~' -o -name 'tmp*.py' \) -print -exec rm -f {} \;
