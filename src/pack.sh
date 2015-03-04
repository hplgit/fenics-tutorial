#!/bin/sh
set -x
src="fenics_tutorial_examples"

# Pack example programs
rm -rf $src
mkdir $src
sh clean.sh
cp -r stationary transient $src
clean.py $src
rm -f $src/stationary/poisson/membrane2_class.py

tarfile=fenics_tutorial_examples
tar cvzf $tarfile.tar.gz $src
zip -r   $tarfile.zip    $src

find $src -name '*.py' -type f -exec doconce replace nabla_grad grad {} \;
sh clean.sh
tar cvzf ${tarfile}_nonabla.tar.gz $src
zip -r   ${tarfile}_nonabla.zip    $src
