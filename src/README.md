This directory contains the source code for the FEniCS Tutorial.

The tutorial is written in DocOnce and compiled to LaTeX PDF, Sphinx
(FEniCS style), and Bootstrap HTML.

The parent documents are:

 * `ftut1.do.txt` (FEniCS Tutorial Volume 1
 * `ftut2.do.txt` (FEniCS Tutorial Volume 2, in preparation)

The directories `vol1` and `vol2` contain the sources for chapters
and code for the two volumes.

To build the book (in all formats), type `make` or run `scripts/make.sh`.

Building the book requires a relatively long list of packages, most notably
`doconce`, `preprocess`, `publish`, and `sphinx`. The easiest way to install
these packages is via the script `install_rich.sh` which can be found in
the subdirectory `scripts`. If running on Ubuntu, the script should just
work and install all required packages.

If running on another operating system, the easiest solution is to
create a FEniCS Docker image and run the script from within that
directory. Make sure you have installed Docker and the FEniCS Docker
scripts; see `http://fenicsproject.org/download for instructions`.
The position yourself in the top level directory of this repository
and then run the following commands:

    fenicsproject create fenics-tutorial stable
    fenicsproject start fenics-tutorial

Then run the install script inside the container:

    src/scripts/install.sh
