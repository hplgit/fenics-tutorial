#!/bin/sh
#rm -f automake_sphinx.py  # not used here, see make.sh
doconce clean
cp tu2_bib.rst copy  # copy before removing all *.rst
rm -rf *~ *.out *.log *.dvi *.aux *.ilg *.idx *.ind tu2.tex *.bbl tu2.rst ERRORS newcommands_replace.tex newcommands_keep.tex tmp* *.rst
mv copy tu2_bib.rst  # restore
