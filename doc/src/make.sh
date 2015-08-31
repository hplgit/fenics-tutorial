#!/bin/bash
# Usage: make.sh latex|sphinx

name=ftut
version=1.0
version=1.1

set -x

function system {
  "$@"
  if [ $? -ne 0 ]; then
    echo "make.sh: unsuccessful command $@"
    echo "abort!"
    exit 1
  fi
}

# We have Norwegian characters in the acknowledgement and this
# requires latin1 for latex and utf-8 for rst/sphinx

# First make sure .do.txt is latin1
#for file in *.do.txt; do
#  enc=`doconce guess_encoding $file`
#  if [ $enc = "utf-8" ]; then
#     echo "change from $enc to latin1 encoding in $file"
#     doconce change_encoding utf-8 iso-8859-1 $file
#  fi
#done
#

# Make latex
preprocess -DFORMAT newcommands_keep.p.tex > newcommands_keep.tex
system doconce format pdflatex $name --device=paper --latex_title_layout=titlepage "--latex_code_style=default:vrb-blue1@sys:vrb[frame=lines,label=\\fbox{{\tiny Terminal}},framesep=2.5mm,framerule=0.7pt,fontsize=\fontsize{9pt}{9pt}]"

#enc=`doconce guess_encoding $name.p.tex`
#if [ $enc = "utf-8" ]; then
#   echo 'change to latin1 encoding in $name.p.tex'
#   doconce change_encoding utf-8 iso-8859-1 $name.p.tex
#fi

#doconce replace 'ptex2tex}' 'ptex2tex,subfigure}' $name.p.tex
#doconce subst 'This document presents a' '\\tableofcontents\n\n\\clearpage\n\nThis document presents a' $name.p.tex
system pdflatex $name
system bibtex $name
system makeindex $name
pdflatex $name
pdflatex $name
cp $name.pdf ${name}-4print.pdf

exit
# Make sphinx
preprocess -DHTML newcommands_keep.p.tex > newcommands_keep.tex
dir=sphinx-rootdir
system doconce format sphinx $name
system doconce split_rst $name.rst
system doconce sphinx_dir dirname=$dir copyright='H. P. Langtangen' version=$version theme=fenics $name
system python automake_sphinx.py

dest=../pub
cp -r *.pdf sphinx-rootdir/_build/html $dest
