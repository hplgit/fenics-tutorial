#!/bin/sh
# Usage: make.sh latex|sphinx

name=ftut
version=1.0
version=1.1

# Format
if [ $# -eq 0 ]; then
  target=latex
else
  target=$1
fi

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
if [ $target = "latex" ]; then
# Make latex
doconce format latex $name --device=paper --latex_title_layout=std

#enc=`doconce guess_encoding $name.p.tex`
#if [ $enc = "utf-8" ]; then
#   echo 'change to latin1 encoding in $name.p.tex'
#   doconce change_encoding utf-8 iso-8859-1 $name.p.tex
#fi

doconce replace 'ptex2tex}' 'ptex2tex,subfigure}' $name.p.tex
doconce subst 'This document presents a' '\\tableofcontents\n\n\\clearpage\n\nThis document presents a' $name.p.tex
ptex2tex $name
latex $name
bibtex $name
makeindex $name
latex $name
makeindex $name
latex $name
# dvipdf gives error messages if $name.pdf is already open in evince...not serious
dvipdf $name
dest=fenics_tutorial${version}-4print
cp $name.pdf ${dest}.pdf

grep 'multiply defined' $name.log > ERRORS

echo
echo "Result: $dest.pdf"
fi

if [ $target = "sphinx" ]; then
# Make sphinx
# (Just remove sphinx-rootdir to have it recreated)

dir=sphinx-rootdir
rm -rf $dir
doconce sphinx_dir dirname=$dir author='H. P. Langtangen' version=$version theme=fenics $name
python automake_sphinx.py

dest=fenics_tutorial${version}-1
rm -rf $dest
cp -r $dir/_build/html .
mv html $dest
echo "Result: $dest/index.html"
fi

if [ $target = "sphinxm" ]; then
# Make sphinx with multiple .rst files

dir=sphinx-rootdir
rm -rf $dir
doconce format sphinx $name
doconce split_rst $name.rst

doconce sphinx_dir dirname=$dir author='H. P. Langtangen' version=$version theme=fenics $name
python automake_sphinx.py

dest=fenics_tutorial${version}
rm -rf $dest
cp -r $dir/_build/html .
mv html $dest
echo "Result: $dest/index.html"
fi

rm -rf ._*

# Copy
# rm -rf ~/vc/INF5620/doc/pub/fenics_tutorial1.1-1
# cp -r fenics_tutorial1.1-1 ~/vc/INF5620/doc/pub
# rm -rf ~/vc/INF5620/doc/pub/fenics_tutorial1.1
# cp -r fenics_tutorial1.1 ~/vc/INF5620/doc/pub
# cp $name.pdf ~/vc/INF5620/doc/pub/fenics_tutorial1.1-4print.pdf
