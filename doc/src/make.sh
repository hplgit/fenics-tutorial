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

rm -rf tmp_*.do.txt  # don't spellcheck old versions
system doconce spellcheck -d .dict4spell.txt *.do.txt

# EXV: Extended Version of the book (used for exercises and
# advanced material not to appear in the 150 page printed SSBrief version)
EXV=True

# We have Norwegian characters in the acknowledgement and this requires utf-8

# Generate latex output
system preprocess -DFORMAT=pdflatex newcommands.p.tex > newcommands.tex
function compile {
    options="$@"
# Blue headings, FEniCS book style code:
#system doconce format pdflatex $name --latex_title_layout=titlepage "--latex_code_style=default:lst[style=yellow2_fb]@sys:vrb[frame=lines,label=\\fbox{{\tiny Terminal}},framesep=2.5mm,framerule=0.7pt,fontsize=\fontsize{9pt}{9pt}]" --encoding=utf-8 --latex_copyright=titlepages --latex_section_headings=blue
# Fix: make full box around code blocks a la the FEniCS book
#doconce replace 'frame=tb,' 'frame=tblr,' $name.tex
system doconce format pdflatex $name --exercise_numbering=chapter --latex_style=Springer_sv --latex_title_layout=std --latex_list_of_exercises=none --latex_admon=mdfbox --latex_admon_color=1,1,1 --latex_table_format=left --latex_admon_title_no_period --latex_no_program_footnotelink "--latex_code_style=default:lst[style=graycolor]@sys:vrb[frame=lines,label=\\fbox{{\tiny Terminal}},framesep=2.5mm,framerule=0.7pt,fontsize=\fontsize{9pt}{9pt}]" --exercises_as_subsections --encoding=utf-8 $options

# Compile latex
system pdflatex $name
system bibtex $name
system makeindex $name
pdflatex $name
pdflatex $name
}

# Printed book
compile --device=paper EXV=False
cp $name.pdf fenics-tutorial-4print.pdf
cp $name.log fenics-tutorial-4print.log  # needed to track the no of pages!
# Online book (exetended version with exercises etc.)
compile --device=screen EXV=True
cp $name.pdf fenics-tutorial-4screen.pdf

# Make sphinx
preprocess -DFORMAT=html newcommands.p.tex > newcommands.tex
dir=sphinx-rootdir
system doconce format sphinx $name --encoding=utf-8 EXV=$EXV
system doconce split_rst $name.rst
system doconce sphinx_dir dirname=$dir version=$version theme=fenics $name
system python automake_sphinx.py

# Make Bootstrap HTML
system doconce format html $name --encoding=utf-8 --html_style=bootswatch_journal EXV=$EXV
system doconce split_html $name.html --pagination

# Publish in doc/pub
dest=../pub
rm -rf $dest/sphinx
cp -r fenics-tutorial-4print.pdf sphinx-rootdir/_build/html $dest
mv -f $dest/html $dest/sphinx
cp -r $name.html ._*.html fig $dest
