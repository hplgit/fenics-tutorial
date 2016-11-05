#!/bin/bash
# Usage: make.sh latex|sphinx

if [ $# -gt 0 ]; then
    name=ftut${1}
    bookno=$1
else
    name=ftut1
    bookno=1
fi

version=1.6  # FEniCS version

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
#system doconce spellcheck -d .dict4spell.txt *.do.txt
#doconce spellcheck -d .dict4spell.txt *.do.txt

# EXV: Extended Version of the book (used for exercises and
# advanced material not to appear in the 150 page printed SSBrief version)
EXV=False

# We have Norwegian characters in the acknowledgement and this requires utf-8

# Generate latex output
system preprocess -DFORMAT=pdflatex newcommands.p.tex > newcommands.tex
function compile {
    options="$@"
# Blue headings, FEniCS book style code:
#system doconce format pdflatex $name --latex_title_layout=titlepage "--latex_code_style=default:lst[style=yellow2_fb]@sys:vrb[frame=lines,label=\\fbox{{\tiny Terminal}},framesep=2.5mm,framerule=0.7pt,fontsize=\fontsize{9pt}{9pt}]" --encoding=utf-8 --latex_copyright=titlepages --latex_section_headings=blue
# Fix: make full box around code blocks a la the FEniCS book
#doconce replace 'frame=tb,' 'frame=tblr,' $name.tex
system doconce format pdflatex $name --exercise_numbering=chapter --latex_style=Springer_sv --latex_title_layout=std --latex_list_of_exercises=none --latex_admon=mdfbox --latex_admon_color=1,1,1 --latex_table_format=left --latex_admon_title_no_period --latex_no_program_footnotelink "--latex_code_style=default:lst[style=graycolor]@sys:vrb[frame=lines,label=\\fbox{{\tiny Terminal}},framesep=2.5mm,framerule=0.7pt,fontsize=\fontsize{9pt}{9pt}]" --exercises_as_subsections --encoding=utf-8 --movie_prefix=https://raw.githubusercontent.com/hplgit/fenics-tutorial/brief/doc/src/ --allow_refs_to_external_docs $options

# Fix layout for admons: gray box, light gray background for title
doconce replace 'linecolor=black,' 'linecolor=gray,' $name.tex
doconce replace 'Released under CC Attr' '\\ Released under CC Attr' $name.tex
doconce subst 'frametitlebackgroundcolor=.*?,' 'frametitlebackgroundcolor=gray!5,' $name.tex

# Compile latex
system pdflatex $name
#system bibtex $name
#system makeindex $name
#pdflatex $name
#pdflatex $name
}

# Printed book
compile --device=paper EXV=False
cp $name.pdf fenics-tutorial${bookno}-4print.pdf
