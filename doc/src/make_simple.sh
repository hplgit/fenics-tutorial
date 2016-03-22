#!/bin/bash
# Usage: make.sh latex|sphinx

name=ftut
version=1.0
version=1.1

# Doconce
preprocess -DFORMAT=pdflatex newcommands.p.tex > newcommands.tex
doconce format pdflatex $name --device=paper --latex_title_layout=titlepage "--latex_code_style=default:lst[style=yellow2_fb]@sys:vrb[frame=lines,label=\\fbox{{\tiny Terminal}},framesep=2.5mm,framerule=0.7pt,fontsize=\fontsize{9pt}{9pt}]" --encoding=utf-8 --latex_copyright=titlepages --latex_section_headings=blue
doconce replace 'frame=tb,' 'frame=tblr,' $name.tex

# LaTeX
pdflatex $name
bibtex $name
makeindex $name
pdflatex $name
pdflatex $name
cp $name.pdf fenics-tutorial-4print.pdf
