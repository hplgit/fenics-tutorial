#!/bin/bash
# Usage: make.sh 1|2
# for making volume 1 or 2

if [ $# -gt 0 ]; then
    name=ftut$1
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
system doconce spellcheck -d .dict4spell.txt *.do.txt
#doconce spellcheck -d .dict4spell.txt *.do.txt

# EXV: Extended Version of the book (used for exercises and
# advanced material not to appear in the 150 page printed SSBrief version)
EXV=False

# We have Norwegian characters in the acknowledgement and this requires utf-8

# Generate latex output
system preprocess -DFORMAT=pdflatex newcommands.p.tex > newcommands.tex

function edit_solution_admons {
    # We use question admon for typesetting solution, but let's edit to
    # somewhat less eye catching than the std admon
    # (also note we use --latex_admon_envir_map= in compile)
    doconce replace 'notice_mdfboxadmon}[Solution.]' 'question_mdfboxadmon}[Solution.]' ${name}.tex
    doconce replace 'end{notice_mdfboxadmon} % title: Solution.' 'end{question_mdfboxadmon} % title: Solution.' ${name}.tex
    doconce subst -s '% "question" admon.+?question_mdfboxmdframed\}' '% "question" admon
\colorlet{mdfbox_question_background}{gray!5}
\\newmdenv[        % edited for solution admons in exercises
  skipabove=15pt,
  skipbelow=15pt,
  outerlinewidth=0,
  backgroundcolor=white,
  linecolor=black,
  linewidth=1pt,       % frame thickness
  frametitlebackgroundcolor=blue!5,
  frametitlerule=true,
  frametitlefont=\\normalfont\\bfseries,
  shadow=false,        % frame shadow?
  shadowsize=11pt,
  leftmargin=0,
  rightmargin=0,
  roundcorner=5,
  needspace=0pt,
]{question_mdfboxmdframed}' ${name}.tex
}

function compile {
    options="$@"
system doconce format pdflatex $name --exercise_numbering=chapter --latex_style=Springer_sv --latex_title_layout=std --latex_list_of_exercises=none --latex_admon=mdfbox --latex_admon_color=1,1,1 --latex_table_format=left --latex_admon_title_no_period --latex_no_program_footnotelink "--latex_code_style=default:lst[style=graycolor]@sys:vrb[frame=lines,label=\\fbox{{\tiny Terminal}},framesep=2.5mm,framerule=0.7pt,fontsize=\fontsize{9pt}{9pt}]" --exercises_as_subsections --encoding=utf-8 --movie_prefix=https://raw.githubusercontent.com/hplgit/fenics-tutorial/brief/doc/src/ --allow_refs_to_external_docs $options

# Auto edits
edit_solution_admons
# With t4/svmono linewidth has some too large value before \mymainmatter
# is called, so the box width as linewidth+2mm is wrong, it must be
# explicitly set to 120mm.
doconce replace '\setlength{\lstboxwidth}{\linewidth+2mm}' '\setlength{\lstboxwidth}{120mm}' $name.tex  # lst

# Fix layout for admons: gray box, light gray background for title
doconce replace 'linecolor=black,' 'linecolor=gray,' $name.tex
doconce replace 'Released under CC Attr' '\\ Released under CC Attr' $name.tex
doconce subst 'frametitlebackgroundcolor=.*?,' 'frametitlebackgroundcolor=gray!5,' $name.tex

# Compile latex
system pdflatex $name
system bibtex $name
system makeindex $name
pdflatex $name
pdflatex $name
}

# Printed book
compile --device=paper EXV=False
cp $name.pdf fenics-tutorial-vol${bookno}.pdf
cp $name.log fenics-tutorial-vol${bookno}.log  # save to track the no of pages!
cp $name.dlog ${name}.dlog  # for examining error messages

# PDF online ebook (extended version with exercises etc.)
#compile --device=screen EXV=True
#cp $name.pdf fenics-tutorial-vol${bookno}-extended-beta.pdf

# Make sphinx
rm -f *._ftut*.rst
preprocess -DFORMAT=html newcommands.p.tex > newcommands.tex
dir=sphinx-rootdir
system doconce format sphinx $name --encoding=utf-8 EXV=$EXV --allow_refs_to_external_docs
cp $name.dlog ${name}-sphinx.dlog  # for examining error messages
system doconce split_rst $name.rst
system doconce sphinx_dir dirname=${dir}${bookno} version=$version theme=fenics $name
system python automake_sphinx.py

# Make Bootstrap HTML (but enlargen the journal font)
system doconce format html $name --encoding=utf-8 --html_style=bootswatch_journal "--html_body_style=font-size:20px;line-height:1.5" --html_code_style=inherit EXV=$EXV --allow_refs_to_external_docs
cp $name.dlog ${name}-html.dlog  # for examining error messages
system doconce split_html $name.html --pagination

# Replace http by https to make Bootstrap HTML work on FEniCS server
sed -i.bak -e 's=http://fenicsproject=https://fenicsproject=g' *.html .*.html
sed -i.bak -e 's=http://netdna=https://netdna=g' *.html .*.html
sed -i.bak -e 's=http://ajax=https://ajax=g' *.html .*.html
sed -i.bak -e 's=http://cdn=https://cdn=g' *.html .*.html

# Root directory for published documents
dest=../pub

# Copy PDF to output (publication) directory
cp fenics-tutorial*.pdf $dest/pdf

# Copy HTML to output (publication) directory
cp -r $name.html ._*.html fig mov $dest/html

# Copy Sphinx to output (publication) directory
rm -rf $dest/sphinx${bookno}
cp -r sphinx-rootdir${bookno}/_build/html $dest/sphinx${bookno}

# Copy tutorial programs to output (publication) directory
python scripts/number_src_files.py
