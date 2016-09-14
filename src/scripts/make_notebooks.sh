#!/bin/bash
set -x

function system {
  "$@"
  if [ $? -ne 0 ]; then
    echo "make.sh: unsuccessful command $@"
    echo "abort!"
    exit 1
  fi
}

EXV=True

system preprocess -DFORMAT=html newcommands.p.tex > newcommands.tex
filenames='poisson0 membrane0 diffusion0'
for filename in $filenames; do
    doconce format ipynb $filename --encoding=utf-8 EXV=$EXV
done

# Publish in doc/pub
dest=../pub
cp *.ipynb $dest
