#!/usr/bin/env bash
#
# This script installs all required packages for building the book.

sudo apt-get update

# Install basic packages
sudo apt-get install \
     git mercurial texlive texlive-latex-extra ispell

# Install Chinese LaTeX
sudo apt-get install texlive-lang-chinese

# Install standard Python packages
sudo apt-get install \
     python-future python-mako python-lxml python-sphinx python-pip ipython

# Install extra Python packages
sudo pip2 install git+https://github.com/hplgit/doconce.git
sudo pip2 install git+https://github.com/hplgit/preprocess.git
sudo pip2 install hg+https://bitbucket.org/logg/publish
sudo pip2 install python-Levenshtein
