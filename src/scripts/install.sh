#!/usr/bin/env bash
#
# This script installs all required packages for building the book.

sudo apt-get update

# Install basic packages
sudo apt-get install \
     git mercurial texlive texlive-latex-extra ispell

# Install standard Python packages
sudo apt-get install \
     python-future python-mako python-lxml python-sphinx

# Install extra Python packages
sudo pip install git+https://github.com/hplgit/doconce.git
sudo pip install git+https://github.com/hplgit/preprocess.git
sudo pip install hg+https://bitbucket.org/logg/publish
sudo pip install python-Levenshtein
