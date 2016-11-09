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
sudo pip2 install git+https://github.com/hplgit/doconce.git
sudo pip2 install git+https://github.com/hplgit/preprocess.git
sudo pip2 install hg+https://bitbucket.org/logg/publish
sudo pip2 install python-Levenshtein
