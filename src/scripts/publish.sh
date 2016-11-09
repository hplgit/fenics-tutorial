#!/usr/bin/env sh

rsync -avz --delete ../pub/ fenics-web@fenicsproject.org:/home/fenics-web/pub/tutorial/
