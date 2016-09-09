#!/bin/sh
set -x
if [ $# -ge 1 ]; then
  name=$1
else
  name="tmp_%04d.png"
fi
if [ $# -ge 2 ]; then
  fps=$2
else
  fps=25
fi
prog=ffmpeg

# Make movies in modern web friendly formats
rm -f movie.flv movie.mp4 movie.webm movie.ogg
$prog -i $name -r $fps -vcodec flv movie.flv
$prog -i $name -r $fps -vcodec libx264 movie.mp4
$prog -i $name -r $fps -vcodec libvpx movie.webm
$prog -i $name -r $fps -vcodec libtheora movie.ogg
