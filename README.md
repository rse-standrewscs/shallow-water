# shallow-water

[![Build Status](https://travis-ci.com/chocol4te/shallow-water.svg?token=Yy278F6KPxruhJLf8xog&branch=master)](https://travis-ci.com/chocol4te/shallow-water)

3D shallow water code by David Dritschel

http://www-vortex.mcs.st-and.ac.uk/software.html

## Build instructions

    cd src
    make all clean

## Usage instructions

In the root directory of the project:

    ./vstrip < in_vstrip
    ./balinit
    ./swto3d
    ./nhswps > log &

When finished, inspect `monitor.asc` and `log` files.
