# shallow-water

3D shallow water code by David Dritschel

http://www-vortex.mcs.st-and.ac.uk/software.html

## build instructions

    cd src
    make all clean

## usage instructions

In the root directory of the project:

    ./vstrip < in_vstrip 
    ./balinit 
    ./swto3d 
    ./nhswps > log &

When finished, inspect `monitor.asc` and `log` files.
