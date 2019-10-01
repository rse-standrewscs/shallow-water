# shallow-water
3D shallow water code

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
