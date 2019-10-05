#!/bin/sh
set -e

# Basic test script to be used until more sophisticated testing infrastructure is created

sed -i "s/integer,parameter:: ng=128,nz=16/integer,parameter:: ng=16,nz=2/g" ../src/parameters.f90

# Build source
echo '# BUILDING'
cd ../src
make clean > /dev/null
make all > /dev/null

# Execute
echo '# RUNNING'
cd ..
./vstrip < in_vstrip > /dev/null
./balinit > /dev/null
./swto3d > /dev/null
./nhswps > log

cd tst

# Restore original parameters
sed -i "s/integer,parameter:: ng=16,nz=2/integer,parameter:: ng=128,nz=16/g" ../src/parameters.f90

cmp --silent ../monitor.asc monitor.asc && echo '# SUCCESS' || echo '# Error! monitor.asc not equal to expected'
