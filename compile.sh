#!/bin/sh
numpyfile=`python -c "import numpy; import sys; sys.stdout.write(numpy.__file__)"`
numpydir=`dirname $numpyfile` 

pythonfile=`locate python2.7/Python.h`
pythondir=`dirname $pythonfile`

g++ -O2 -fPIC -I$pythondir -I$numpydir/core/include/numpy -c c_module.cpp -o c_module.o
g++ -shared -fPIC -o lorentzmodule.so c_module.o 
