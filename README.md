# Lorentz_as_module
I solve the Lorentz system in Python using the boost odeint library via the Python C API.
This code originated from my need to have control over the deployed numerical methods, while I still wanted to do analysis and plotting of the trajectory in Python.

By using odeint via the Numpy C-API I achieve exactly that, while also improving performance, because the computationally expensive solving procedure is now performed at C++ speed.

In order for the code to compile, you need to have installed: Numpy, python2.7-dev (you need the Python.h header file) and odeint. The latter is part of the boost library.
