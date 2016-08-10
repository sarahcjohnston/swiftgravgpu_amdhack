#!/bin/bash

# Generate the initial conditions if they are not present.
if [ ! -e sodShock.hdf5 ]
then
    echo "Generating initial conditions for the 1D SodShock example..."
    python makeIC.py
fi

# Run SWIFT
../swift -s -t 1 sodShock.yml

# Plot the result
python plotSolution.py 1
