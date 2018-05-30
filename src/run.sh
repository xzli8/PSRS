#!/bin/bash

# EXE :
EXE=./PSRS

# input parameters :
np=4
n=64

mpirun -np ${np} ${EXE} ${n}
