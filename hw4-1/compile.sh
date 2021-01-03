# !/bin/bash
# make
nvcc  -O3 -std=c++11 -Xptxas=-v -arch=sm_61  -lm -o ${1} ${1}.cu
mv ${1} ./execs/${1}