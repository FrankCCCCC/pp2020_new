# !/bin/bash
nvcc -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler="-pthread" -lm -o ${1} ${1}.cu
mv ${1} ./execs/${1}