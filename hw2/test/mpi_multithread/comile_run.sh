# !/bin/bash
mpicxx mpi_multithread_test.cc -o mpi_multithread_test
srun mpi_multithread_test