# !/bin/bash
# g++ ./hw2a.cc ./libs/thread_pool/thread_pool.h ./libs/thread_pool/thread_pool.c ./libs/queue/queue.h ./libs/queue/queue.c -o ./execs/hw2a -O3 -lm -pthread -lpng -mavx512vl
# g++ ./hw2a.cc -o ./execs/hw2a -O3 -lm -pthread -lpng
make
mv ./hw2a ./execs/hw2a
mv ./hw2b ./execs/hw2b
