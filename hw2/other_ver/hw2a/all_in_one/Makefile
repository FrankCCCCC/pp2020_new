CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3
hw2a: CFLAGS += -pthread
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = hw2a
# DEPS = libs/thread_pool/thread_pool.h libs/thread_pool/thread_pool.c libs/queue/queue.h libs/queue/queue.c

.PHONY: all
all:
	$(CXX) $(TARGETS).cc $(CXXFLAGS) $(LDLIBS) -pthread -o $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
