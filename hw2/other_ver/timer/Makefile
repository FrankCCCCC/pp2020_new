CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3
hw2a: CFLAGS += -pthread
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp -pthread
CXXFLAGS = $(CFLAGS)
hw2a_t=hw2a-t
hw2b_t=hw2b-t
DEPS_A = ../../libs/timer/timer.c ../../libs/timer/timer.h
DEPS_B = ../../libs/timer_mpi/timer_mpi.c ../../libs/timer_mpi/timer_mpi.h

main: hw2a hw2b
	
hw2a:
	$(CXX) $(hw2a_t).cc $(DEPS_A) $(CXXFLAGS) $(LDLIBS) -o $(hw2a_t)
	# $(CXX) $(hw2a_t).cc $(DEPS) $(CXXFLAGS) $(LDLIBS) -o $(hw2a_t)
hw2b:
	$(CXX) $(hw2b_t).cc $(DEPS_B) $(CXXFLAGS) $(LDLIBS) -o $(hw2b_t)

.PHONY: clean
clean:
	rm -f $(hw2a_t) $(hw2b_t) $(hw2a_t:=.o) $(hw2b_t:=.o)
