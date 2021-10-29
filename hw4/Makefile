CXX=g++
CFLAGS=-c
SOURCES1=matpower.cpp matrix_operations.cpp
SOURCES2=page_rank.cpp matrix_operations.cpp
EXECUTABLE=task1 task2
LIBS=-lblas -lcblas -fopenmp -m64 -I${MKLROOT}/include -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl

all: $(EXECUTABLE)

task1: $(SOURCES1)
			$(CXX) $(SOURCES1) $(LIBS) $(EXTFLAGS) -o task1

task2: $(SOURCES2)
			$(CXX) $(SOURCES2) $(LIBS) $(EXTFLAGS) -o task2

clean:
	rm -rf *o task1 task2
