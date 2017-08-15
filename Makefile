CPP = g++
OFLAG = -o
GSLFLAG = -L/usr/lib/x86_64-linux-gnu -lgsl -lgslcblas -lm
OPENMPFLAG = -fopenmp
CUDAFLAGS = -lcudart -lcublas -lcusolver

source = pureDetectEpi.cpp


all:\
	out2 \
	out


cuLUsolve.o: cuLUsolve.h cuLUsolve.cu
	nvcc -c cuLUsolve.cu

cuMultifit.o: cuLUsolve.h cuMultifit.h cuMultifit.cu
	nvcc -c cuMultifit.cu

cudaDetectEpi.o: cuMultifit.h cudaDetectEpi.cpp
	g++ -c cudaDetectEpi.cpp


mpout: $(source)
	$(CPP) $(source) $(OFLAG) mpout -O3 $(GSLFLAG) $(OPENMPFLAG)

out: $(source)
	$(CPP) $(source) $(OFLAG) out $(GSLFLAG)

mpout2: cuLUsolve.o cuMultifit.o cudaDetectEpi.o
	nvcc $^ -o $@ $(CUDAFLAGS) $(GSLFLAG) $(OPENMPFLAG)

out2: cuLUsolve.o cuMultifit.o cudaDetectEpi.o
	nvcc $^ -o $@ $(CUDAFLAGS) $(GSLFLAG)

solveBeta:
	nvcc -o ols2 solveBeta2.cu -lcublas_device -lcudadevrt -arch=sm_35 -rdc=true -lgsl -lgslcblas -g
