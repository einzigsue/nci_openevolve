# Compiler settings
CXX       := g++
NVCC      := nvcc
CXXFLAGS  := -O2 -I$(CUDA_HOME)/include
NVCCFLAGS := -O2
LDFLAGS   := -L$(CUDA_HOME)/lib64 -lcudart

# Sources and objects
CU_SRC    := nbody_kernel.cu
CPP_SRC   := nbody_cpu.cpp
CU_OBJ    := $(CU_SRC:.cu=.o)
CPP_OBJ   := $(CPP_SRC:.cpp=.o)

# Executables
EXES      := main validate

all: $(EXES)

# Compile CUDA file
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ file needing cuda_runtime.h
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link main
main: main.o $(CU_OBJ) $(CPP_OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

# Link validate
validate: validate.o $(CU_OBJ) $(CPP_OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.o $(EXES)

