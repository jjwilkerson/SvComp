# Compiler settings
NVCC = /usr/local/cuda-12.1/bin/nvcc
CCFLAGS = -I"../NetLib" -I/usr/local/include/utf8 -I/usr/local/cuda-samples/Common -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 -ccbin g++
LINKERFLAGS = -L"../NetLib" -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu -L/usr/local/cuda-12.1/lib64 -lNetLib -lboost_system -lboost_regex -lboost_filesystem -ljsoncpp -lcublas

# Find all .cpp files in the current directory and subdirectories
SRCS = $(wildcard *.cpp) $(wildcard */*.cpp) $(wildcard */*/*.cpp)

# Object files corresponding to source files
OBJS = $(SRCS:.cpp=.o)
PROGRAM_VERSION = 1 # Default program version (single)

# Name of the executable
EXEC = SvComp

# Default target (single with residual-recurrent)
all: $(EXEC)

# Define build rules for different configurations
index: PROGRAM_VERSION = 2
index: clean all

# Compile source files into object files
%.o: %.cpp
	$(NVCC) -DPROGRAM_VERSION=$(PROGRAM_VERSION) $(CCFLAGS) -c $< -o $@

# Build the executable
$(EXEC): $(OBJS)
	$(NVCC) --cudart=static -o $@ $^ $(LINKERFLAGS)

# Clean up
clean:
	rm -f $(OBJS) $(EXEC)
