# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings

COMPILER = /home/users/geleta/VAEgen/compiled/hdf5-1.12.0/c++/src/h5c++
CPPFLAGS  = -g -Wall -std=c++11 #-stdlib=libc++

# Boost C++ lib.
BOOSTHEADERS = -I/home/users/geleta/VAEgen/compiled/boost_1_76_0
BOOSTLIB = -L/home/users/geleta/VAEgen/compiled/boost_1_76_0/stage/lib
BOOSTFLAGS = -lboost_regex

# HDF5 lib.
HDF5HEADERS = -I/home/users/geleta/VAEgen/compiled/hdf5-1.12.0/src
HDF5LIB = -L/home/users/geleta/VAEgen/compiled/hdf5-1.12.0/src/.libs
HDF5FlAGS = -lhdf5 -lhdf5_cpp

# Collect links.
HEADERS = $(BOOSTHEADERS) $(HDF5HEADERS)
LIBRARIES = $(BOOSTLIB) $(HDF5LIB)
FLAGS = $(BOOSTFLAGS) $(HDF5FlAGS)

# Build target.
TARGET = readvcf

# BEGIN MAKE
all: $(TARGET)

$(TARGET): $(TARGET).cc
	$(COMPILER) $(CPPFLAGS) $(HEADERS) $(LIBRARIES) -O3 -o $(TARGET).elf $(TARGET).cc $(FLAGS)

clean:
	$(RM) $(TARGET)
