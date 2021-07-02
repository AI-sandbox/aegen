# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings

COMPILER = g++ #/home/users/geleta/VAEgen/compiled/hdf5-1.12.0/c++/src/h5c++
CPPFLAGS  = -g -Wall -std=c++11 #-stdlib=libc++

# Boost C++ lib.
BOOSTHEADERS = -I/home/users/geleta/VAEgen/compiled/boost_1_76_0
BOOSTLIB = -L/home/users/geleta/VAEgen/compiled/boost_1_76_0/stage/lib
BOOSTFLAGS = -lboost_regex

# HDF5 lib.
HDF5HEADERS = -I/home/users/geleta/VAEgen/compiled/hdf5-1.12.0
HDF5LIB = -L/home/users/geleta/VAEgen/compiled/hdf5-1.12.0/src/.libs
HDF5FlAGS = -lhdf5 -lhdf5_cpp

# Collect links.
HEADERS = $(BOOSTHEADERS) $(HDF5HEADERS)
LIBRARIES = $(BOOSTLIB) $(HDF5LIB)
FLAGS = $(BOOSTFLAGS) $(HDF5FlAGS)

# Build target.
TARGET = readvcf

# Load requires libraries
# hdf5/1.10.6 lib version does not work
# because for some reason the header file
# <H5Cpp.h> is missing. Use hdf5/1.10.0p1 instead.
ml load system devel hdf5/1.10.0p1 boost/1.76.0

# BEGIN MAKE
all: $(TARGET)

$(TARGET): $(TARGET).cc
	$(COMPILER) $(CPPFLAGS) -O3 -o $(TARGET).elf $(TARGET).cc $(FLAGS)

clean:
	$(RM) $(TARGET)
