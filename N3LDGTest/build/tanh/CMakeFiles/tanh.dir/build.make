# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/xzjiang/GPU-study/N3LDGTest

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/xzjiang/GPU-study/N3LDGTest/build

# Include any dependencies generated for this target.
include tanh/CMakeFiles/tanh.dir/depend.make

# Include the progress variables for this target.
include tanh/CMakeFiles/tanh.dir/progress.make

# Include the compile flags for this target's objects.
include tanh/CMakeFiles/tanh.dir/flags.make

tanh/CMakeFiles/tanh.dir/example1.cpp.o: tanh/CMakeFiles/tanh.dir/flags.make
tanh/CMakeFiles/tanh.dir/example1.cpp.o: ../tanh/example1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/xzjiang/GPU-study/N3LDGTest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tanh/CMakeFiles/tanh.dir/example1.cpp.o"
	cd /data/xzjiang/GPU-study/N3LDGTest/build/tanh && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tanh.dir/example1.cpp.o -c /data/xzjiang/GPU-study/N3LDGTest/tanh/example1.cpp

tanh/CMakeFiles/tanh.dir/example1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tanh.dir/example1.cpp.i"
	cd /data/xzjiang/GPU-study/N3LDGTest/build/tanh && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/xzjiang/GPU-study/N3LDGTest/tanh/example1.cpp > CMakeFiles/tanh.dir/example1.cpp.i

tanh/CMakeFiles/tanh.dir/example1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tanh.dir/example1.cpp.s"
	cd /data/xzjiang/GPU-study/N3LDGTest/build/tanh && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/xzjiang/GPU-study/N3LDGTest/tanh/example1.cpp -o CMakeFiles/tanh.dir/example1.cpp.s

tanh/CMakeFiles/tanh.dir/example1.cpp.o.requires:

.PHONY : tanh/CMakeFiles/tanh.dir/example1.cpp.o.requires

tanh/CMakeFiles/tanh.dir/example1.cpp.o.provides: tanh/CMakeFiles/tanh.dir/example1.cpp.o.requires
	$(MAKE) -f tanh/CMakeFiles/tanh.dir/build.make tanh/CMakeFiles/tanh.dir/example1.cpp.o.provides.build
.PHONY : tanh/CMakeFiles/tanh.dir/example1.cpp.o.provides

tanh/CMakeFiles/tanh.dir/example1.cpp.o.provides.build: tanh/CMakeFiles/tanh.dir/example1.cpp.o


# Object files for target tanh
tanh_OBJECTS = \
"CMakeFiles/tanh.dir/example1.cpp.o"

# External object files for target tanh
tanh_EXTERNAL_OBJECTS =

tanh/tanh: tanh/CMakeFiles/tanh.dir/example1.cpp.o
tanh/tanh: tanh/CMakeFiles/tanh.dir/build.make
tanh/tanh: /usr/local/cuda-8.0/lib64/libcudart_static.a
tanh/tanh: /usr/lib/x86_64-linux-gnu/librt.so
tanh/tanh: /usr/local/cuda-8.0/lib64/libcublas.so
tanh/tanh: /usr/local/cuda-8.0/lib64/libcublas_device.a
tanh/tanh: /data/xzjiang/GPU-study/N3LDG/lib/libmatrix.a
tanh/tanh: /usr/local/cuda-8.0/lib64/libcudart_static.a
tanh/tanh: /usr/lib/x86_64-linux-gnu/librt.so
tanh/tanh: /usr/local/cuda-8.0/lib64/libcublas.so
tanh/tanh: /usr/local/cuda-8.0/lib64/libcublas_device.a
tanh/tanh: /data/xzjiang/GPU-study/N3LDG/lib/libmatrix.a
tanh/tanh: tanh/CMakeFiles/tanh.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/xzjiang/GPU-study/N3LDGTest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tanh"
	cd /data/xzjiang/GPU-study/N3LDGTest/build/tanh && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tanh.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tanh/CMakeFiles/tanh.dir/build: tanh/tanh

.PHONY : tanh/CMakeFiles/tanh.dir/build

tanh/CMakeFiles/tanh.dir/requires: tanh/CMakeFiles/tanh.dir/example1.cpp.o.requires

.PHONY : tanh/CMakeFiles/tanh.dir/requires

tanh/CMakeFiles/tanh.dir/clean:
	cd /data/xzjiang/GPU-study/N3LDGTest/build/tanh && $(CMAKE_COMMAND) -P CMakeFiles/tanh.dir/cmake_clean.cmake
.PHONY : tanh/CMakeFiles/tanh.dir/clean

tanh/CMakeFiles/tanh.dir/depend:
	cd /data/xzjiang/GPU-study/N3LDGTest/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/xzjiang/GPU-study/N3LDGTest /data/xzjiang/GPU-study/N3LDGTest/tanh /data/xzjiang/GPU-study/N3LDGTest/build /data/xzjiang/GPU-study/N3LDGTest/build/tanh /data/xzjiang/GPU-study/N3LDGTest/build/tanh/CMakeFiles/tanh.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tanh/CMakeFiles/tanh.dir/depend

