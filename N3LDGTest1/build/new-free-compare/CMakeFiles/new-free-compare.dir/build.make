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
CMAKE_SOURCE_DIR = /data/xzjiang/GPU-study/N3LDGTest1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/xzjiang/GPU-study/N3LDGTest1/build

# Include any dependencies generated for this target.
include new-free-compare/CMakeFiles/new-free-compare.dir/depend.make

# Include the progress variables for this target.
include new-free-compare/CMakeFiles/new-free-compare.dir/progress.make

# Include the compile flags for this target's objects.
include new-free-compare/CMakeFiles/new-free-compare.dir/flags.make

new-free-compare/CMakeFiles/new-free-compare.dir/example1.cpp.o: new-free-compare/CMakeFiles/new-free-compare.dir/flags.make
new-free-compare/CMakeFiles/new-free-compare.dir/example1.cpp.o: ../new-free-compare/example1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/xzjiang/GPU-study/N3LDGTest1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object new-free-compare/CMakeFiles/new-free-compare.dir/example1.cpp.o"
	cd /data/xzjiang/GPU-study/N3LDGTest1/build/new-free-compare && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/new-free-compare.dir/example1.cpp.o -c /data/xzjiang/GPU-study/N3LDGTest1/new-free-compare/example1.cpp

new-free-compare/CMakeFiles/new-free-compare.dir/example1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/new-free-compare.dir/example1.cpp.i"
	cd /data/xzjiang/GPU-study/N3LDGTest1/build/new-free-compare && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/xzjiang/GPU-study/N3LDGTest1/new-free-compare/example1.cpp > CMakeFiles/new-free-compare.dir/example1.cpp.i

new-free-compare/CMakeFiles/new-free-compare.dir/example1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/new-free-compare.dir/example1.cpp.s"
	cd /data/xzjiang/GPU-study/N3LDGTest1/build/new-free-compare && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/xzjiang/GPU-study/N3LDGTest1/new-free-compare/example1.cpp -o CMakeFiles/new-free-compare.dir/example1.cpp.s

new-free-compare/CMakeFiles/new-free-compare.dir/example1.cpp.o.requires:

.PHONY : new-free-compare/CMakeFiles/new-free-compare.dir/example1.cpp.o.requires

new-free-compare/CMakeFiles/new-free-compare.dir/example1.cpp.o.provides: new-free-compare/CMakeFiles/new-free-compare.dir/example1.cpp.o.requires
	$(MAKE) -f new-free-compare/CMakeFiles/new-free-compare.dir/build.make new-free-compare/CMakeFiles/new-free-compare.dir/example1.cpp.o.provides.build
.PHONY : new-free-compare/CMakeFiles/new-free-compare.dir/example1.cpp.o.provides

new-free-compare/CMakeFiles/new-free-compare.dir/example1.cpp.o.provides.build: new-free-compare/CMakeFiles/new-free-compare.dir/example1.cpp.o


# Object files for target new-free-compare
new__free__compare_OBJECTS = \
"CMakeFiles/new-free-compare.dir/example1.cpp.o"

# External object files for target new-free-compare
new__free__compare_EXTERNAL_OBJECTS =

new-free-compare/new-free-compare: new-free-compare/CMakeFiles/new-free-compare.dir/example1.cpp.o
new-free-compare/new-free-compare: new-free-compare/CMakeFiles/new-free-compare.dir/build.make
new-free-compare/new-free-compare: /usr/local/cuda-8.0/lib64/libcudart_static.a
new-free-compare/new-free-compare: /usr/lib/x86_64-linux-gnu/librt.so
new-free-compare/new-free-compare: /usr/local/cuda-8.0/lib64/libcublas.so
new-free-compare/new-free-compare: /usr/local/cuda-8.0/lib64/libcublas_device.a
new-free-compare/new-free-compare: /data/xzjiang/GPU-study/N3LDG/lib/libmatrix.a
new-free-compare/new-free-compare: /data/xzjiang/GPU-study/cnmem/build/libcnmem.so
new-free-compare/new-free-compare: /usr/local/cuda-8.0/lib64/libcudart_static.a
new-free-compare/new-free-compare: /usr/lib/x86_64-linux-gnu/librt.so
new-free-compare/new-free-compare: /usr/local/cuda-8.0/lib64/libcublas.so
new-free-compare/new-free-compare: /usr/local/cuda-8.0/lib64/libcublas_device.a
new-free-compare/new-free-compare: /data/xzjiang/GPU-study/N3LDG/lib/libmatrix.a
new-free-compare/new-free-compare: /data/xzjiang/GPU-study/cnmem/build/libcnmem.so
new-free-compare/new-free-compare: new-free-compare/CMakeFiles/new-free-compare.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/xzjiang/GPU-study/N3LDGTest1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable new-free-compare"
	cd /data/xzjiang/GPU-study/N3LDGTest1/build/new-free-compare && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/new-free-compare.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
new-free-compare/CMakeFiles/new-free-compare.dir/build: new-free-compare/new-free-compare

.PHONY : new-free-compare/CMakeFiles/new-free-compare.dir/build

new-free-compare/CMakeFiles/new-free-compare.dir/requires: new-free-compare/CMakeFiles/new-free-compare.dir/example1.cpp.o.requires

.PHONY : new-free-compare/CMakeFiles/new-free-compare.dir/requires

new-free-compare/CMakeFiles/new-free-compare.dir/clean:
	cd /data/xzjiang/GPU-study/N3LDGTest1/build/new-free-compare && $(CMAKE_COMMAND) -P CMakeFiles/new-free-compare.dir/cmake_clean.cmake
.PHONY : new-free-compare/CMakeFiles/new-free-compare.dir/clean

new-free-compare/CMakeFiles/new-free-compare.dir/depend:
	cd /data/xzjiang/GPU-study/N3LDGTest1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/xzjiang/GPU-study/N3LDGTest1 /data/xzjiang/GPU-study/N3LDGTest1/new-free-compare /data/xzjiang/GPU-study/N3LDGTest1/build /data/xzjiang/GPU-study/N3LDGTest1/build/new-free-compare /data/xzjiang/GPU-study/N3LDGTest1/build/new-free-compare/CMakeFiles/new-free-compare.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : new-free-compare/CMakeFiles/new-free-compare.dir/depend

