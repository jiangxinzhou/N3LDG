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
include add/CMakeFiles/add.dir/depend.make

# Include the progress variables for this target.
include add/CMakeFiles/add.dir/progress.make

# Include the compile flags for this target's objects.
include add/CMakeFiles/add.dir/flags.make

add/CMakeFiles/add.dir/example2.cpp.o: add/CMakeFiles/add.dir/flags.make
add/CMakeFiles/add.dir/example2.cpp.o: ../add/example2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/xzjiang/GPU-study/N3LDGTest1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object add/CMakeFiles/add.dir/example2.cpp.o"
	cd /data/xzjiang/GPU-study/N3LDGTest1/build/add && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/add.dir/example2.cpp.o -c /data/xzjiang/GPU-study/N3LDGTest1/add/example2.cpp

add/CMakeFiles/add.dir/example2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/add.dir/example2.cpp.i"
	cd /data/xzjiang/GPU-study/N3LDGTest1/build/add && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/xzjiang/GPU-study/N3LDGTest1/add/example2.cpp > CMakeFiles/add.dir/example2.cpp.i

add/CMakeFiles/add.dir/example2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/add.dir/example2.cpp.s"
	cd /data/xzjiang/GPU-study/N3LDGTest1/build/add && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/xzjiang/GPU-study/N3LDGTest1/add/example2.cpp -o CMakeFiles/add.dir/example2.cpp.s

add/CMakeFiles/add.dir/example2.cpp.o.requires:

.PHONY : add/CMakeFiles/add.dir/example2.cpp.o.requires

add/CMakeFiles/add.dir/example2.cpp.o.provides: add/CMakeFiles/add.dir/example2.cpp.o.requires
	$(MAKE) -f add/CMakeFiles/add.dir/build.make add/CMakeFiles/add.dir/example2.cpp.o.provides.build
.PHONY : add/CMakeFiles/add.dir/example2.cpp.o.provides

add/CMakeFiles/add.dir/example2.cpp.o.provides.build: add/CMakeFiles/add.dir/example2.cpp.o


# Object files for target add
add_OBJECTS = \
"CMakeFiles/add.dir/example2.cpp.o"

# External object files for target add
add_EXTERNAL_OBJECTS =

add/add: add/CMakeFiles/add.dir/example2.cpp.o
add/add: add/CMakeFiles/add.dir/build.make
add/add: /usr/local/cuda-8.0/lib64/libcudart_static.a
add/add: /usr/lib/x86_64-linux-gnu/librt.so
add/add: /usr/local/cuda-8.0/lib64/libcublas.so
add/add: /usr/local/cuda-8.0/lib64/libcublas_device.a
add/add: /data/xzjiang/GPU-study/N3LDG/lib/libmatrix.a
add/add: /usr/local/cuda-8.0/lib64/libcudart_static.a
add/add: /usr/lib/x86_64-linux-gnu/librt.so
add/add: /usr/local/cuda-8.0/lib64/libcublas.so
add/add: /usr/local/cuda-8.0/lib64/libcublas_device.a
add/add: /data/xzjiang/GPU-study/N3LDG/lib/libmatrix.a
add/add: add/CMakeFiles/add.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/xzjiang/GPU-study/N3LDGTest1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable add"
	cd /data/xzjiang/GPU-study/N3LDGTest1/build/add && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/add.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
add/CMakeFiles/add.dir/build: add/add

.PHONY : add/CMakeFiles/add.dir/build

add/CMakeFiles/add.dir/requires: add/CMakeFiles/add.dir/example2.cpp.o.requires

.PHONY : add/CMakeFiles/add.dir/requires

add/CMakeFiles/add.dir/clean:
	cd /data/xzjiang/GPU-study/N3LDGTest1/build/add && $(CMAKE_COMMAND) -P CMakeFiles/add.dir/cmake_clean.cmake
.PHONY : add/CMakeFiles/add.dir/clean

add/CMakeFiles/add.dir/depend:
	cd /data/xzjiang/GPU-study/N3LDGTest1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/xzjiang/GPU-study/N3LDGTest1 /data/xzjiang/GPU-study/N3LDGTest1/add /data/xzjiang/GPU-study/N3LDGTest1/build /data/xzjiang/GPU-study/N3LDGTest1/build/add /data/xzjiang/GPU-study/N3LDGTest1/build/add/CMakeFiles/add.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : add/CMakeFiles/add.dir/depend

