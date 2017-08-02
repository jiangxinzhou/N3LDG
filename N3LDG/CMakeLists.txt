project(Matrix)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set (LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(matrix)
include_directories(/usr/local/cuda-8.0/samples/common/inc)
include_directories(${PROJECT_SOURCE_DIR}/eigen)
include_directories(${CNMEM_DIR})


if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W1 /MP")   # -Wall produces 20k warnings. Enable parallel compilation
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -funroll-loops -fno-finite-math-only -Wall -Wno-missing-braces -std=c++11 -Ofast -g -march=native")
endif()

if(USE_GPU)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_GPU")
endif()

if(USE_FLOAT)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_FLOAT")
endif()

if(USE_CUDA)
  find_package(CUDA REQUIRED)
  set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
  message("CUDA_INCLUDE: " ${CUDA_INCLUDE_DIRS})
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  #list(APPEND CUDA_LIBRARIES /usr/lib64/libpthread.so)
  MESSAGE("CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
  list(REMOVE_ITEM CUDA_LIBRARIES -lpthread)
  set(LIBS ${LIBS} ${CUDA_LIBRARIES})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_HAS_CUDA_FP16 -DEIGEN_USE_GPU")
endif()

FIND_PACKAGE(Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

add_subdirectory(matrix)
