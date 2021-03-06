project(N3LDG_TEST)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

#set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(N3LDG_INCLUDE_DIR ${N3L_DIR}/core)
set(EIGEN_INCLUDE_DIR ${N3L_DIR}/eigen)
set(MATRIX_INCLUDE_DIR ${N3L_DIR}/matrix)
set(LIBS ${LIBS} ${N3L_DIR}/lib/libmatrix.a)
set(LIBS ${LIBS} ${CNMEM_DIR}/build/libcnmem.so)

include_directories(/usr/local/cuda-8.0/samples/common/inc)
include_directories(${N3LDG_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${MATRIX_INCLUDE_DIR})
include_directories(${MATRIX_INCLUDE_DIR})
include_directories(${CNMEM_DIR})

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W1 /MP")   # -Wall produces 20k warnings. Enable parallel compilation
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -funroll-loops -fno-finite-math-only -Wall -Wno-missing-braces -std=c++11 -Ofast -g -march=native")
endif()


find_package(CUDA REQUIRED)
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
message("CUDA_INCLUDE: " ${CUDA_INCLUDE_DIRS})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
#list(APPEND CUDA_LIBRARIES /usr/lib64/libpthread.so)
MESSAGE("CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
list(REMOVE_ITEM CUDA_LIBRARIES -lpthread)
set(LIBS ${LIBS} ${CUDA_LIBRARIES})
FIND_PACKAGE(Threads REQUIRED) 
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT}) 
# set(LIBS ${LIBS} /usr/local/cuda-8.0/lib64/libcublas.so.8.0)
message("LIBS: " ${LIBS})

#add_subdirectory(product)
#add_subdirectory(add)
#add_subdirectory(add1)
#add_subdirectory(multiply)
#add_subdirectory(multiply1)
#add_subdirectory(tanh)
#add_subdirectory(tanh1)
#add_subdirectory(copy-compare)
add_subdirectory(new-free-compare)

