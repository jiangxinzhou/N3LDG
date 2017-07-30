cuda_add_library(matrix gpu_matrix.cu cpu_matrix.cc)
cuda_add_cublas_to_target(matrix)
target_link_libraries(matrix ${LIBS})
