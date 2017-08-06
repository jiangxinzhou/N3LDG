if(USE_CUDA)
	cuda_add_library(matrix gpu_matrix.cu cpu_matrix.cc)
	cuda_add_cublas_to_target(matrix)
else()
	add_library(matrix cpu_matrix.cc)
endif()

target_link_libraries(matrix ${LIBS})
