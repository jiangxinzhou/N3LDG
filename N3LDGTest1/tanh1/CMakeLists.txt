aux_source_directory(. DIR_SRCS)
cuda_add_executable(tanh1 ${DIR_SRCS}) 
cuda_add_cublas_to_target(tanh1)
target_link_libraries(tanh1 ${LIBS})
