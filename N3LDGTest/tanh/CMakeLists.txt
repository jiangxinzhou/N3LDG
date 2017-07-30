aux_source_directory(. DIR_SRCS)
cuda_add_executable(tanh ${DIR_SRCS}) 
cuda_add_cublas_to_target(tanh)
target_link_libraries(tanh ${LIBS})
