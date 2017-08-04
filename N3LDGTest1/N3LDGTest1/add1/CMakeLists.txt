aux_source_directory(. DIR_SRCS)
cuda_add_executable(add1 ${DIR_SRCS}) 
cuda_add_cublas_to_target(add1)
target_link_libraries(add1 ${LIBS})
