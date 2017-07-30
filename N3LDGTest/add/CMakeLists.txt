aux_source_directory(. DIR_SRCS)
cuda_add_executable(add ${DIR_SRCS}) 
cuda_add_cublas_to_target(add)
target_link_libraries(add ${LIBS})
