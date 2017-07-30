aux_source_directory(. DIR_SRCS)
cuda_add_executable(product ${DIR_SRCS}) 
cuda_add_cublas_to_target(product)
target_link_libraries(product ${LIBS})
