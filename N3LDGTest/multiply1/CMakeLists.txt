aux_source_directory(. DIR_SRCS)
cuda_add_executable(multiply1 ${DIR_SRCS}) 
cuda_add_cublas_to_target(multiply1)
target_link_libraries(multiply1 ${LIBS})
