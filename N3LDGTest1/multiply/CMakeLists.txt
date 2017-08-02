aux_source_directory(. DIR_SRCS)
cuda_add_executable(multiply ${DIR_SRCS}) 
cuda_add_cublas_to_target(multiply)
target_link_libraries(multiply ${LIBS})
