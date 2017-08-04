aux_source_directory(. DIR_SRCS)
cuda_add_executable(copy-compare ${DIR_SRCS}) 
cuda_add_cublas_to_target(copy-compare)
target_link_libraries(copy-compare ${LIBS})
