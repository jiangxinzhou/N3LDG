aux_source_directory(. DIR_SRCS)
cuda_add_executable(new-free-compare1 ${DIR_SRCS}) 
cuda_add_cublas_to_target(new-free-compare1)
target_link_libraries(new-free-compare1 ${LIBS})
