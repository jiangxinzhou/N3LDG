cuda_add_executable(new-free-compare1 example1.cu) 
cuda_add_cublas_to_target(new-free-compare1)
target_link_libraries(new-free-compare1 ${LIBS})
