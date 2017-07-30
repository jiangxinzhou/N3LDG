#include "gpu_matrix.h"
#include <iostream>

__global__ inline void naiveMatrixTranspose(dtype *odata, const dtype *idata, const int rows, const int cols) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < cols && y < rows)
    odata[x*rows + y] = idata[y*cols+ x];
}

void gpu_matrix::transpose(const gpu_matrix &rhs) {
	resize(rhs.col, rhs.row);

	dim3 grid;
	grid.x = (unsigned int) ceil((float) col / 32);
	grid.y = (unsigned int) ceil((float) row / 32);
	dim3 threads(32, 32);
	naiveMatrixTranspose<<<grid, threads>>>(v, rhs.v, row, col);
}

void gpu_matrix::transpose(){
	gpu_matrix rhs;
	rhs = *this;
	this->transpose(rhs);
}	
	
gpu_matrix::~gpu_matrix(){
	if(v){
		cudaFree(v);
	}
	v = NULL;
	row = 0;
	col = 0;
	size = 0;
}

void gpu_matrix::init(int r, int c){
	row = r;
	col = c;
	size = row * col;
	if(size != 0){
		CCE(cudaMalloc((void**)&v, sizeof(dtype) * size));
		zero();
	}
} 

gpu_matrix::gpu_matrix():row(0), col(0), size(0), v(NULL) {}

gpu_matrix::gpu_matrix(dtype* v_data, size_t r, size_t c){
  init(r, c);
  CCE(cudaMemcpy(v, v_data, sizeof(dtype) * row * col, cudaMemcpyHostToDevice));
}

void gpu_matrix::resize(int r, int c)
{
	if(row == r && col == c)
		return;
	
	if(v){
		cudaFree(v);
	}

	init(r, c);
}

void gpu_matrix::zeros(){
	CCE(cudaMemset((void*)v, 0, sizeof(dtype) * size));
}

void gpu_matrix::ones(){
	dtype one = 1.0;
	for(int i=0; i<size; i++){
		CCE(cudaMemcpy((v+i), &one, sizeof(dtype), cudaMemcpyHostToDevice));
	}
	
}

gpu_matrix& gpu_matrix::operator=(const gpu_matrix &rhs){
	resize(rhs.row, rhs.col);
	CCE(cudaMemcpy(v, rhs.v, row * col * sizeof(dtype), cudaMemcpyDeviceToDevice));
	return *this;
}

gpu_matrix& gpu_matrix::operator=(const cpu_matrix &rhs){
	resize(rhs.row, rhs.col);
	CCE(cudaMemcpy(v, rhs.v, row * col * sizeof(dtype), cudaMemcpyHostToDevice));
	return *this;
}

void gpu_matrix::add(const gpu_matrix &a, const gpu_matrix &b){
	thrust::device_ptr<dtype> ptr0(v);
	thrust::device_ptr<dtype> ptr1(a.v);
	thrust::device_ptr<dtype> ptr2(b.v);
	thrust::transform(ptr1, ptr1 + row * col, ptr2, ptr0, thrust::plus<dtype>());
}

void gpu_matrix::sub(const gpu_matrix &a, const gpu_matrix &b){
	thrust::device_ptr<dtype> ptr0(v);
	thrust::device_ptr<dtype> ptr1(a.v);
	thrust::device_ptr<dtype> ptr2(b.v);
	thrust::transform(ptr1, ptr1 + row * col, ptr2, ptr0, thrust::minus<dtype>());
}

void gpu_matrix::multiply(const gpu_matrix &a, const gpu_matrix &b){
	thrust::device_ptr<dtype> ptr0(v);
	thrust::device_ptr<dtype> ptr1(a.v);
	thrust::device_ptr<dtype> ptr2(b.v);
	thrust::transform(ptr1, ptr1 + row * col, ptr2, ptr0, thrust::multiplies<dtype>());
}

void gpu_matrix::divide(const gpu_matrix &a, const gpu_matrix &b){
	thrust::device_ptr<dtype> ptr0(v);
	thrust::device_ptr<dtype> ptr1(a.v);
	thrust::device_ptr<dtype> ptr2(b.v);
	thrust::transform(ptr1, ptr1 + row * col, ptr2, ptr0, thrust::divides<dtype>());
}

void gpu_matrix::self_add(const gpu_matrix &rhs){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(rhs.v);
	thrust::transform(ptr_a, ptr_a + row * col, ptr_b, ptr_a, thrust::plus<dtype>());
}

void gpu_matrix::self_sub(const gpu_matrix &rhs){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(rhs.v);
	thrust::transform(ptr_a, ptr_a + row * col, ptr_b, ptr_a, thrust::minus<dtype>());
}

void gpu_matrix::self_multiply(const gpu_matrix &rhs){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(rhs.v);
	thrust::transform(ptr_a, ptr_a + row * col, ptr_b, ptr_a, thrust::multiplies<dtype>());
}

void gpu_matrix::self_divide(const gpu_matrix &rhs){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(rhs.v);
	thrust::transform(ptr_a, ptr_a + row * col, ptr_b, ptr_a, thrust::divides<dtype>());
}

void gpu_matrix::product(const gpu_matrix &a, const gpu_matrix &b){
	int m = row;
	int n = col;
	int k = a.col;
	int lda = a.row;
	int ldb = b.row;
	int ldc = row;
	dtype alpha = 1.0;
	dtype beta = 0.0;

#if USE_FLOAT
	CCE(cublasSgemm(CUBLAS_HANDLE::getInstance(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a.v, lda, b.v, ldb, &beta, v, ldc));
#else
	CCE(cublasDgemm(CUBLAS_HANDLE::getInstance(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a.v, lda, b.v, ldb, &beta, v, ldc));
#endif
}

void gpu_matrix::tanh(const gpu_matrix &rhs){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(rhs.v);
	thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Tanh());
}

void gpu_matrix::sigmoid(const gpu_matrix &rhs){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(rhs.v);
	thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Sigmoid());
}

void gpu_matrix::relu(const gpu_matrix &rhs){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(rhs.v);
	thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Relu());
}

void gpu_matrix::leaky_relu(const gpu_matrix &rhs){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(rhs.v);	
	thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Leaky_relu());
}

void gpu_matrix::exp(const gpu_matrix &rhs){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(rhs.v);	
	thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Exp());
}

void gpu_matrix::square(const gpu_matrix &rhs){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(rhs.v);	
	thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Square());
}

void gpu_matrix::cube(const gpu_matrix &rhs){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(rhs.v);	
	thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Cube());
}

void gpu_matrix::activate(const gpu_matrix &rhs, FUNC_TYPE functor){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(rhs.v);	
	thrust::transform(ptr_b, ptr_b + row * col, ptr_a, Activate(functor));
}

void gpu_matrix::dactivate(const gpu_matrix &a, const gpu_matrix &b, DFUNC_TYPE functor){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(a.v);	
	thrust::device_ptr<dtype> ptr_c(b.v);	
	thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dActivate(functor));
}

void gpu_matrix::dtanh(const gpu_matrix &a, const gpu_matrix &b){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(a.v);	
	thrust::device_ptr<dtype> ptr_c(b.v);
	thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dTanh());
}

void gpu_matrix::dsigmoid(const gpu_matrix &a, const gpu_matrix &b){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(a.v);	
	thrust::device_ptr<dtype> ptr_c(b.v);
	thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dSigmoid());
}

void gpu_matrix::drelu(const gpu_matrix &a, const gpu_matrix &b){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(a.v);	
	thrust::device_ptr<dtype> ptr_c(b.v);
	thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dRelu());
}

void gpu_matrix::dleaky_relu(const gpu_matrix &a, const gpu_matrix &b){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(a.v);	
	thrust::device_ptr<dtype> ptr_c(b.v);
	thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dLeaky_relu());
}

void gpu_matrix::dexp(const gpu_matrix &a, const gpu_matrix &b){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(a.v);	
	thrust::device_ptr<dtype> ptr_c(b.v);
	thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dExp());
}

void gpu_matrix::dsquare(const gpu_matrix &a, const gpu_matrix &b){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(a.v);	
	thrust::device_ptr<dtype> ptr_c(b.v);
	thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dSquare());
}

void gpu_matrix::dcube(const gpu_matrix &a, const gpu_matrix &b){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::device_ptr<dtype> ptr_b(a.v);	
	thrust::device_ptr<dtype> ptr_c(b.v);
	thrust::transform(ptr_b, ptr_b + row * col, ptr_c, ptr_a, dCube());
}
