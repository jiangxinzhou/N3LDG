#include "gpu_matrix.h"



// inline int find_gpu_has_most_free_space(){
	// int nDevices;
	// int device_id_max_free_space = 0;
	// size_t mem_free;
	// size_t mem_free_max = -1;
	// size_t mem_total;
	// cudaGetDeviceCount(&nDevices);
	// for(int i=0; i < nDevices; i++) {
		// cudaSetDevice(i);
		// cudaMemGetInfo(&mem_free, &mem_total);
		// // if(mem_free_max < mem_free){
			// // device_id_max_free_space = i;
			// // mem_free_max = mem_free;
		// // }
	// }
	
	// return device_id_max_free_space;
// }

void InitGPU(cnmemDevice_t &device, size_t mem_size, int device_id)
{	
	memset(&device, 0, sizeof(device));
	device.device = device_id;
	device.size = mem_size;
	cudaSetDevice(device_id);
	assert(CNMEM_STATUS_SUCCESS == cnmemInit(1, &device, CNMEM_FLAGS_CANNOT_GROW));
	cudaSetDevice(device_id);
}

void FinalizeGPU()
{
	assert(CNMEM_STATUS_SUCCESS == cnmemFinalize());
}


__global__ inline void naiveMatrixTranspose(dtype *odata, const dtype *idata, const int rows, const int cols) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < cols && y < rows)
    odata[x*rows + y] = idata[y*cols+ x];
}

// __global__ inline void max_pooling_kernel(dtype *src, dtype *target, int row, int n){
	// int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	// target[tid] = src[tid*row];
	// if(tid < n){
		// for(int i=tid*row+1; i<tid*row+row; i++){
			// target[tid] = (target[tid] >= src[i]) ? target[tid] : src[i];
		// }
	// }
// }

// __global__ void min_pooling_kernel(dtype *src, dtype *target, int row, int n){
	// int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	// target[tid] = src[tid*row];
	// if(tid < n){
		// for(int i=tid*row+1; i<tid*row+row; i++){
			// target[tid] = (target[tid] <= src[i]) ? target[tid] : src[i];
		// }
	// }
// }

// __global__ void average_pooling_kernel(dtype *src, dtype *target, int row, int n){
	// int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	// target[tid] = 0;
	// if(tid < n){
		// for(int i=tid*row; i<tid*row+row; i++){
			// target[tid] += src[i];
		// }
	// }
	// target[tid] /= row;
// }


// void gpu_matrix::max_pooling(const gpu_matrix &rhs){
	// max_pooling_kernel<<<(rhs.col + THREADS - 1)/THREADS, THREADS>>>(rhs.v, v, rhs.row, rhs.size);
// }

// void gpu_matrix::min_pooling(const gpu_matrix &rhs){
	// min_pooling_kernel<<<(rhs.col + THREADS - 1)/THREADS, THREADS>>>(rhs.v, v, rhs.row, rhs.size);
// }

// void gpu_matrix::average_pooling(const gpu_matrix &rhs){
	// average_pooling_kernel<<<(rhs.col + THREADS - 1)/THREADS, THREADS>>>(rhs.v, v, rhs.row, rhs.size);
// }

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
	row = 0;
	col = 0;
	size = 0;
}

void gpu_matrix::delloc(){
	if(v){
		assert(CNMEM_STATUS_SUCCESS == cnmemFree(v, NULL));
	}
	v = NULL;
}

void gpu_matrix::init(int r, int c){
	row = r;
	col = c;
	size = row * col;
	if(size != 0){
		assert(CNMEM_STATUS_SUCCESS == cnmemMalloc((void**)&v, sizeof(dtype) * size, NULL));
		//CCE(cudaMalloc((void**)&v, sizeof(dtype) * size));
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
		assert(CNMEM_STATUS_SUCCESS == cnmemMalloc((void**)&v, sizeof(dtype) * size, NULL));
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
	assert((row == rhs.row) && (col == rhs.col) && (size == rhs.size));
	//resize(rhs.row, rhs.col);
	CCE(cudaMemcpy(v, rhs.v, row * col * sizeof(dtype), cudaMemcpyDeviceToDevice));
	return *this;
}

gpu_matrix& gpu_matrix::operator=(const cpu_matrix &rhs){
	assert((row == rhs.row) && (col == rhs.col) && (size == rhs.size));
	//resize(rhs.row, rhs.col);
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

void gpu_matrix::product(dtype alpha, dtype beta, bool aTranspose, bool bTranspose, const gpu_matrix &a, const gpu_matrix &b){
	int m = row;
	int  n = col;
	int k = aTranspose ? a.row : a.col;
	int lda = a.row;
	int ldb = b.row;
	int ldc = row;
	cublasOperation_t opa = aTranspose ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t opb = bTranspose ? CUBLAS_OP_T : CUBLAS_OP_N;
	
#if USE_FLOAT
	CCE(cublasSgemm(CUBLAS_HANDLE::getInstance(), opa, opb, m, n, k, &alpha, a.v, lda, b.v, ldb, &beta, v, ldc));
#else
	CCE(cublasDgemm(CUBLAS_HANDLE::getInstance(), opa, opb, m, n, k, &alpha, a.v, lda, b.v, ldb, &beta, v, ldc));
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

void gpu_matrix::dropout(const gpu_matrix &rhs, gpu_matrix &drop_mask, double drop_value, int seed){
	thrust::counting_iterator<unsigned int> index_sequence_begin(seed);
	thrust::device_ptr<dtype> ptr(drop_mask.v);
	thrust::transform(index_sequence_begin, index_sequence_begin + rhs.size, ptr, prg(0.0, 1.0, drop_value));
	
	
	multiply(rhs, drop_mask);
}

void gpu_matrix::assign(dtype scale){
	thrust::device_ptr<dtype> ptr_a(v);
	thrust::transform(ptr_a, ptr_a + row * col, ptr_a, Assign(scale));
}


void max_pooling_helper(vector<gpu_matrix> &ins, vector<gpu_matrix> &mask){
	int dim = ins[0].size;
	int size = mask.size();
	vector<cpu_matrix> t_ins;// needn't delloc manually
	
	t_ins.resize(ins.size());
	for(int i=0; i<t_ins.size(); i++){
		t_ins[i].init(ins[i].row, ins[i].col);
		t_ins[i] = ins[i];
	}
	
	
	for(int i=0; i<dim; i++){
		int max_iter = -1;
		for(int j=0; j<size; j++){
			if((max_iter == -1) || (t_ins[j].get(0, i) > t_ins[max_iter].get(0, i))){
				max_iter = j;
			}
		}
		//mask is on gpu
		mask[max_iter].assign(0, i, 1.0);
	}
}

void min_pooling_helper(vector<gpu_matrix> &ins, vector<gpu_matrix> &mask){
	int dim = ins[0].size;
	int size = mask.size();
	vector<cpu_matrix> t_ins;// needn't delloc manually
	
	t_ins.resize(ins.size());
	for(int i=0; i<t_ins.size(); i++){
		t_ins[i].init(ins[i].row, ins[i].col);
		t_ins[i] = ins[i];
	}
	
	
	for(int i=0; i<dim; i++){
		int min_iter = -1;
		for(int j=0; j<size; j++){
			if((min_iter == -1) || (t_ins[j].get(0, i) < t_ins[min_iter].get(0, i))){
				min_iter = j;
			}
		}
		//mask is on gpu
		mask[min_iter].assign(0, i, 1.0);
	}
}