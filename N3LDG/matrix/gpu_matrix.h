#ifndef _gpu_matrix_
#define _gpu_matrix_



#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>


#include<thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "include/cnmem.h"
#include "cpu_matrix.h"
#include "functors.h"
#include <assert.h>


#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <vector>

using namespace std;

#define CCE(x) checkCudaErrors(x)

const int THREADS = 128;

class cpu_matrix;

class CUBLAS_HANDLE{
private:
	cublasHandle_t handle;
	CUBLAS_HANDLE() { CCE(cublasCreate(&handle)); }
	~CUBLAS_HANDLE() { CCE(cublasDestroy(handle)); }
public:
	static cublasHandle_t& getInstance(){
		static CUBLAS_HANDLE H;
		return H.handle;
	}
}; 

void InitGPU(cnmemDevice_t &device, size_t mem_size = 1024, int device_id = 0);
void FinalizeGPU();


class DEVICE{
public:
	cnmemDevice_t device;
public:
	//static void initial() { device.numStreams = 0;device.streams = NULL; device.streamSizes = NULL; }
	static cnmemDevice_t& getInstance(){
		static DEVICE D;
		return D.device;
	}
};


struct prg
{
    dtype a, b, dp_val;

    __host__ __device__
    prg(dtype _a=0.0, dtype _b=1.0, dtype _dp_val=0.0) : a(_a), b(_b), dp_val(_dp_val) {};

	
    __host__ __device__ inline
        dtype operator()(const unsigned int n) const
        {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<double> dist(a, b);
            rng.discard(n);
			
			if(dist(rng) <= dp_val)
				return 0;
			else 
				return 1;
        }
};

class gpu_matrix
{
// public:
	// Device *dev;
public:
	dtype *v;
	int row, col, size;
public:
	gpu_matrix();
	~gpu_matrix();
	void init(int r, int c);
	gpu_matrix(dtype* v_data, size_t r, size_t c);
	void delloc();
	void resize(int r, int c);
	// void delloc() { cudaFree(v); }
	inline void zero() { if(v) cudaMemset((void*)v, 0, size * sizeof(dtype)); }
	void zeros();
	void ones();
	void random(dtype bound);//it is not implemented
	void assign(int icol, int jrow, dtype value){
		CCE(cudaMemcpy(v + icol*row + jrow, &value, sizeof(dtype), cudaMemcpyHostToDevice));
	}
	dtype get(int icol, int jrow){
		dtype r;
		CCE(cudaMemcpy(&r, v + icol*row + jrow, sizeof(dtype), cudaMemcpyDeviceToHost));
		return r;
	}
	gpu_matrix& operator=(const gpu_matrix &rhs);
	gpu_matrix& operator=(const cpu_matrix &rhs);
	inline dtype* operator[](const int icol){ return v + icol*row; }
	inline const dtype* operator[](const int icol)const{ return v+icol*row; }
	void transpose(const gpu_matrix &rhs); 
	void transpose();
	void add(const gpu_matrix &a, const gpu_matrix &b);
	void sub(const gpu_matrix &a, const gpu_matrix &b);
	void multiply(const gpu_matrix &a, const gpu_matrix &b);
	void divide(const gpu_matrix &a, const gpu_matrix &b);
	void product(const gpu_matrix &a, const gpu_matrix &b);
	void product(dtype alpha, dtype beta, bool aTranspose, bool bTranspose, const gpu_matrix &a, const gpu_matrix &b);
	void self_add(const gpu_matrix &rhs);
	void self_sub(const gpu_matrix &rhs);
	void self_multiply(const gpu_matrix &rhs);
	void self_divide(const gpu_matrix &rhs);
	void tanh(const gpu_matrix &rhs);
	void sigmoid(const gpu_matrix &rhs);
	void relu(const gpu_matrix &rhs);
	void leaky_relu(const gpu_matrix &rhs);
	void exp(const gpu_matrix &rhs);
	void square(const gpu_matrix &rhs);
	void cube(const gpu_matrix &rhs);
	void dtanh(const gpu_matrix &a, const gpu_matrix &b);
	void dsigmoid(const gpu_matrix &a, const gpu_matrix &b);
	void drelu(const gpu_matrix &a, const gpu_matrix &b);
	void dleaky_relu(const gpu_matrix &a, const gpu_matrix &b);
	void dexp(const gpu_matrix &a, const gpu_matrix &b);
	void dsquare(const gpu_matrix &a, const gpu_matrix &b);
	void dcube(const gpu_matrix &a, const gpu_matrix &b);
	void activate(const gpu_matrix &rhs, FUNC_TYPE functor);
	void dactivate(const gpu_matrix &a, const gpu_matrix &b, DFUNC_TYPE functor);
	// void max_pooling(const gpu_matrix &rhs);
	// void min_pooling(const gpu_matrix &rhs);
	// void average_pooling(const gpu_matrix &rhs);
	
	void dropout(const gpu_matrix &rhs, gpu_matrix &drop_mask, double drop_value, int seed);
	void assign(dtype scale);
	void lookup(const gpu_matrix &E, int idx){
		assert(E.row == size);
		CCE(cudaMemcpy(v, E[idx], sizeof(dtype) * size, cudaMemcpyDeviceToDevice));
	}
	void display(){
		dtype *tv = new dtype[size];
		CCE(cudaMemcpy(tv, v, size * sizeof(dtype), cudaMemcpyDeviceToHost));
		for(int i=0; i<row; i++){
			for(int j=0; j<col; j++){
				std::cout << tv[j*row + i] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	};
	
	void concat(const vector<gpu_matrix> &rhs_vec){
		assert(col == rhs_vec.size());
		assert(row == rhs_vec[0].size);
		for(int i=0; i<col; i++){
			CCE(cudaMemcpy(v + i*row, rhs_vec[i].v, sizeof(dtype) * row, cudaMemcpyDeviceToDevice));
		}
	}
};

void max_pooling_helper(vector<gpu_matrix> &ins, vector<gpu_matrix> &mask);
void min_pooling_helper(vector<gpu_matrix> &ins, vector<gpu_matrix> &mask);

// template <typename T>
// gpu_matrix<T> gpu_matrix<T>::operator * (const device_matrix<T>& rhs) const
// {
	// gpu_matrix<T> res(row, col, 0);
// #if USE_FLOAT
	// CCE(cublasSgemm(,,))
// #else
	// dtype 
	// CCE(cublasDgemm(d->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, ,,, ));
// #endif
	
// }

#endif
