#ifndef _Ccpu_matrix_
#define _Ccpu_matrix_

#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
#include "gpu_matrix.h"
#include "functors.h"

class gpu_matrix;

using namespace Eigen;
#if USE_FLOAT
typedef Eigen::Map<MatrixXf> Mat;
typedef Eigen::TensorMap<Eigen::Tensor<float, 1>>  Vec;
#else
typedef Eigen::Map<MatrixXd> Mat;
typedef Eigen::TensorMap<Eigen::Tensor<double, 1>>  Vec;
#endif

class cpu_matrix{
public:
	dtype *v;
	int row, col, size;
public:
	const Mat mat() const {
		return Mat(v, row, col);
	}
	Mat mat(){
		return Mat(v, row, col);
	}
	const Vec vec() const {
		return Vec(v, size);
	}
	Vec vec(){
		return Vec(v, size);
	}
public:
	cpu_matrix();
	~cpu_matrix();
	void init(int r, int c);
	cpu_matrix(dtype* v_data, size_t r, size_t c);
	void delloc();
	void resize(int r, int c);
	inline void zero() { if(v) memset((void*)v, 0, size * sizeof(dtype)); }
	void zeros();
	void ones();
	void random(dtype bound);
	cpu_matrix& operator = (const cpu_matrix &rhs);
	cpu_matrix& operator = (const gpu_matrix &rhs);
	inline dtype* operator[](const int icol){ return v + icol*row; }
	inline const dtype* operator[](const int icol)const{ return v+icol*row; }
	void transpose(const cpu_matrix &rhs);
	void transpose();
	void add(const cpu_matrix &a, const cpu_matrix &b);	
	void sub(const cpu_matrix &a, const cpu_matrix &b);
	void multiply(const cpu_matrix &a, const cpu_matrix &b);
	void divide(const cpu_matrix &a, const cpu_matrix &b);
	void product(const cpu_matrix &a, const cpu_matrix &b);

	//======================================/
	void self_add(const cpu_matrix &rhs);
	void self_sub(const cpu_matrix &rhs);
	void self_multiply(const cpu_matrix &rhs);
	void self_divide(const cpu_matrix &rhs);
	//======================================/
	void tanh(const cpu_matrix &rhs);
	void sigmoid(const cpu_matrix &rhs);
	void relu(const cpu_matrix &rhs);
	void leaky_relu(const cpu_matrix &rhs);
	void square(const cpu_matrix &rhs);
	void cube(const cpu_matrix &rhs);
	void dtanh(const cpu_matrix &a, const cpu_matrix &b);
	void dsigmoid(const cpu_matrix &a, const cpu_matrix &b);
	void drelu(const cpu_matrix &a, const cpu_matrix &b);
	void dleaky_relu(const cpu_matrix &a, const cpu_matrix &b);
	void dsquare(const cpu_matrix &a, const cpu_matrix &b);
	void dcube(const cpu_matrix &a, const cpu_matrix &b);
	void activate(const cpu_matrix &rhs, FUNC_TYPE functor);
	void dactivate(const cpu_matrix &a, const cpu_matrix &b, DFUNC_TYPE functor);
	// template<typename CustomUnaryOp>
	// void unary(const cpu_matrix &rhs, const CustomUnaryOp& op){this->vec() = rhs.vec().unaryExpr(op);} 
	// template<typename CustomBinaryOp>
	// void binary(const cpu_matrix &a, const cpu_matrix &b, const CustomBinaryOp& op)
	// {this->vec() = a.vec().binaryExpr(b.vec(), op);}
};

#endif
