#include "cpu_matrix.h"
#include <cstring>
#include <iostream>

cpu_matrix::~cpu_matrix(){
	if(v){
		delete[] v;
	}
	v = NULL;
	row = 0;
	col = 0;
	size = 0;
}

void cpu_matrix::init(int r, int c){
	row = r;
	col = c;
	size = row * col;

	if(size != 0){
		v = new dtype[size];
		zero();
	}
}

cpu_matrix::cpu_matrix() : row(0), col(0), size(0), v(NULL){}


cpu_matrix::cpu_matrix(dtype* v_data, size_t r, size_t c){
  init(r, c);
  memcpy(v, v_data, sizeof(dtype) * row * col);
}

void cpu_matrix::resize(int r, int c){
	if(row == r && col == c)
		return;
	
	if(v){
		delete[] v;
	}
	init(r, c);
}

cpu_matrix& cpu_matrix::operator = (const cpu_matrix &rhs){
	resize(rhs.row, rhs.col);
	memcpy(v, rhs.v, sizeof(dtype) * size);
	return *this;
}

cpu_matrix& cpu_matrix::operator = (const gpu_matrix &rhs){
	resize(rhs.row, rhs.col);
	CCE(cudaMemcpy(v, rhs.v, row * col * sizeof(dtype), cudaMemcpyDeviceToHost));
	return *this;
}
//cpu_matrix cpu_matrix::operator + (const cpu_matrix &rhs) const{
//	std::cout << "+" << "\n";
//	cpu_matrix res(row, col);
//	res.mat() = this->mat() + rhs.mat();
//	
//	return res;
//}


void cpu_matrix::zeros(){
	for(int i=0; i<size; i++)
		v[i] = 0.0;
}

void cpu_matrix::ones(){
	for(int i=0; i<size; i++)
		v[i] = 1.0;
}

void cpu_matrix::random(dtype bound){
	 dtype min = -bound, max = bound;
     for (int i = 0; i < size; i++) {
		 v[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
	 }
}

void cpu_matrix::transpose(const cpu_matrix &rhs) {
	this->mat() = rhs.mat().transpose();
}

void cpu_matrix::transpose() { 
	this->mat() = this->mat().transpose(); 
}

void cpu_matrix::add(const cpu_matrix &a, const cpu_matrix &b){
	this->mat() = a.mat() + b.mat();
}

void cpu_matrix::sub(const cpu_matrix &a, const cpu_matrix &b){
	this->mat() = a.mat() - b.mat();
}

void cpu_matrix::multiply(const cpu_matrix &a, const  cpu_matrix &b){
	this->vec() = a.vec() * b.vec();
}

void cpu_matrix::divide(const cpu_matrix &a, const cpu_matrix &b){
	this->vec() = a.vec() / b.vec();
}

void cpu_matrix::product(const cpu_matrix &a, const cpu_matrix &b){
	this->mat() = a.mat() * b.mat();
}


void cpu_matrix::self_add(const cpu_matrix &rhs){
	this->mat() += rhs.mat();
}

void cpu_matrix::self_sub(const cpu_matrix &rhs){
	this->mat() -= rhs.mat();
}

void cpu_matrix::self_multiply(const cpu_matrix &rhs){
	this->vec() *= rhs.vec();
}

void cpu_matrix::self_divide(const cpu_matrix &rhs){
	this->vec() /= rhs.vec();
}

void cpu_matrix::sigmoid(const cpu_matrix &rhs){ 
	this->vec() = rhs.vec().unaryExpr(Sigmoid()); 
}

void cpu_matrix::dsigmoid(const cpu_matrix &a, const cpu_matrix &b){
	this->vec() = a.vec().binaryExpr(b.vec(), dSigmoid()); 
}

void cpu_matrix::relu(const cpu_matrix &rhs){
	this->vec() = rhs.vec().unaryExpr(Relu()); 
}

void cpu_matrix::drelu(const cpu_matrix &a, const cpu_matrix &b){
	this->vec() = a.vec().binaryExpr(b.vec(), dRelu()); 
}

void cpu_matrix::leaky_relu(const cpu_matrix &rhs){
	this->vec() = rhs.vec().unaryExpr(Leaky_relu()); 
}

void cpu_matrix::dleaky_relu(const cpu_matrix &a, const cpu_matrix &b){
	this->vec() = a.vec().binaryExpr(b.vec(), dLeaky_relu()); 
}

void cpu_matrix::tanh(const cpu_matrix &rhs){ 
	this->vec() = rhs.vec().unaryExpr(Tanh()); 
}

void cpu_matrix::dtanh(const cpu_matrix &a, const cpu_matrix &b){
	this->vec() = a.vec().binaryExpr(b.vec(), dTanh());
}

void cpu_matrix::activate(const cpu_matrix &rhs, FUNC_TYPE functor){
	this->vec() = rhs.vec().unaryExpr(Activate(functor));
}

void cpu_matrix::dactivate(const cpu_matrix &a, const cpu_matrix &b, DFUNC_TYPE functor){
	this->vec() = a.vec().binaryExpr(b.vec(), dActivate(functor));
}
