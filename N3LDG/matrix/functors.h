#ifndef __FUNCTORS__
#define __FUNCTORS__

#if USE_FLOAT
typedef  float dtype;
#else
typedef  double dtype;
#endif

enum FUNC_TYPE { Tanh_type = 0, Sigmoid_type = 1, Relu_type = 2};
enum DFUNC_TYPE { dTanh_type = 0, dSigmoid_type	= 1, dRelu_type = 2};

struct Activate{	
	FUNC_TYPE type;
	Activate(FUNC_TYPE t) : type(t) {}
	__host__ __device__ inline dtype operator()(const dtype& x) const { 
		if(type == Tanh_type){
			return tanh(x); 
		}
		if(type == Sigmoid_type){
			return 1.0 / (1.0 + exp(-x)); ; 
		}
		if(type == Relu_type){
			if (x <= 0) return 0;
			return x;
		}	
	} 
};

struct dActivate{
	DFUNC_TYPE type;
	dActivate(DFUNC_TYPE t) : type(t) {}
	__host__ __device__ inline dtype operator()(const dtype& x, const dtype& y) const { 
		if(type == dTanh_type){
			return (1 + y) * (1 - y); 
		}
		if(type == dSigmoid_type){
			return  (1 - y) * y; 
		}
		if(type == dRelu_type){
			if (x <= 0) return 0;
			return 1;
		}	
	} 
};

struct Assign{
	dtype _a;
	Assign(dtype a) : _a(a) {}
	__host__ __device__ inline dtype operator()(const dtype &x) const {
		return _a;
	}
};


struct Tanh{ 
__host__ __device__ inline dtype operator()(const dtype& x) const { 
		return tanh(x); 
	} 
};

struct dTanh{ 
__host__ __device__ inline dtype operator()(const dtype& x, const dtype& y) const { 
		return (1 + y) * (1 - y); 
	} 
};

struct Sigmoid{ 
__host__ __device__ inline dtype operator()(const dtype& x) const { 
		return 1.0 / (1.0 + exp(-x)); 
	} 
};

struct dSigmoid{ 
__host__ __device__ inline dtype operator()(const dtype& x, const dtype& y) const { 
		return  (1 - y) * y; 
	} 
};

struct Relu{ 
__host__ __device__ inline dtype operator()(const dtype& x) const { 
		if (x <= 0) return 0;
		return x;
	} 
};

struct dRelu{ 
__host__ __device__ inline dtype operator()(const dtype& x, const dtype& y) const { 
		if (x <= 0) return 0;
		return 1;
	} 
};

struct Leaky_relu{ 
__host__ __device__ inline dtype operator()(const dtype& x) const { 
		if (x < 0) return (0.1*x);
		return x;
	} 
};

struct dLeaky_relu{ 
__host__ __device__ inline dtype operator()(const dtype& x, const dtype& y) const { 
		if (x < 0) return 0.1;
		return 1;
	} 
};

struct Exp{ 
__host__ __device__ inline dtype operator()(const dtype& x) const { 
		return exp(x);
	} 
};

struct dExp{ 
__host__ __device__ inline dtype operator()(const dtype& x, const dtype& y) const { 
		return y;
	} 
};

struct Square{ 
__host__ __device__ inline dtype operator()(const dtype& x) const { 
		return x*x; 
	} 
};

struct dSquare{ 
__host__ __device__ inline dtype operator()(const dtype& x, const dtype& y) const { 
		return 2*x; 
	} 
};

struct Cube{ 
__host__ __device__ inline dtype operator()(const dtype& x) const { 
		return x*x*x; 
	} 
};

struct dCube{ 
__host__ __device__ inline dtype operator()(const dtype& x, const dtype& y) const { 
		return 3*x*x; 
	} 
};



#endif
