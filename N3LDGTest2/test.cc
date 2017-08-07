#include "test.h"

void test_zeros(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu zeros:" << endl;
		cpu_matrix a;
		a.init(Dim1, Dim2);
		a.random(100);
	
		cout << "before" << endl;

		a.display();
	
		a.zeros();
	
		cout << "after" << endl;
		a.display();
	}
#if USE_GPU
	else{
		cout << "this is gpu zeros:" << endl;
		gpu_matrix a;
		a.init(Dim1, Dim2);
		a.random(100);
	
		cout << "before" << endl;
		a.display();
	
		a.zeros();
	
		cout << "after" << endl;
		a.display();
	}
#endif
}

void test_ones(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu ones:" << endl;
		cpu_matrix a;
		a.init(Dim1, Dim2);
		a.random(100);
		
		cout << "before" << endl;
		a.display();
	
		a.ones();
	
		cout << "after" << endl;
		a.display();
	}
#if USE_GPU
	else{
		cout << "this is gpu ones:" << endl;
		gpu_matrix a;
		a.init(Dim1, Dim2);
		a.random(100);
	
		cout << "before" << endl;
		a.display();
	
		a.ones();
	
		cout << "after" << endl;
		a.display();
	}
#endif
}

void test_assign(int icol, int jrow, dtype value, bool is_cpu){
	if(is_cpu){
		cout << "this is cpu assign:" << endl;
		cout << "icol:" << icol << " " << "jrow:" << jrow << " " << "value:" << value << endl;
		cpu_matrix a;
		a.init(Dim1, Dim2);
		a.random(100);
	
		cout << "before" << endl;
		a.display();
	
		a.assign(icol, jrow, value);
	
		cout << "after" << endl;
		a.display();
	}
#if USE_GPU
	else{
		cout << "this is gpu assign:" << endl;
		cout << "icol:" << icol << " " << "jrow:" << jrow << " " << "value:" << value << endl;
		gpu_matrix a;
		a.init(Dim1, Dim2);
		a.random(100);
		
		cout << "before" << endl;
		a.display();
	
		a.assign(icol, jrow, value);
	
		cout << "after" << endl;
		a.display();
	}
#endif
}

void test_get(int icol, int jrow, bool is_cpu){
	if(is_cpu){
		cout << "this is cpu get:" << endl;
		cout << "icol:" << icol << " " << "jrow:" << jrow << endl;
		cpu_matrix a;
		a.init(Dim1, Dim2);
		a.random(100);

		cout << "before" << endl;
		a.display();

		dtype value = a.get(icol, jrow);

		cout << "after" << endl;
		cout << "value" << value << endl;
	}
#if USE_GPU
	else{
		cout << "this is gpu get:" << endl;
		cout << "icol:" << icol << " " << "jrow:" << jrow << endl;
		gpu_matrix a;
		a.init(Dim1, Dim2);
		a.random(100);
		cout << "before" << endl;
		a.display();
	
		dtype value = a.get(icol, jrow);
	
		cout << "after" << endl;
		cout << "value" << value << endl;
	}
#endif
}

void test_op_equal(){
#if USE_GPU
	if(true){
		cout << "this is cpu_matrix = gpu_matrix:" << endl;
		cpu_matrix a;
		gpu_matrix b;
		a.init(Dim1, Dim2);
		a.random(100);
		b.init(Dim1, Dim2);
		b.random(100);

		cout << "before" << endl;
		cout << "cpu_matrix: \n" << endl; a.display();
		cout << "gpu_matrix: \n" << endl; b.display();	
		

		cout << "after" << endl;
		a = b;
		cout << "cpu_matrix: \n" << endl; a.display();
	}
	
	
	if(true){
		cout << "this is gpu_matrix = cpu_matrix:" << endl;
		gpu_matrix a;
		cpu_matrix b;
		a.init(Dim1, Dim2);
		a.random(100);
		b.init(Dim1, Dim2);
		b.random(100);
		
		cout << "before" << endl;
		cout << "gpu_matrix: \n" << endl; a.display();
		cout << "cpu_matrix: \n" << endl; b.display();	
	
		cout << "after" << endl;
		a = b;
		cout << "gpu_matrix: \n" << endl; a.display();
	}
#endif
	
	if(true){
		cout << "this is cpu_matrix = cpu_matrix:" << endl;
		cpu_matrix a;
		cpu_matrix b;
		a.init(Dim1, Dim2);
		a.random(100);
		b.init(Dim1, Dim2);
		b.random(100);

		cout << "before" << endl;
		cout << "cpu_matrix a: \n" << endl; a.display();
		cout << "cpu_matrix b: \n" << endl; b.display();	
		

		cout << "after" << endl;
		a = b;
		cout << "cpu_matrix b: \n" << endl; a.display();
	}
	
#if USE_GPU
	if(true){
		cout << "this is gpu_matrix = gpu_matrix:" << endl;
		gpu_matrix a;
		gpu_matrix b;
		a.init(Dim1, Dim2);
		a.random(100);
		b.init(Dim1, Dim2);
		b.random(100);

		cout << "before" << endl;
		cout << "gpu_matrix a: \n" << endl; a.display();
		cout << "gpu_matrix b: \n" << endl; b.display();	
		

		cout << "after" << endl;
		a = b;
		cout << "gpu_matrix b: \n" << endl; a.display();
	}
#endif
}

void test_random(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_random:" << endl;
		cpu_matrix a;
		a.init(Dim1, Dim2);
		a.random(100);
		a.display();
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_random:" << endl;
		gpu_matrix a;
		a.init(Dim1, Dim2);
		a.random(100);
		a.display();
	}	
#endif
}

// void test_self_transpose(bool is_cpu){
	// if(is_cpu){
		// cout << "this is cpu_self_transpose:" << endl;
		// cpu_matrix a;
		// a.init(Dim1, Dim2);
		// a.random(100);
		
		// cout << "before" << endl;
		// a.display();
		
		// a.transpose();
		// cout << "after" << endl;
		// a.display();
		
	// }
	// else
	// {
		// cout << "this is gpu_transpose:" << endl;
		// gpu_matrix a;
		// a.init(Dim1, Dim2);
		// a.random(100);
		
		// cout << "before" << endl;
		// a.display();
		
		// a.transpose();
		// cout << "after" << endl;
		// a.display();
	// }
// }


void test_transpose(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_transpose:" << endl;
		cpu_matrix a;
		cpu_matrix b;
		a.init(Dim1, Dim2);
		b.init(Dim2, Dim1);
		a.random(100);
		
		cout << "before cpu a" << endl;
		a.display();
		
		b.transpose(a);
		cout << "after cpu b" << endl;
		b.display();
		
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_transpose:" << endl;
		gpu_matrix a;
		gpu_matrix b;
		a.init(Dim1, Dim2);
		b.init(Dim2, Dim1);
		a.random(100);
		
		cout << "before gpu a" << endl;
		a.display();
		
		b.transpose(a);
		cout << "after gpu b" << endl;
		b.display();
	}
#endif
}

void test_product(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_product:" << endl;
		cpu_matrix a;
		cpu_matrix b;
		cpu_matrix c;
		a.init(Dim1, Dim2);
		b.init(Dim2, Dim3);
		c.init(Dim1, Dim3);
		a.random(100);
		b.random(100);
		
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout << "b" << endl; b.display();
		
		c.product(a, b);
		cout << "after" << endl;
		cout <<  "c" << endl; c.display();	
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_product:" << endl;
		gpu_matrix a;
		gpu_matrix b;
		gpu_matrix c;
		a.init(Dim1, Dim2);
		b.init(Dim2, Dim3);
		c.init(Dim1, Dim3);
		a.random(100);
		b.random(100);
		
		cout << "before" << endl;
		cout << "transA: N " << "transB: N" << endl;  
		cout << "a" << endl; a.display();
		cout << "b" << endl; b.display();
		
		c.product(1, 0, false, false, a, b);
		cout << "after" << endl;
		cout <<  "c" << endl; c.display();
		
		cout << "before" << endl;
		cout << "transA: T " << "transB: N" << endl; 
		a.resize(Dim2, Dim1);
		a.random(100);
		cout << "a" << endl; a.display();
		cout << "b" << endl; b.display();
		
		c.product(1, 0, true, false, a, b);
		cout << "after" << endl;
		cout <<  "c" << endl; c.display();
		
		cout << "before" << endl;
		cout << "transA: N " << "transB: T" << endl; 
		b.resize(Dim3, Dim2);
		b.random(100);
		cout << "a" << endl; a.display();
		cout << "b" << endl; b.display();
		
		c.product(1, 0, false, true, a, b);
		cout << "after" << endl;
		cout <<  "c" << endl; c.display();
		
		cout << "before" << endl;
		cout << "transA: T " << "transB: T" << endl;  
		cout << "a" << endl; a.display();
		cout << "b" << endl; b.display();
		
		a.resize(Dim2, Dim1);
		b.resize(Dim3, Dim2);
		a.random(100);
		b.random(100);
		c.product(1, 0, true, true, a, b);
		cout << "after" << endl;
		cout <<  "c" << endl; c.display();
	}
#endif
}

void test_add(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_add:" << endl;
		cpu_matrix a;
		cpu_matrix b;
		cpu_matrix c;
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
		c.init(Dim1, Dim2);
		a.random(100);
		b.random(100);
		
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout << "b" << endl; b.display();
		
		c.add(a, b);
		cout << "after" << endl;
		cout <<  "c" << endl; c.display();
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_add:" << endl;
		gpu_matrix a;
		gpu_matrix b;
		gpu_matrix c;
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
		c.init(Dim1, Dim2);
		a.random(100);
		b.random(100);
		
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout << "b" << endl; b.display();
		
		c.add(a, b);
		cout << "after" << endl;
		cout <<  "c" << endl; c.display();
	}
#endif
}

void test_sub(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_sub:" << endl;
		cpu_matrix a;
		cpu_matrix b;
		cpu_matrix c;
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
		c.init(Dim1, Dim2);
		a.random(100);
		b.random(100);
		
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout << "b" << endl; b.display();
		
		c.sub(a, b);
		cout << "after" << endl;
		cout <<  "c" << endl; c.display();
		
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_sub:" << endl;
		gpu_matrix a;
		gpu_matrix b;
		gpu_matrix c;
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
		c.init(Dim1, Dim2);
		a.random(100);
		b.random(100);
		
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout << "b" << endl; b.display();
		
		c.sub(a, b);
		cout << "after" << endl;
		cout <<  "c" << endl; c.display();
	}
#endif
}

void test_multiply(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_multi:" << endl;
		cpu_matrix a;
		cpu_matrix b;
		cpu_matrix c;
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
		c.init(Dim1, Dim2);
		a.random(100);
		b.random(100);
		
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout << "b" << endl; b.display();
		
		c.multiply(a, b);
		cout << "after" << endl;
		cout <<  "c" << endl; c.display();
		
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_mutli:" << endl;
		gpu_matrix a;
		gpu_matrix b;
		gpu_matrix c;
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
		c.init(Dim1, Dim2);
		a.random(100);
		b.random(100);
		
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout << "b" << endl; b.display();
		
		c.multiply(a, b);
		cout << "after" << endl;
		cout <<  "c" << endl; c.display();
	}
#endif
}

void test_divide(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_divide:" << endl;
		cpu_matrix a;
		cpu_matrix b;
		cpu_matrix c;
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
		c.init(Dim1, Dim2);
		a.random(100);
		b.random(100);
		
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout << "b" << endl; b.display();
		
		c.divide(a, b);
		cout << "after" << endl;
		cout <<  "c" << endl; c.display();
		
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_divide:" << endl;
		gpu_matrix a;
		gpu_matrix b;
		gpu_matrix c;
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
		c.init(Dim1, Dim2);
		a.random(100);
		b.random(100);
		
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout << "b" << endl; b.display();
		
		c.divide(a, b);
		cout << "after" << endl;
		cout <<  "c" << endl; c.display();
	}
#endif
}

void test_self_add(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_self_add:" << endl;
		cpu_matrix a;
		cpu_matrix b;
	
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
	
		a.random(100);
		b.random(100);
	
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout <<  "b" << endl; b.display();
		
		b.self_add(a);
		cout << "after" << endl;
		cout <<  "b" << endl; b.display();
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_self_add:" << endl;
		gpu_matrix a;
		gpu_matrix b;
	
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
	
		a.random(100);
		b.random(100);
	
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout <<  "b" << endl; b.display();
		
		
		b.self_add(a);
		cout << "after" << endl;
		cout <<  "b" << endl; b.display();
	}
#endif
}

void test_self_sub(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_self_sub:" << endl;
		cpu_matrix a;
		cpu_matrix b;
	
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
	
		a.random(100);
		b.random(100);
	
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout <<  "b" << endl; b.display();
		
		
		b.self_sub(a);
		cout << "after" << endl;
		cout <<  "b" << endl; b.display();
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_self_sub:" << endl;
		gpu_matrix a;
		gpu_matrix b;
	
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
	
		a.random(100);
		b.random(100);
	
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout <<  "b" << endl; b.display();
		
		
		b.self_sub(a);
		cout << "after" << endl;
		cout <<  "b" << endl; b.display();
	}
#endif
}

void test_self_multiply(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_self_multiply:" << endl;
		cpu_matrix a;
		cpu_matrix b;
	
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
	
		a.random(100);
		b.random(100);
	
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout <<  "b" << endl; b.display();
		
		
		b.self_multiply(a);
		cout << "after" << endl;
		cout <<  "b" << endl; b.display();
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_self_sub:" << endl;
		gpu_matrix a;
		gpu_matrix b;
	
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
	
		a.random(100);
		b.random(100);
	
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout <<  "b" << endl; b.display();
		
		
		b.self_multiply(a);
		cout << "after" << endl;
		cout <<  "b" << endl; b.display();
	}
#endif
}

void test_self_divide(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_self_divide:" << endl;
		cpu_matrix a;
		cpu_matrix b;
	
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
	
		a.random(100);
		b.random(100);
	
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout <<  "b" << endl; b.display();
		
		
		b.self_divide(a);
		cout << "after" << endl;
		cout <<  "b" << endl; b.display();
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_self_divide:" << endl;
		gpu_matrix a;
		gpu_matrix b;
	
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
	
		a.random(100);
		b.random(100);
	
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout <<  "b" << endl; b.display();
		
		
		b.self_divide(a);
		cout << "after" << endl;
		cout <<  "b" << endl; b.display();
	}
#endif
}

void test_tanh(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_tanh:" << endl;
		cpu_matrix a;
		cpu_matrix b;
	
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
	
		a.random(100);
	
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout <<  "b" << endl; b.display();
		
		
		b.tanh(a);
		cout << "after" << endl;
		cout <<  "b" << endl; b.display();
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_tanh:" << endl;
		gpu_matrix a;
		gpu_matrix b;
	
		a.init(Dim1, Dim2);
		b.init(Dim1, Dim2);
	
		a.random(100);
	
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		cout <<  "b" << endl; b.display();
		
		
		b.tanh(a);
		cout << "after" << endl;
		cout <<  "b" << endl; b.display();
	}
#endif
}

void test_assign(dtype value, bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_scale_assign:" << endl;
		cout << "value:" << value << endl;
		cpu_matrix a;
	
		a.init(Dim1, Dim2);
	
		a.random(100);
	
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		
		a.assign(value);
		cout << "after" << endl;
		cout <<  "a" << endl; a.display();
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_scale_assign:" << endl;
		cout << "value:" << value << endl;
		gpu_matrix a;
	
		a.init(Dim1, Dim2);
	
		a.random(100);
	
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		
		a.assign(value);
		cout << "after" << endl;
		cout <<  "a" << endl; a.display();
	}
#endif
}


void test_lookup(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_lookup:" << endl;
		cpu_matrix a;
		cpu_matrix b;
	
		a.init(Dim1, Dim2);
		b.init(Dim1, 1);
	
		a.random(100);
	
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		
		b.lookup(a, 2);
		cout << "after" << endl;
		cout <<  "b" << endl; b.display();
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_lookup:" << endl;
		gpu_matrix a;
		gpu_matrix b;
	
		a.init(Dim1, Dim2);
		b.init(Dim1, 1);
	
		a.random(100);
	
		cout << "before" << endl;
		cout << "a" << endl; a.display();
		
		b.lookup(a, 2);
		cout << "after" << endl;
		cout <<  "b" << endl; b.display();
	}
#endif
}

void test_concat(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_concat:" << endl;
		vector<cpu_matrix> vec_a;
		cpu_matrix b;
		vec_a.resize(5);
		b.init(Dim1, 5);
		for(int i=0; i<vec_a.size(); i++){
			vec_a[i].init(Dim1, 1);
			vec_a[i].random(100);
			cout << "cpu vec " << i << endl;
			vec_a[i].display();
		}
		
		cout << "concat" << endl;
		
		b.concat(vec_a);
		
		cout << "res " << endl; b.display();
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_concat:" << endl;
		vector<gpu_matrix> vec_a;
		gpu_matrix b;
		vec_a.resize(5);
		b.init(Dim1, 5);
		for(int i=0; i<vec_a.size(); i++){
			vec_a[i].init(Dim1, 1);
			vec_a[i].random(100);
			cout << "gpu vec " << i << endl;
			vec_a[i].display();
		}
		
		cout << "concat" << endl;
		
		b.concat(vec_a);
		
		cout << "res " << endl; b.display();
	}
#endif
}

void test_drop_out(dtype drop_value, bool is_cpu){
	if(is_cpu){
		cout << "this is cpu drop out" << endl;
		cpu_matrix a;
		a.init(Dim1, Dim2);
		a.random(100);
		cpu_matrix b;
		b.init(Dim1, Dim2);
		cpu_matrix mask;
		mask.init(Dim1, Dim2);
		
		cout << "before"  << endl;
		cout << "a" << endl; a.display();
		
		b.dropout(a, mask, drop_value, 0);
		cout << "after" << endl;
		cout << "mask" << endl; mask.display();
		cout << "b" << endl; b.display();
	}
#if USE_GPU
	else{
		cout << "this is gpu drop out" << endl;
		gpu_matrix a;
		a.init(Dim1, Dim2);
		a.random(100);
		gpu_matrix b;
		b.init(Dim1, Dim2);
		gpu_matrix mask;
		mask.init(Dim1, Dim2);
		
		cout << "before"  << endl;
		cout << "a" << endl; a.display();
		
		b.dropout(a, mask, drop_value, 0);
		cout << "after" << endl;
		cout << "mask" << endl; mask.display();
		cout << "b" << endl; b.display();
	}
#endif
}

void test_max_pooling_helper(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_max_pooling_help:" << endl;
		vector<cpu_matrix> vec_a;
		vec_a.resize(5);
		for(int i=0; i<vec_a.size(); i++){
			vec_a[i].init(Dim1, 1);
			vec_a[i].random(100);
			cout << "cpu vec " << i << endl;vec_a[i].display();
		}
		
		vector<cpu_matrix> mask;
		mask.resize(5);
		for(int i=0; i<vec_a.size(); i++){
			mask[i].init(Dim1, 1);
		}
		
		cout << "after" << endl;
		max_pooling_helper(vec_a, mask);
		for(int i=0; i<vec_a.size(); i++){
			cout << "cpu mask " << i << endl;mask[i].display();
		}
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_max_pooling_help:" << endl;
		vector<gpu_matrix> vec_a;
		vec_a.resize(5);
		for(int i=0; i<vec_a.size(); i++){
			vec_a[i].init(Dim1, 1);
			vec_a[i].random(100);
			cout << "gpu vec " << i << endl;vec_a[i].display();
		}
		
		vector<gpu_matrix> mask;
		mask.resize(5);
		for(int i=0; i<vec_a.size(); i++){
			mask[i].init(Dim1, 1);
		}
		
		cout << "after" << endl;
		max_pooling_helper(vec_a, mask);
		for(int i=0; i<vec_a.size(); i++){
			cout << "gpu mask " << i << endl;mask[i].display();
		}
	}
#endif
}

void test_min_pooling_helper(bool is_cpu){
	if(is_cpu){
		cout << "this is cpu_min_pooling_help:" << endl;
		vector<cpu_matrix> vec_a;
		vec_a.resize(5);
		for(int i=0; i<vec_a.size(); i++){
			vec_a[i].init(Dim1, 1);
			vec_a[i].random(100);
			cout << "cpu vec " << i << endl;vec_a[i].display();
		}
		
		vector<cpu_matrix> mask;
		mask.resize(5);
		for(int i=0; i<vec_a.size(); i++){
			mask[i].init(Dim1, 1);
		}
		
		cout << "after" << endl;
		min_pooling_helper(vec_a, mask);
		for(int i=0; i<vec_a.size(); i++){
			cout << "cpu mask " << i << endl;mask[i].display();
		}
	}
#if USE_GPU
	else
	{
		cout << "this is gpu_main_pooling_help:" << endl;
		vector<gpu_matrix> vec_a;
		vec_a.resize(5);
		for(int i=0; i<vec_a.size(); i++){
			vec_a[i].init(Dim1, 1);
			vec_a[i].random(100);
			cout << "gpu vec " << i << endl;vec_a[i].display();
		}
		
		vector<gpu_matrix> mask;
		mask.resize(5);
		for(int i=0; i<vec_a.size(); i++){
			mask[i].init(Dim1, 1);
		}
		
		cout << "after" << endl;
		min_pooling_helper(vec_a, mask);
		for(int i=0; i<vec_a.size(); i++){
			cout << "gpu mask " << i << endl;mask[i].display();
		}
	}
#endif
}

void speed_test_zeros(bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			a.zeros();
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_test_zeros: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			a.zeros();
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_test_zeros: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_ones(bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			a.ones();
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_test_ones: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			a.ones();
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_test_ones: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_assign(int icol, int jrow, dtype value, bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			a.assign(icol, jrow, value);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_test_assign_one_value: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			a.assign(icol, jrow, value);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_test_assign_one_value: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_get(int icol, int jrow, bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			a.get(icol, jrow);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_test_get_one_value: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			a.get(icol, jrow);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_test_get_one_value: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_op_equal(){
#if USE_GPU
	if(true){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		gpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b = a;
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu = cpu: ";
		std::cout << diff.count() << std::endl;
	}
	
	if(true){
		gpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		cpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b = a;
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu = gpu: ";
		std::cout << diff.count() << std::endl;
	}
	
#endif

	if(true){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		cpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b = a;
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu = cpu: ";
		std::cout << diff.count() << std::endl;
	}

#if USE_GPU
	if(true){
		gpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		gpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b = a;
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu = gpu: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_transpose(bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		cpu_matrix b;
		b.init(dm2, dm1);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.transpose(a);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu-transpose: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		gpu_matrix b;
		b.init(dm2, dm1);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.transpose(a);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu-transpose: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_add(bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		cpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		cpu_matrix c;
		c.init(dm1, dm2);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			c.add(a, b);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_add: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		gpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		gpu_matrix c;
		c.init(dm1, dm2);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			c.add(a, b);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_add: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_sub(bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		cpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		cpu_matrix c;
		c.init(dm1, dm2);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			c.sub(a, b);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_sub: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		gpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		gpu_matrix c;
		c.init(dm1, dm2);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			c.sub(a, b);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_sub: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_multiply(bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		cpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		cpu_matrix c;
		c.init(dm1, dm2);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			c.multiply(a, b);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_multiply: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		gpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		gpu_matrix c;
		c.init(dm1, dm2);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			c.multiply(a, b);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_multiply: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_product(bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		cpu_matrix b;
		b.init(dm2, dm3);
		b.random(100);
		cpu_matrix c;
		c.init(dm1, dm3);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			c.product(a, b);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_product: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		gpu_matrix b;
		b.init(dm2, dm3);
		b.random(100);
		gpu_matrix c;
		c.init(dm1, dm3);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			c.product(1, 0, false, false, a, b);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_product: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_divide(bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		cpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		cpu_matrix c;
		c.init(dm1, dm2);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			c.divide(a, b);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_divide: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		gpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		gpu_matrix c;
		c.init(dm1, dm2);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			c.divide(a, b);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_divide: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_self_add(bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		cpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.self_add(a);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_self_add: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		gpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.self_add(a);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_self_add: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_self_sub(bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		cpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.self_sub(a);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_self_sub: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		gpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.self_sub(a);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_self_sub: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_self_multiply(bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		cpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.self_multiply(a);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_self_multiply: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		gpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.self_multiply(a);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_self_multiply: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_self_divide(bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		cpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.self_divide(a);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_self_divide: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		gpu_matrix b;
		b.init(dm1, dm2);
		b.random(100);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.self_divide(a);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_self_divide: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_tanh(bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.ones();
		cpu_matrix b;
		b.init(dm1, dm2);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.tanh(a);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_tanh: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		a.ones();
		gpu_matrix b;
		b.init(dm1, dm2);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.tanh(a);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_tanh: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_assign(dtype value, bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			a.assign(value);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_assign_value: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			a.assign(value);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_assign_value: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_lookup(bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		cpu_matrix b;
		b.init(dm1, 1);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.lookup(a, 3);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_lookup: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		gpu_matrix b;
		b.init(dm1, 1);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.lookup(a, 3);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_lookup: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_concat(bool is_cpu){
	if(is_cpu){
		vector<cpu_matrix> vec_a;
		vec_a.resize(10);
		for(int i=0; i<vec_a.size(); i++){
			vec_a[i].init(dm1, 1);
		}
		cpu_matrix b;
		b.init(dm1, 10);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.concat(vec_a);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_concat: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		vector<gpu_matrix> vec_a;
		vec_a.resize(10);
		for(int i=0; i<vec_a.size(); i++){
			vec_a[i].init(dm1, 1);
		}
		gpu_matrix b;
		b.init(dm1, 10);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.concat(vec_a);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_concat: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_drop_out(dtype drop_value, bool is_cpu){
	if(is_cpu){
		cpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		
		cpu_matrix b;
		b.init(dm1, dm2);
		
		cpu_matrix mask;
		mask.init(dm1, dm2);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.dropout(a, mask, drop_value, 100);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_drop_out: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		gpu_matrix a;
		a.init(dm1, dm2);
		a.random(100);
		
		gpu_matrix b;
		b.init(dm1, dm2);
		
		gpu_matrix mask;
		mask.init(dm1, dm2);
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			b.dropout(a, mask, drop_value, 0);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_drop_out: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_max_pooling_helper(bool is_cpu){
	if(is_cpu){
		vector<cpu_matrix> vec_a;
		vec_a.resize(10);
		vector<cpu_matrix> mask;
		mask.resize(10);
		
		for(int i=0; i<vec_a.size(); i++){
			vec_a[i].init(dm1, 1);
			vec_a[i].random(100);
			mask[i].init(dm1, 1);
		}
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			max_pooling_helper(vec_a, mask);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_max_pooling: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		vector<gpu_matrix> vec_a;
		vec_a.resize(10);
		vector<gpu_matrix> mask;
		mask.resize(10);
		
		for(int i=0; i<vec_a.size(); i++){
			vec_a[i].init(dm1, 1);
			vec_a[i].random(100);
			mask[i].init(dm1, 1);
		}
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			max_pooling_helper(vec_a, mask);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_max_pooling: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}

void speed_test_min_pooling_helper(bool is_cpu){
		if(is_cpu){
		vector<cpu_matrix> vec_a;
		vec_a.resize(10);
		vector<cpu_matrix> mask;
		mask.resize(10);
		
		for(int i=0; i<vec_a.size(); i++){
			vec_a[i].init(dm1, 1);
			vec_a[i].random(100);
			mask[i].init(dm1, 1);
		}
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			min_pooling_helper(vec_a, mask);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "cpu_min_pooling: ";
		std::cout << diff.count() << std::endl;
	}
#if USE_GPU
	else{
		vector<gpu_matrix> vec_a;
		vec_a.resize(10);
		vector<gpu_matrix> mask;
		mask.resize(10);
		
		for(int i=0; i<vec_a.size(); i++){
			vec_a[i].init(dm1, 1);
			vec_a[i].random(100);
			mask[i].init(dm1, 1);
		}
		
		auto start = std::chrono::high_resolution_clock::now();
		for(int i=0; i<cnt; i++){
			min_pooling_helper(vec_a, mask);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end-start;
		std::cout << "gpu_min_pooling: ";
		std::cout << diff.count() << std::endl;
	}
#endif
}