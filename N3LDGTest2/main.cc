#include "test.h"

extern int dm1;
extern int dm2;
extern int dm3;
extern int cnt;


int main(int argc, char **argv){
	// cnt = atoi(argv[1]);
	// dm1 = atoi(argv[2]);
	// dm2 = atoi(argv[3]);
	// dm3 = atoi(argv[4]);
	InitGPU(DEVICE::getInstance());
	
	cout << "count: " << cnt << endl; 
	cout << "dm1: " << dm1 << " dm2: " << dm2 << " dm3: " << dm3 << endl;
	speed_test_zeros();
	speed_test_ones();
	speed_test_assign(1, 1, 10);
	speed_test_get(1, 1);
	speed_test_transpose();
	speed_test_add();
	speed_test_sub();
	speed_test_multiply();
	speed_test_divide();
	speed_test_self_add();
	speed_test_self_sub();
	speed_test_self_multiply();
	speed_test_self_divide();
	speed_test_product();
	speed_test_tanh();
	 speed_test_assign(10);
	speed_test_lookup();
	 speed_test_concat();
	speed_test_drop_out(0.33);
	speed_test_max_pooling_helper();
	speed_test_min_pooling_helper();
	
	speed_test_op_equal();
	
	speed_test_zeros(false);
	speed_test_ones(false);
	speed_test_assign(1, 1, 10, false);
	speed_test_get(1, 1, false);
	speed_test_transpose(false);
	speed_test_add(false);
	speed_test_sub(false);
	speed_test_multiply(false);
	speed_test_divide(false);
	speed_test_self_add(false);
	speed_test_self_sub(false);
	speed_test_self_multiply(false);
	speed_test_self_divide(false);
	speed_test_product(false);
	speed_test_tanh(false);
	speed_test_assign(10, false);
	speed_test_lookup(false);
	speed_test_concat(false);
	speed_test_drop_out(0.33, false);
	speed_test_max_pooling_helper(false);
	speed_test_min_pooling_helper(false);
	
	cout << endl;
	
	
	cout << "Dim1: " << Dim1 << " Dim2: " << Dim2 << " Dim3: " << Dim3 << endl;
	test_zeros();
	test_ones();
	test_assign(1, 1, 10);
	test_get(1, 1);
	test_random();
	test_transpose();
	// test_self_transpose();
	test_product();
	test_add();
	test_sub();
	test_multiply();
	test_divide();
	test_self_add();
	test_self_sub();
	test_self_multiply();
	test_self_divide();
	test_tanh();
	test_assign(10);
	test_lookup();
	test_concat();
	test_drop_out(0.33);
	test_max_pooling_helper();
	test_min_pooling_helper();
	
	test_op_equal();
	
	test_zeros(false);
	test_ones(false);
	test_assign(1, 1, 10, false);
	test_get(1, 1, false);
	//test_random(false);
	test_transpose(false);
	// test_self_transpose();
	test_product(false);
	test_add(false);
	test_sub(false);
	test_multiply(false);
	test_divide(false);
	test_self_add(false);
	test_self_sub(false);
	test_self_multiply(false);
	test_self_divide(false);
	test_tanh(false);
	test_assign(10, false);
	test_lookup(false);
	test_concat(false);
	test_drop_out(0.33, false);
	test_max_pooling_helper(false);
	test_min_pooling_helper(false);
	
	
	FinalizeGPU();
	return 0;
}