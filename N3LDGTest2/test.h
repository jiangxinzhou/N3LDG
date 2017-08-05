#ifndef _TEST_
#define _TEST_

#include "gpu_matrix.h"
#include "cpu_matrix.h"

#include <iostream>

#include <chrono>

using namespace std;

const int Dim1 = 3;
const int Dim2 = 5;
const int Dim3 = 4;

static int dm1 = 300;
static int dm2 = 10000;
static int dm3 = 400;

static int cnt = 10000;



void test_zeros(bool is_cpu = true);
void test_ones(bool is_cpu = true);
void test_assign(int icol, int jrow, dtype value, bool is_cpu = true);
void test_get(int icol, int jrow, bool is_cpu = true);
void test_op_equal();
void test_random(bool is_cpu = true);
void test_transpose(bool is_cpu = true);
// void test_self_transpose(bool is_cpu = true);
void test_product(bool is_cpu = true);
void test_add(bool is_cpu = true);
void test_sub(bool is_cpu = true);
void test_multiply(bool is_cpu = true);
void test_divide(bool is_cpu = true);
void test_self_add(bool is_cpu = true);
void test_self_sub(bool is_cpu = true);
void test_self_multiply(bool is_cpu = true);
void test_self_divide(bool is_cpu = true);
void test_tanh(bool is_cpu = true);
void test_assign(dtype value, bool is_cpu = true);
void test_lookup(bool is_cpu = true);
void test_concat(bool is_cpu = true);
void test_drop_out(dtype drop_value, bool is_cpu = true);
void test_max_pooling_helper(bool is_cpu = true);
void test_min_pooling_helper(bool is_cpu = true);


void speed_test_product(bool is_cpu= true);
void speed_test_zeros(bool is_cpu = true);
void speed_test_ones(bool is_cpu = true);
void speed_test_assign(int icol, int jrow, dtype value, bool is_cpu = true);
void speed_test_get(int icol, int jrow, bool is_cpu = true);
void speed_test_op_equal();
void speed_test_random(bool is_cpu = true);
void speed_test_transpose(bool is_cpu = true);
// void test_self_transpose(bool is_cpu = true);
void speed_test_add(bool is_cpu = true);
void speed_test_sub(bool is_cpu = true);
void speed_test_multiply(bool is_cpu = true);
void speed_test_divide(bool is_cpu = true);
void speed_test_self_add(bool is_cpu = true);
void speed_test_self_sub(bool is_cpu = true);
void speed_test_self_multiply(bool is_cpu = true);
void speed_test_self_divide(bool is_cpu = true);
void speed_test_tanh(bool is_cpu = true);
void speed_test_assign(dtype value, bool is_cpu = true);
void speed_test_lookup(bool is_cpu = true);
void speed_test_concat(bool is_cpu = true);
void speed_test_drop_out(dtype drop_value, bool is_cpu = true);
void speed_test_max_pooling_helper(bool is_cpu = true);
void speed_test_min_pooling_helper(bool is_cpu = true);

#endif


