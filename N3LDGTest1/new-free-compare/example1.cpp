#include "cpu_matrix.h"
#include "gpu_matrix.h"
#include <iostream>
#include <chrono>

int main()
{
	
	cudaFree(0);
	int dim1 = 20;
	int dim2 = 10;
	int dim3 = 40;
	
	InitGPU(DEVICE::getInstance(), 0, 1024);
	
	gpu_matrix gA;
	
	gA.init(dim1, dim2);
	// int count = 100000;
	// auto start = std::chrono::high_resolution_clock::now();
	// for(int i=0; i<count; i++){
		// cA.init(dim1, dim2);
		// cA.delloc();
	// }
	// auto end = std::chrono::high_resolution_clock::now();
	// std::chrono::duration<double> diff = end-start;
	// std::cout << "cpu-new-free: ";
	// std::cout << diff.count() << std::endl;
	
	// start = std::chrono::high_resolution_clock::now();
	// for(int i=0; i<count; i++){
		// gA.init(dim1, dim2);
		// gA.delloc();
	// }
	// end = std::chrono::high_resolution_clock::now();
	// diff = end-start;
	// std::cout << "gpu-new-free: ";
	// std::cout << diff.count() << std::endl;
	
	// cpu_matrix cB;
	// cB.init(dim1, dim2);
	// cpu_matrix cC;
	// cC.init(dim2, dim3);
	// cpu_matrix cD;
	// cD.init(dim1, dim3);
	
	// start = std::chrono::high_resolution_clock::now();
	// for(int i=0; i<count; i++){
		// cD.product(cB, cC);
	// }
	// end = std::chrono::high_resolution_clock::now();
	// diff = end-start;
	// std::cout << "cpu-product: ";
	// std::cout << diff.count() << std::endl;
	
	
	// gpu_matrix gB;
	// gB.init(dim1, dim2);
	// gpu_matrix gC;
	// gC.init(dim2, dim3);
	// gpu_matrix gD;
	// gD.init(dim1, dim3);
	
	// start = std::chrono::high_resolution_clock::now();
	// for(int i=0; i<count; i++){
		// gD.product(gB, gC);
	// }
	// end = std::chrono::high_resolution_clock::now();
	// diff = end-start;
	// std::cout << "gpu-product: ";
	// std::cout << diff.count() << std::endl;
	
	
	// cpu_matrix cE;
	// cE.init(dim1, dim2);
	// cpu_matrix cF;
	// cF.init(dim1, dim2);
	// cpu_matrix cG;
	// cG.init(dim1, dim2);
	// start = std::chrono::high_resolution_clock::now();
	// for(int i=0; i<count; i++){
		// cG.add(cE, cF);
	// }
	// end = std::chrono::high_resolution_clock::now();
	// diff = end-start;
	// std::cout << "cpu-add: ";
	// std::cout << diff.count() << std::endl;
	
	
	// gpu_matrix gE;
	// gE.init(dim1, dim2);
	// gpu_matrix gF;
	// gF.init(dim1, dim2);
	// gpu_matrix gG;
	// gG.init(dim1, dim2);
	// start = std::chrono::high_resolution_clock::now();
	// for(int i=0; i<count; i++){
		// gG.add(gE, gF);
	// }
	// end = std::chrono::high_resolution_clock::now();
	// diff = end-start;
	// std::cout << "gpu-add: ";
	// std::cout << diff.count() << std::endl;

	
	// cpu_matrix ca;
	// ca.init(dim1, dim2);
	// cpu_matrix cb;
	// cb.init(dim1, dim2);
	// cpu_matrix cc;
	// cc.init(dim1, dim2);
	// start = std::chrono::high_resolution_clock::now();
	// for(int i=0; i<count; i++){
		// cc.multiply(ca, cb);
	// }
	// end = std::chrono::high_resolution_clock::now();
	// diff = end-start;
	// std::cout << "cpu-multiply: ";
	// std::cout << diff.count() << std::endl;
	
	// gpu_matrix ga;
	// ga.init(dim1, dim2);
	// gpu_matrix gb;
	// gb.init(dim1, dim2);
	// gpu_matrix gc;
	// gc.init(dim1, dim2);
	// start = std::chrono::high_resolution_clock::now();
	// for(int i=0; i<count; i++){
		// gc.multiply(ga, gb);
	// }
	// end = std::chrono::high_resolution_clock::now();
	// diff = end-start;
	// std::cout << "gpu-multiply: ";
	// std::cout << diff.count() << std::endl;
	
	// cpu_matrix cd;
	// cd.init(dim1, dim2);
	// cpu_matrix ce;
	// ce.init(dim1, dim2);
	// cpu_matrix cf;
	// cf.init(dim1, dim2);
	// start = std::chrono::high_resolution_clock::now();
	// for(int i=0; i<count; i++){
		// cf.tanh(cd);
	// }
	// end = std::chrono::high_resolution_clock::now();
	// diff = end-start;
	// std::cout << "cpu-tanh: ";
	// std::cout << diff.count() << std::endl;
	
	
	// gpu_matrix gd;
	// gd.init(dim1, dim2);
	// gpu_matrix ge;
	// ge.init(dim1, dim2);
	// gpu_matrix gf;
	// gf.init(dim1, dim2);
	// start = std::chrono::high_resolution_clock::now();
	// for(int i=0; i<count; i++){
		// gf.tanh(gd);
	// }
	// end = std::chrono::high_resolution_clock::now();
	// diff = end-start;
	// std::cout << "gpu-tanh: ";
	// std::cout << diff.count() << std::endl;
	
	FinalizeGPU();
	return 0;
}
