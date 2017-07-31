#include "cpu_matrix.h"
#include "gpu_matrix.h"
#include <iostream>
#include <chrono>

int main()
{
	
	cudaFree(0);
	
	cpu_matrix cA;
	gpu_matrix gA;

	int count = 100000;
	auto start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		cA.init(200, 500);
		cA.delloc();
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end-start;
	std::cout << "cpu-new-free: ";
	std::cout << diff.count() << std::endl;
	
	start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		gA.init(200, 500);
		gA.delloc();
	}
	end = std::chrono::high_resolution_clock::now();
	diff = end-start;
	std::cout << "gpu-new-free: ";
	std::cout << diff.count() << std::endl;
	
	cpu_matrix cB;
	cB.init(200, 500);
	cpu_matrix cC;
	cC.init(500, 400);
	cpu_matrix cD;
	cD.init(200, 400);
	
	start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		cD.product(cB, cC);
	}
	end = std::chrono::high_resolution_clock::now();
	diff = end-start;
	std::cout << "cpu(200x500)product(500x400):";
	std::cout << diff.count() << std::endl;
	
	
	gpu_matrix gB;
	gB.init(200, 500);
	gpu_matrix gC;
	gC.init(500, 400);
	gpu_matrix gD;
	gD.init(200, 400);
	
	start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		gD.product(gB, gC);
	}
	end = std::chrono::high_resolution_clock::now();
	diff = end-start;
	std::cout << "gpu(200x500)product(500x400):";
	std::cout << diff.count() << std::endl;
	
	
	cpu_matrix cE;
	cE.init(200, 500);
	cpu_matrix cF;
	cF.init(200, 500);
	cpu_matrix cG;
	cG.init(200, 500);
	start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		cG.add(cE, cF);
	}
	end = std::chrono::high_resolution_clock::now();
	diff = end-start;
	std::cout << "cpu(200x500)add(200x500):";
	std::cout << diff.count() << std::endl;
	
	
	gpu_matrix gE;
	gE.init(200, 500);
	gpu_matrix gF;
	gF.init(200, 500);
	gpu_matrix gG;
	gG.init(200, 500);
	start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		gG.add(gE, gF);
	}
	end = std::chrono::high_resolution_clock::now();
	diff = end-start;
	std::cout << "gpu(200x500)add(200x500):";
	std::cout << diff.count() << std::endl;

	
	cpu_matrix ca;
	ca.init(200, 500);
	cpu_matrix cb;
	cb.init(200, 500);
	cpu_matrix cc;
	cc.init(200, 500);
	start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		cc.multiply(ca, cb);
	}
	end = std::chrono::high_resolution_clock::now();
	diff = end-start;
	std::cout << "cpu(200x500)multiply(200x500):";
	std::cout << diff.count() << std::endl;
	
	gpu_matrix ga;
	ga.init(200, 500);
	gpu_matrix gb;
	gb.init(200, 500);
	gpu_matrix gc;
	gc.init(200, 500);
	start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		gc.multiply(ga, gb);
	}
	end = std::chrono::high_resolution_clock::now();
	diff = end-start;
	std::cout << "gpu(200x500)multiply(200x500):";
	std::cout << diff.count() << std::endl;
	
	cpu_matrix cd;
	cd.init(200, 500);
	cpu_matrix ce;
	ce.init(200, 500);
	cpu_matrix cf;
	cf.init(200, 500);
	start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		cf.multiply(cd, ce);
	}
	end = std::chrono::high_resolution_clock::now();
	diff = end-start;
	std::cout << "cpu(200x500)tanh(200x500):";
	std::cout << diff.count() << std::endl;
	
	gpu_matrix gd;
	gd.init(200, 500);
	gpu_matrix ge;
	ge.init(200, 500);
	gpu_matrix gf;
	gf.init(200, 500);
	start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		gf.multiply(gd, ge);
	}
	end = std::chrono::high_resolution_clock::now();
	diff = end-start;
	std::cout << "gpu(200x500)tanh(200x500):";
	std::cout << diff.count() << std::endl;
	
	return 0;
}
