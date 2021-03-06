#include "cpu_matrix.h"
#include "gpu_matrix.h"
#include <iostream>
#include <chrono>

int main()
{
	cpu_matrix cA;
	cA.init(200, 10000);
	cpu_matrix cB;
	cB.init(200, 10000);
	cpu_matrix cC;
	cC.init(200, 10000);
	
	gpu_matrix gA;
	gpu_matrix gB;
	gpu_matrix gC;
	gC.init(200, 10000);
	
	gB = cB;
	
	int count = 10000;
	auto start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		cC.tanh(cA);
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end-start;
	std::cout << "cpu: ";
	std::cout << diff.count() << std::endl;
	
	start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		gA = cA;
		gC.tanh(gA);
		cC = gC;
	}
	end = std::chrono::high_resolution_clock::now();
	diff = end-start;
	std::cout << "gpu: ";
	std::cout << diff.count() << std::endl;
	
	return 0;
}
