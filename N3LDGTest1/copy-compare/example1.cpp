#include "cpu_matrix.h"
#include "gpu_matrix.h"
#include <iostream>
#include <chrono>

int main()
{
	cpu_matrix cA;
	cA.init(200, 500);
	cA.random(100);
	
	cpu_matrix cB;
	cB.init(200, 500);
	cB.random(100);
	

	gpu_matrix gA;
	gA = cA;

	gpu_matrix gB;
	gB = cB;
	

	int count = 100000;
	auto start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		gA = cA;
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end-start;
	std::cout << "cpu2gpu: ";
	std::cout << diff.count() << std::endl;
	
	start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		cA = gA;
	}
	end = std::chrono::high_resolution_clock::now();
	diff = end-start;
	std::cout << "gpu2cpu: ";
	std::cout << diff.count() << std::endl;
	
	
	start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		cA = cB;
	}
	end = std::chrono::high_resolution_clock::now();
	diff = end-start;
	std::cout << "cpu2cpu: ";
	std::cout << diff.count() << std::endl;
	
	start = std::chrono::high_resolution_clock::now();
	for(int i=0; i<count; i++){
		gA = gB;
	}
	end = std::chrono::high_resolution_clock::now();
	diff = end-start;
	std::cout << "gpu2gpu: ";
	std::cout << diff.count() << std::endl;
	
	return 0;
}
