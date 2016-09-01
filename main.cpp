#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <iterator>
#include "png.h"
#define WRITE_TO_DISK 1
float factor;

extern "C"
PNG contrastChange(PNG input, int size, unsigned int width,
unsigned int height, float factor);

int main(int arg, char* args[]) {
	cudaError_t cudaStatus;
	std::string inputPath = "lena.png";
	std::string outputPath = "after_lena.png";

	std::cout << "Enter contrast factor \n";
	std::cout << "Factor: ";
	while (std::cin >> factor) {
		std::cout << "\n";

		PNG inputPng(inputPath);
		PNG outPng;
		outPng.Create(inputPng.w, inputPng.h);

		const unsigned int w = inputPng.w;
		const unsigned int h = inputPng.h;
		int size = w * h * 4;

		outPng = contrastChange(inputPng, size, w, h, factor);

		#ifdef WRITE_TO_DISK
		 outPng.Save(outputPath);
		#endif
		outPng.Free();
		inputPng.Free();

		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			std::cout << "Device reset failed!" << std::endl;
			exit(1);
		}
		std::cout << "Enter contrast factor \n";
		std::cout << "Factor: ";
	}
	
	return 0;
}
