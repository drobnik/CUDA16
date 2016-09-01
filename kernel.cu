#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <iterator>
#include "png.h"
#include "kernel_header.cuh"

extern "C"
PNG contrastChange(PNG input, int size, unsigned int width,
					unsigned int height, float factor) {

	unsigned char *in = 0;
	unsigned char *out = 0;
	cudaError_t cudaStatus;
	cudaDeviceProp prop;

	PNG outputPng;
	outputPng.Create(input.w, input.h);

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cout << "No CUDA devices found!\n";
		exit(1);
	}

	cudaGetDeviceProperties(&prop, 0);
	std::cout << "Using device: " << prop.name << "\n";

	cudaMalloc((void**)&in, size * sizeof(unsigned char));
	cudaMalloc((void**)&out, size * sizeof(unsigned char));

	cudaMemcpy(in, &input.data[0], size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	input.Free();

	contrast<< <width, height >> >(in, out, factor);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << "\n";

		cudaFree(in);
		cudaFree(out);

		exit(1);
	} else if(cudaStatus == cudaSuccess){
		std::cout << "An image has been processed.\n\n";
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Could not synchronize device!\n";
		cudaFree(in);
		cudaFree(out);
		exit(1);
	}

	auto tmp = new unsigned char[width * height * 4];

	cudaStatus = cudaMemcpy(tmp, out, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(in);
	cudaFree(out);

	std::copy(&tmp[0], &tmp[width * height * 4], std::back_inserter(outputPng.data));

	delete[] tmp;

	if (cudaStatus != cudaSuccess) {
		std::cout << "Copying from the buffer failed.\n";
		exit(1);
	}

	return outputPng;
}