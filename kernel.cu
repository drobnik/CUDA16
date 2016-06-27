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
/*
int main(int arg, char* args[]) {
	contrFactor = 0.5;
	PNG inPng("Lenna.png");
	PNG outPng;
	outPng.Create(inPng.w, inPng.h);

	//store width and height so we can use them for our output image later
	const unsigned int w = inPng.w;
	const unsigned int h = inPng.h;
	//4 because there are 4 color channels R, G, B, and A
	int size = w * h * 4;

	unsigned char *in = 0;
	unsigned char *out = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cout << "No CUDA devices found!" << std::endl;
		exit(1);
	}

	//prints the device the kernel will be running on
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::cout << "Using device: " << prop.name << std::endl;

	// Allocate GPU buffers for the images
	cudaMalloc((void**)&in, size * sizeof(unsigned char));
	cudaMalloc((void**)&out, size * sizeof(unsigned char));

	// Copy image data from host memory to GPU buffers.
	cudaMemcpy(in, &inPng.data[0], size * sizeof(unsigned char), cudaMemcpyHostToDevice);

	//free the input image because we do not need it anymore
	inPng.Free();

	// Launch a kernel on the GPU with one thread for each element.
	copy << <w, h >> >(in, out, contrFactor);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		cudaFree(in);
		cudaFree(out);
		exit(1);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Could not synchronize device!" << std::endl;
		cudaFree(in);
		cudaFree(out);
		exit(1);
	}

	//temporary array to store the result from opencl
	auto tmp = new unsigned char[w * h * 4];
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(tmp, out, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaFree(in);
	cudaFree(out);

	//copy the data from the temp array to the png
	std::copy(&tmp[0], &tmp[w * h * 4], std::back_inserter(outPng.data));

	//write the image to file
	outPng.Save("cuda_tutorial_2.png");
	//free the iamge's resources since we are done with it
	outPng.Free();

	//free the temp array
	delete[] tmp;

	if (cudaStatus != cudaSuccess) {
		std::cout << "Could not copy buffer memory to host!" << std::endl;
		exit(1);
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Device reset failed!" << std::endl;
		exit(1);
	}

	return 0;
	}*/