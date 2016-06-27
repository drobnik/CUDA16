#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__device__
char roundPix(float x) {
	if (x > 255) {
		return 255;
	}
	else if (x < 0) {
		return 0;
	}
	else {
		return trunc(x);
	}
}

__global__ void contrast(const unsigned char* in, unsigned char* out, float factor) {
	int x = blockIdx.x;
	int y = threadIdx.x;

	int width = blockDim.x;
	int index = (x + y * width) * 4;

	float r = roundPix(factor * (in[index] - 128) + 128);
	float g = roundPix(factor * (in[index + 1] - 128) + 128);
	float b = roundPix(factor * (in[index + 2] - 128) + 128);

	out[index] = (char)r;    //R
	out[index + 1] = (char)g; //G
	out[index + 2] = (char)b; //B
	out[index + 3] = in[index + 3]; //A
}