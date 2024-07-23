#pragma once

#include "device_launch_parameters.h"
#include <cmath>

constexpr float SIGMOID_SCALAR = 2.5f / 400.0f;

__device__ float relu(float x) {
	return fmaxf(0.0f, x);
}

__device__ float reluDer(float x) {
	return x > 0 ? 1.0f : 0.0f;
}

__device__ float sigmoid(float x) {
	return 1.0f / (1.0f + expf(-SIGMOID_SCALAR * x));
}

__device__ float sigmoidDer(float x) {
	float s = 1.0f / (1.0f + expf(-SIGMOID_SCALAR * x));
	return s * (1 - s) * SIGMOID_SCALAR;
}
