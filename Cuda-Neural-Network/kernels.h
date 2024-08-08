#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>

#include "helper.h"


__global__ void feedKernel(
	float* batch_input, 
	float* weights, 
	float* biases, 
	float* weighted_inputs, 
	float* activations, 
	int input_size, int layer_size
) {
	int batch = blockIdx.x;
	int neuron = threadIdx.x;

	// gridDim.x is the batch size
	if (batch >= gridDim.x || neuron >= layer_size) {
		return;
	}

	float sum = 0.0f;
	for (int i = 0; i < input_size; ++i) {
		sum += batch_input[MI(batch, i, input_size)] * weights[MI(neuron, i, input_size)];
	}
	sum += biases[neuron];

	weighted_inputs[MI(batch, neuron, layer_size)] = sum;

	float activation = 0.0f;
	if (layer_size == 1) {
		constexpr float SIGMOID_SCALAR = 2.5f / 400.0f;
		activation = 1.0f / (1.0f + exp(-SIGMOID_SCALAR * sum));
	}
	else {
		activation = sum > 0 ? sum : 0;
	}
	activations[MI(batch, neuron, layer_size)] = activation;
}

__global__ void backpropOutputKernel(
	float* targets, 
	float* predictions, 
	float* batch_input, 
	float* weighted_inputs, 
	float* weights_grad, 
	float* biases_grad, 
	float* deltas, 
	int input_size, int layer_size
) {
	int batch = blockIdx.x;
	int neuron = threadIdx.x;

	// gridDim.x is the batch size
	if (batch >= gridDim.x || neuron >= layer_size) {
		return;
	}

	// calculate delta for output layer
	constexpr float SIGMOID_SCALAR = 2.5f / 400.0f;
	float sig = 1.0f / (1.0f + exp(-SIGMOID_SCALAR * weighted_inputs[batch]));
	float deltaOutput = 2 * (predictions[batch] - targets[batch]) * (sig * (1 - sig) * SIGMOID_SCALAR);
	deltas[batch] = deltaOutput;

	// update gradients
	for (int i = 0; i < input_size; ++i) {
		atomicAdd(&weights_grad[MI(neuron, i, input_size)], deltaOutput * batch_input[MI(batch, i, input_size)]);
	}

	atomicAdd(&biases_grad[neuron], deltaOutput);
}

__global__ void backpropHiddenKernel(
	float* batch_input, 
	float* weighted_inputs, 
	float* weights_grad, 
	float* biases_grad,
	float* next_weights, 
	float* next_deltas, 
	int input_size, int layer_size
) {
	int batch = blockIdx.x;
	int neuron = threadIdx.x;

	// gridDim.x is the batch size
	if (batch >= gridDim.x || neuron >= layer_size) {
		return;
	}

	// calculate delta for hidden layer
	float delta = next_weights[neuron] * next_deltas[batch] * (weighted_inputs[MI(batch, neuron, layer_size)] > 0 ? 1.0f : 0.0f);
	// deltas[MI(batch, neuron, layer_size)] = delta; // not needed since I have only one hidden layer

	// update gradients
	for (int i = 0; i < input_size; ++i) {
		atomicAdd(&weights_grad[MI(neuron, i, input_size)], delta * batch_input[MI(batch, i, input_size)]);
	}

	atomicAdd(&biases_grad[neuron], delta);
}

__global__ void adamKernel(
	float* weights,
	float* biases,
	float* weights_grad,
	float* biases_grad,
	float* m_w, float* v_w,
	float* m_b, float* v_b,
	float beta1, float beta2, float eps, float lr,
	int input_size, int layer_size
) {
	int layer_neuron = blockIdx.x;
	int input_neuron = threadIdx.x;

	if (layer_neuron >= layer_size || input_neuron >= input_size) {
		return;
	}

	// update weights
	int weight_idx = MI(layer_neuron, input_neuron, input_size);

	m_w[weight_idx] = beta1 * m_w[weight_idx] + (1.0f - beta1) * weights_grad[weight_idx];
	v_w[weight_idx] = beta2 * v_w[weight_idx] + (1.0f - beta2) * weights_grad[weight_idx] * weights_grad[weight_idx];

	float weights_delta = lr * m_w[weight_idx] / (sqrt(v_w[weight_idx]) + eps);
	weights[weight_idx] -= weights_delta;
	weights_grad[weight_idx] = 0.0f;

	// update biases, we only want to update the bias once for each neuron
	// since each neuron has input_size threads by adding input_neuron == 0 (can be any number between 0 and input_size - 1)
	// we update it only once by one of the threads
	if (input_neuron == 0) {
		m_b[layer_neuron] = beta1 * m_b[layer_neuron] + (1.0f - beta1) * biases_grad[layer_neuron];
		v_b[layer_neuron] = beta2 * v_b[layer_neuron] + (1.0f - beta2) * biases_grad[layer_neuron] * biases_grad[layer_neuron];

		float biases_delta = lr * m_b[layer_neuron] / (sqrt(v_b[layer_neuron]) + eps);
		biases[layer_neuron] -= biases_delta;
		biases_grad[layer_neuron] = 0.0f;
	}
}
