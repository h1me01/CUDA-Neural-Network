#pragma once

#include "helper.h"

class Layer {
public:
	float* targets, * inputs;
	float* activations, * weighted_inputs, * deltas;
	float* weights, * weights_grad, * m_weights, * v_weights;
	float* biases, * biases_grad, * m_biases, * v_biases;

	Layer(int batch_size, int num_features, int layer_size) {
		allocDevMem(&targets, batch_size);
		allocDevMem(&inputs, batch_size * num_features);

		size_t size = batch_size * layer_size;
		allocDevMem(&activations, size);
		allocDevMem(&weighted_inputs, size);
		allocDevMem(&deltas, size);

		size = num_features * layer_size;
		allocDevMem(&weights, size);
		allocDevMem(&weights_grad, size);
		allocDevMem(&m_weights, size);
		allocDevMem(&v_weights, size);

		allocDevMem(&biases, layer_size);
		allocDevMem(&biases_grad, layer_size);
		allocDevMem(&m_biases, layer_size);
		allocDevMem(&v_biases, layer_size);

		// set biases to 0
		setDevMem(biases, 0.0f, layer_size);

		// set gradients to 0
		setDevMem(weights_grad, 0.0f, size);
		setDevMem(biases_grad, 0.0f, layer_size);

		// set m and v to 0
		setDevMem(m_weights, 0.0f, size);
		setDevMem(v_weights, 0.0f, size);
		setDevMem(m_biases, 0.0f, layer_size);
		setDevMem(v_biases, 0.0f, layer_size);
	}

	~Layer() {
		freeDevMem(targets);
		freeDevMem(inputs);
		freeDevMem(activations);
		freeDevMem(weighted_inputs);
		freeDevMem(deltas);
		freeDevMem(weights);
		freeDevMem(weights_grad);
		freeDevMem(m_weights);
		freeDevMem(v_weights);
		freeDevMem(biases);
		freeDevMem(biases_grad);
		freeDevMem(m_biases);
		freeDevMem(v_biases);
	}
};
