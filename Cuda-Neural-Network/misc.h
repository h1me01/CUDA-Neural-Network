#pragma once

#include <random>
#include "dataset.h"

namespace Tools {
	random_device rd;
	mt19937 gen(42);
} // namespace Tools

// he init for weights 
void heInit(float* array, int numFeatures, int numNeurons) {
	normal_distribution<float> dist(0, sqrt(2.0f / numFeatures));

	for (int i = 0; i < numNeurons * numFeatures; ++i) {
		array[i] = dist(Tools::gen);
	}
}

void shuffleData(vector<NetInput>& data) {
	uniform_int_distribution<size_t> dis;

	for (size_t i = data.size() - 1; i > 0; --i) {
		dis.param(uniform_int_distribution<size_t>::param_type(0, i)); 
		size_t j = dis(Tools::gen);
		swap(data[i], data[j]);
	}
}
