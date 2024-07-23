#pragma once

#include <random>
#include "dataset.h"

namespace Tools {
	random_device rd;
	mt19937 gen(42);
} // namespace Tools

// he init for weights 
void heInit(float* array, int numFeatures, int numNeurons) {
	for (int i = 0; i < numNeurons * numFeatures; ++i) {
		normal_distribution<float> dist(0, sqrt(2.0f / numFeatures));

		array[i] = dist(Tools::gen);
	}
}

void shuffleData(vector<NetInput>& data) {
	for (size_t i = data.size() - 1; i > 0; --i) {
		uniform_int_distribution<size_t> dis(0, i);
		size_t j = dis(Tools::gen);
		swap(data[i], data[j]);
	}
}
