#pragma once

#include "dataset.h"
#include "misc.h"

/// HELPER FILE TO HELP ME DEBUG THE GPU VERSION OF IT

/*
 * HE INIT
 */
float heInit(int numFeatures)
{
	normal_distribution<float> dist(0, sqrt(2.0f / numFeatures));
	return dist(Tools::gen);
}

/*
 * NETWORK
 */
class Network
{
public:
	explicit Network()
	{
		int hidden_neurons = HIDDEN_NEURONS;
		int input_neurons = INPUT_NEURONS;
		int batch_size = BATCH_SIZE;

		// Allocate hidden layer memory
		hiddenWeights = new float[hidden_neurons * input_neurons] {};
		hiddenWeightGradients = new float[hidden_neurons * input_neurons] {};
		hiddenWeightsM = new float[hidden_neurons * input_neurons] {};
		hiddenWeightsV = new float[hidden_neurons * input_neurons] {};
		hiddenBiases = new float[hidden_neurons] {};
		hiddenBiasGradients = new float[hidden_neurons] {};
		hiddenBiasesM = new float[hidden_neurons] {};
		hiddenBiasesV = new float[hidden_neurons] {};

		hiddenActivations = new float[batch_size * hidden_neurons] {};
		hiddenWeightedInputs = new float[batch_size * hidden_neurons] {};
		hiddenDeltas = new float[batch_size * hidden_neurons] {};
		hiddenInput = new float[batch_size * input_neurons] {};

		// Allocate output layer memory
		outputWeights = new float[hidden_neurons] {};
		outputWeightGradients = new float[hidden_neurons] {};
		outputWeightsM = new float[hidden_neurons] {};
		outputWeightsV = new float[hidden_neurons] {};

		outputActivation = new float[batch_size] {};
		outputWeightedInput = new float[batch_size] {};
		outputDelta = new float[batch_size] {};
		outputInput = new float[batch_size * hidden_neurons] {};

		//hidden layer init
		for (int i = 0; i < HIDDEN_NEURONS; ++i)
			for (int j = 0; j < INPUT_NEURONS; ++j) {
				hiddenWeights[i * INPUT_NEURONS + j] = heInit(INPUT_NEURONS);
			}

		//output layer init
		outputBias = 0;
		outputBiasGradient = 0;
		outputBiasM = 0;
		outputBiasV = 0;

		for (int i = 0; i < HIDDEN_NEURONS; ++i) {
			outputWeights[i] = heInit(HIDDEN_NEURONS);
		}
	}

	~Network()
	{
		delete[] hiddenWeights;
		delete[] hiddenWeightGradients;
		delete[] hiddenWeightsM;
		delete[] hiddenWeightsV;
		delete[] hiddenBiases;
		delete[] hiddenBiasGradients;
		delete[] hiddenBiasesM;
		delete[] hiddenBiasesV;

		delete[] hiddenActivations;
		delete[] hiddenWeightedInputs;
		delete[] hiddenDeltas;
		delete[] hiddenInput;

		delete[] outputWeights;
		delete[] outputWeightGradients;
		delete[] outputWeightsM;
		delete[] outputWeightsV;

		delete[] outputActivation;
		delete[] outputWeightedInput;
		delete[] outputDelta;
		delete[] outputInput;
	}

	float feedForward(const float* input)
	{
		// hidden layer
		float hiddenOutput[HIDDEN_NEURONS]{};
		for (int i = 0; i < HIDDEN_NEURONS; ++i)
		{
			float dot = 0;
			for (int j = 0; j < INPUT_NEURONS; ++j)
				dot += input[j] * hiddenWeights[i * INPUT_NEURONS + j];
			dot += hiddenBiases[i];

			hiddenOutput[i] = relu(dot);
		}

		// output layer
		float prediction = 0;
		for (int i = 0; i < HIDDEN_NEURONS; ++i)
			prediction += hiddenOutput[i] * outputWeights[i];
		prediction += outputBias;

		return sigmoid(prediction);
	}

	float* feedForward(vector<NetInput> batchInput, int batchSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			float* input = getSparseInput(batchInput[batch]);

			// hidden layer
			for (int i = 0; i < HIDDEN_NEURONS; ++i)
			{
				int idx = batch * HIDDEN_NEURONS + i;
				hiddenWeightedInputs[idx] = 0;

				for (int j = 0; j < INPUT_NEURONS; ++j)
				{
					hiddenWeightedInputs[idx] += input[j] * hiddenWeights[i * INPUT_NEURONS + j];
					hiddenInput[batch * INPUT_NEURONS + j] = input[j];
				}

				hiddenWeightedInputs[idx] += hiddenBiases[i];
				hiddenActivations[idx] = relu(hiddenWeightedInputs[idx]);
			}

			// output layer
			outputWeightedInput[batch] = 0;
			for (int i = 0; i < HIDDEN_NEURONS; ++i)
			{
				int idx = batch * HIDDEN_NEURONS + i;
				outputWeightedInput[batch] += hiddenActivations[idx] * outputWeights[i];
				outputInput[idx] = hiddenActivations[idx];
			}

			outputWeightedInput[batch] += outputBias;
			outputActivation[batch] = sigmoid(outputWeightedInput[batch]);

			delete[] input;
		}

		return outputActivation;
	}

	void feedBackward(vector<float> targets, int batchSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			// output layer

			// calculate output delta
			outputDelta[batch] = 2 * (outputActivation[batch] - targets[batch]) * sigmoidDer(outputWeightedInput[batch]);

			// update output gradients
			for (int i = 0; i < HIDDEN_NEURONS; ++i)
				outputWeightGradients[i] += outputInput[batch * HIDDEN_NEURONS + i] * outputDelta[batch];
			outputBiasGradient += outputDelta[batch];

			// hidden layer

			// calculate hidden deltas
			for (int i = 0; i < HIDDEN_NEURONS; ++i)
				hiddenDeltas[batch * HIDDEN_NEURONS + i] = outputWeights[i] * outputDelta[batch] * reluDer(hiddenWeightedInputs[batch * HIDDEN_NEURONS + i]);

			// update hidden gradients
			for (int i = 0; i < HIDDEN_NEURONS; ++i)
			{
				for (int j = 0; j < INPUT_NEURONS; ++j)
					hiddenWeightGradients[i * INPUT_NEURONS + j] += hiddenInput[batch * INPUT_NEURONS + j] * hiddenDeltas[batch * HIDDEN_NEURONS + i];
				hiddenBiasGradients[i] += hiddenDeltas[batch * HIDDEN_NEURONS + i];
			}
		}
	}

	void updateWeightsAndBiases()
	{
		// output layer
		for (int i = 0; i < HIDDEN_NEURONS; ++i)
			adam(outputWeights[i], outputWeightsM[i], outputWeightsV[i], outputWeightGradients[i]);
		adam(outputBias, outputBiasM, outputBiasV, outputBiasGradient);

		// hidden layer
		for (int i = 0; i < HIDDEN_NEURONS; ++i)
		{
			for (int j = 0; j < INPUT_NEURONS; ++j) {
				int idx = i * INPUT_NEURONS + j;
				adam(hiddenWeights[idx], hiddenWeightsM[idx], hiddenWeightsV[idx], hiddenWeightGradients[idx]);
			}

			adam(hiddenBiases[i], hiddenBiasesM[i], hiddenBiasesV[i], hiddenBiasGradients[i]);
		}
	}

	void train(vector<NetInput>& data, int epochs)
	{
		shuffleData(data);

		const int dataSize = data.size();
		const int valSize = dataSize / 100;
		int trainingSize = dataSize - valSize;

		// adjust trainingSize to be a multiple of BATCH_SIZE
		// or else the whole program will crash
		trainingSize = (trainingSize / BATCH_SIZE) * BATCH_SIZE;

		// calculate the number of batches
		const int numBatches = trainingSize / BATCH_SIZE;

		cout << "Training Network with " << trainingSize << " Positions!\n" << endl;

		vector<NetInput> valData(data.begin() + trainingSize, data.end());
		data.resize(trainingSize);

		auto startTime = chrono::high_resolution_clock::now();

		cout << left << setw(6) << "Epoch" << " | " << "Validation Loss" << endl;
		cout << "------------------------" << endl;

		for (int epoch = 1; epoch <= epochs; ++epoch)
		{
			for (int batch = 0; batch < numBatches; ++batch)
			{
				int startIdx = batch * BATCH_SIZE;
				int endIdx = min((batch + 1) * BATCH_SIZE, trainingSize);

				// feed forward
				vector<NetInput> batchData(data.begin() + startIdx, data.begin() + endIdx);
				feedForward(batchData, BATCH_SIZE);

				// feed backward
				vector<float> batchTargets;
				for (auto& d : batchData) {
					batchTargets.push_back(d.target);
				}
				feedBackward(batchTargets, BATCH_SIZE);

				// update weights and biases
				updateWeightsAndBiases();
			}

			if (epoch % 3 == 0) {
				cout << setw(6) << epoch << " | " << getLoss(valData) << endl;
			}
		}

		auto endTime = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime);
		cout << "\nTraining Neural Network done! (" << duration.count() << " seconds)\n" << endl;
	}

private:
	//hidden layer
	float* hiddenWeights, * hiddenWeightGradients, * hiddenWeightsM, * hiddenWeightsV;
	float* hiddenBiases, * hiddenBiasGradients, * hiddenBiasesM, * hiddenBiasesV;

	float* hiddenActivations, * hiddenWeightedInputs, * hiddenDeltas, * hiddenInput;

	float* outputWeights, * outputWeightGradients, * outputWeightsM, * outputWeightsV;
	float outputBias, outputBiasGradient, outputBiasM, outputBiasV;

	float* outputActivation, * outputWeightedInput, * outputDelta, * outputInput;

	static float relu(float x)
	{
		return x > 0 ? x : 0;
	}

	static float reluDer(float x)
	{
		return x > 0 ? 1 : 0;
	}

	static float sigmoid(float x)
	{
		return 1 / (1 + exp(-SIGMOID_SCALAR * x));
	}

	static float sigmoidDer(float x)
	{
		float s = sigmoid(x);
		return s * (1 - s) * SIGMOID_SCALAR;
	}

	void adam(float& param, float& m, float& v, float& grad)
	{
		if (false) {
			m = 0.95f * m + (1 - 0.95f) * grad;
			v = 0.999f * v + (1 - 0.999f) * grad * grad;
			param -= 0.01f * m / (sqrt(v) + 1e-8f);
		}
		else {
			param -= 0.01f * grad;
		}
	
		grad = 0.0f;
	}

	float getLoss(const vector<NetInput>& data)
	{
		float totalCost = 0;
		for (auto d : data)
		{
			float prediction = feedForward(getSparseInput(d));
			float error = prediction - d.target;
			totalCost += error * error;
		}

		return totalCost / data.size();
	}
};