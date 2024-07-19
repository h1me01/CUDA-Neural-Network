#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "dataset.h"

#define THREADS_PER_BLOCK 256

/*
 * TODO:
 *		 - maybe make loss and shuffle function faster
 *       - maybe make code more readable
 */

 // for cuda error handling 
static void HandleError(cudaError_t err,
	const char* file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void checkKernelErrors(const char* kernelName) {
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("CUDA Error after launching %s: %s\n", kernelName, cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error after launching %s: %s\n", kernelName, cudaGetErrorString(cudaStatus));
	}
}

/*
 * HYPER PARAMETERS
 */
constexpr int INPUT_NEURONS = NUM_FEATURES;
constexpr int HIDDEN_NEURONS = 64;
constexpr int OUTPUT_NEURONS = 1;

constexpr float LEARNING_RATE = 0.01f;
constexpr int BATCH_SIZE = 128;

/*
 * RANDOM NUMBER GENERATOR
 */
namespace Tools {
	random_device rd;
	mt19937 gen(42);
} // namespace Tools

/*
 * HOST FUNCTIONS
 */
string getWeightsFilePath(int epoch) {
	return "C:/Users/semio/Downloads/Astra_Weights/astra_weights_" + to_string(epoch) + "_768-64-1.nnue";
}

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

/*
 * DEVICE FUNCTIONS
 */

 // activation functions
__device__ float relu(float x) {
	return fmaxf(0.0f, x);
}

__device__ float sigmoid(float x) {
	return 1.0f / (1.0f + expf(-x));
}

// activation function derivatives
__device__ float sigmoidDer(float x) {
	float s = 1.0f / (1.0f + expf(-x));
	return s * (1 - s);
}

__device__ float reluDer(float x) {
	return x > 0 ? 1.0f : 0.0f;
}

/*
 * CUDA KERNEL FUNCTIONS
 */

__global__
void forwardPassKernel(
	float* batchInput, float* hiddenWeights, float* hiddenBiases, float* hiddenWeightedInputs, float* hiddenActivations,
	float* outputWeights, float* outputBias, float* outputWeightedInputs, float* outputActivations
) {
	int batch = blockIdx.x;
	int hiddenNeuron = threadIdx.x;

	if (batch >= BATCH_SIZE) return;

	__shared__ float sharedHiddenActivations[HIDDEN_NEURONS];
	__shared__ float sharedOutputWeights[HIDDEN_NEURONS];

	// Load output weights into shared memory
	if (hiddenNeuron < HIDDEN_NEURONS) {
		sharedOutputWeights[hiddenNeuron] = outputWeights[hiddenNeuron];
	}

	__syncthreads();

	// Calculate the hidden layer activations
	if (hiddenNeuron < HIDDEN_NEURONS) {
		float hiddenSum = 0.0f;
		for (int i = 0; i < INPUT_NEURONS; ++i) {
			hiddenSum += batchInput[batch * INPUT_NEURONS + i] * hiddenWeights[hiddenNeuron * INPUT_NEURONS + i];
		}
		hiddenSum += hiddenBiases[hiddenNeuron];
		hiddenWeightedInputs[batch * HIDDEN_NEURONS + hiddenNeuron] = hiddenSum;

		float activation = relu(hiddenSum);
		hiddenActivations[batch * HIDDEN_NEURONS + hiddenNeuron] = activation;
		sharedHiddenActivations[hiddenNeuron] = activation;
	}

	__syncthreads();

	// Calculate the output layer activations
	if (hiddenNeuron == 0) {
		float outputSum = 0.0f;
		for (int i = 0; i < HIDDEN_NEURONS; ++i) {
			outputSum += sharedHiddenActivations[i] * sharedOutputWeights[i];
		}
		outputSum += outputBias[0];

		outputWeightedInputs[batch] = outputSum;
		outputActivations[batch] = sigmoid(outputSum);
	}
}

// backward pass kernel for output layer 
__global__
void backwardOutputKernel(
	float* targets, float* batchInput, float* deltas, float* activations, float* weightedInputs, float* weightGradients, float* biasGradient
) {
	int batch = blockIdx.x;

	__shared__ float sharedDelta;
	__shared__ float sharedBiasGradient;

	if (threadIdx.x == 0) {
		sharedBiasGradient = 0.0f;
	}

	__syncthreads();

	if (batch < BATCH_SIZE) {
		if (threadIdx.x == 0) {
			sharedDelta = 2 * (activations[batch] - targets[batch]) * sigmoidDer(weightedInputs[batch]);
			deltas[batch] = sharedDelta;
		}

		__syncthreads();

		// update output weight gradients
		for (int i = threadIdx.x; i < HIDDEN_NEURONS; i += blockDim.x) {
			atomicAdd(&weightGradients[i], sharedDelta * batchInput[batch * HIDDEN_NEURONS + i]);
		}

		// update output bias gradient
		if (threadIdx.x == 0) {
			atomicAdd(&sharedBiasGradient, sharedDelta);
		}
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		atomicAdd(biasGradient, sharedBiasGradient);
	}
}

// cuda backward pass kernel for hidden layer
__global__
void backwardHiddenKernel(
	float* batchInput, float* outputDelta, float* outputWeights, float* weightedInputs, float* weightGradients, float* biasGradients
) {
	int batch = blockIdx.x;
	int neuron = threadIdx.x;

	__shared__ float sharedOutputDelta;
	__shared__ float sharedBatchInput[INPUT_NEURONS];

	if (batch < BATCH_SIZE && neuron < HIDDEN_NEURONS) {
		if (neuron == 0) {
			sharedOutputDelta = outputDelta[batch];
		}

		for (int i = neuron; i < INPUT_NEURONS; i += HIDDEN_NEURONS) {
			sharedBatchInput[i] = batchInput[batch * INPUT_NEURONS + i];
		}

		__syncthreads();

		float hiddenDelta = outputWeights[neuron] * sharedOutputDelta * reluDer(weightedInputs[batch * HIDDEN_NEURONS + neuron]);

		// update hidden weight gradients
		for (int i = 0; i < INPUT_NEURONS; ++i) {
			atomicAdd(&weightGradients[neuron * INPUT_NEURONS + i], sharedBatchInput[i] * hiddenDelta);
		}

		// update hidden bias gradients
		atomicAdd(&biasGradients[neuron], hiddenDelta);
	}
}

__global__
void updateWeightsAndBiasesKernel(
	float* outputWeights, float* outputBias, float* outputWeightGradients, float* outputBiasGradient,
	float* hiddenWeights, float* hiddenBiases, float* hiddenWeightGradients, float* hiddenBiasGradients,
	float lr
) {
	int hiddenNeuron = blockIdx.x;
	int inputNeuron = threadIdx.x;

	__shared__ float sharedOutputWeightGradients[HIDDEN_NEURONS];
	__shared__ float sharedOutputBiasGradient;

	// load output gradients into shared memory
	if (hiddenNeuron == 0 && inputNeuron < HIDDEN_NEURONS) {
		sharedOutputWeightGradients[inputNeuron] = outputWeightGradients[inputNeuron];
		if (inputNeuron == 0) {
			sharedOutputBiasGradient = outputBiasGradient[0];
		}
	}

	__syncthreads();

	// update output layer
	if (hiddenNeuron == 0) {
		if (inputNeuron < HIDDEN_NEURONS) {
			// update output weights
			outputWeights[inputNeuron] -= lr * sharedOutputWeightGradients[inputNeuron];
			outputWeightGradients[inputNeuron] = 0;
		}

		// update output bias (only once)
		if (inputNeuron == 0) {
			*outputBias -= lr * sharedOutputBiasGradient;
			outputBiasGradient[0] = 0;
		}
	}

	// update hidden layer
	if (hiddenNeuron < HIDDEN_NEURONS) {
		// update hidden biases
		if (inputNeuron == 0) {
			hiddenBiases[hiddenNeuron] -= lr * hiddenBiasGradients[hiddenNeuron];
			hiddenBiasGradients[hiddenNeuron] = 0;
		}

		// update hidden weights
		if (inputNeuron < INPUT_NEURONS) {
			int idx = hiddenNeuron * INPUT_NEURONS + inputNeuron;
			hiddenWeights[idx] -= lr * hiddenWeightGradients[idx];
			hiddenWeightGradients[idx] = 0;
		}
	}
}


/*
 * NETWORK CLASS
 */
class Network_GPU {
public:
	Network_GPU(const string& fileName = "") {
		// allocate device memory
		allocateHiddenLayer();
		allocateOutputLayer();

		// host 
		h_hiddenWeights = new float[HIDDEN_NEURONS * INPUT_NEURONS];
		h_outputWeights = new float[HIDDEN_NEURONS];

		h_hiddenBiases = new float[HIDDEN_NEURONS];
		h_outputBias = new float[OUTPUT_NEURONS];

		// init the weights using he init
		heInit(h_hiddenWeights, INPUT_NEURONS, HIDDEN_NEURONS);
		heInit(h_outputWeights, HIDDEN_NEURONS, OUTPUT_NEURONS);

		// copy from host to device
		HANDLE_ERROR(cudaMemcpy(d_hiddenWeights, h_hiddenWeights, HIDDEN_NEURONS * INPUT_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_outputWeights, h_outputWeights, OUTPUT_NEURONS * HIDDEN_NEURONS * sizeof(float), cudaMemcpyHostToDevice));

		if (fileName != "") {
			loadWeights(fileName);
		}
	}

	~Network_GPU() {
		// hidden layer
		HANDLE_ERROR(cudaFree(d_hiddenInput));

		HANDLE_ERROR(cudaFree(d_hiddenActivations));
		HANDLE_ERROR(cudaFree(d_hiddenWeightedInputs));

		HANDLE_ERROR(cudaFree(d_hiddenWeights));
		HANDLE_ERROR(cudaFree(d_hiddenWeightGradients));
		HANDLE_ERROR(cudaFree(d_hiddenBiases));
		HANDLE_ERROR(cudaFree(d_hiddenBiasGradients));

		// output layer
		HANDLE_ERROR(cudaFree(d_outputActivations));
		HANDLE_ERROR(cudaFree(d_outputWeightedInputs));
		HANDLE_ERROR(cudaFree(d_outputDeltas));

		HANDLE_ERROR(cudaFree(d_outputWeights));
		HANDLE_ERROR(cudaFree(d_outputWeightGradients));
		HANDLE_ERROR(cudaFree(d_outputBias));
		HANDLE_ERROR(cudaFree(d_outputBiasGradient));

		// host cleanup
		delete[] h_hiddenWeights;
		delete[] h_hiddenBiases;
		delete[] h_outputWeights;
		delete[] h_outputBias;
	}

	float feedForward(float* input)
	{
		// hidden layer
		float hiddenOutput[HIDDEN_NEURONS];
		for (int i = 0; i < HIDDEN_NEURONS; ++i)
		{
			float dot = 0;
			for (int j = 0; j < INPUT_NEURONS; ++j) {
				dot += input[j] * h_hiddenWeights[INPUT_NEURONS * i + j];
			}

			dot += h_hiddenBiases[i];
			hiddenOutput[i] = dot > 0 ? dot : 0; // relu activation
		}

		// output layer
		float prediction = 0;
		for (int i = 0; i < HIDDEN_NEURONS; ++i) {
			prediction += hiddenOutput[i] * h_outputWeights[i];
		}
		prediction += h_outputBias[0];

		// don't forget to deallocate input
		delete[] input;

		return 1.0f / (1.0f + expf(-prediction)); // sigmoid activation
	}

	void feedForward(vector<NetInput>& batchInput) {
		// allocate input on host and prepare sparse input
		float* h_input = new float[BATCH_SIZE * INPUT_NEURONS];
		for (int batch = 0; batch < BATCH_SIZE; ++batch) {
			float* currentInput = getSparseInput(batchInput[batch]);

			for (int i = 0; i < INPUT_NEURONS; ++i) {
				h_input[INPUT_NEURONS * batch + i] = currentInput[i];
			}

			delete[] currentInput;
		}

		// copy host input to device input
		HANDLE_ERROR(cudaMemcpy(d_hiddenInput, h_input, BATCH_SIZE * INPUT_NEURONS * sizeof(float), cudaMemcpyHostToDevice));

		forwardPassKernel << < BATCH_SIZE, HIDDEN_NEURONS >> > (
			d_hiddenInput, d_hiddenWeights, d_hiddenBiases, d_hiddenWeightedInputs, d_hiddenActivations,
			d_outputWeights, d_outputBias, d_outputWeightedInputs, d_outputActivations);
		checkKernelErrors("forwardPassKernel");

		// free host memory
		delete[] h_input;
	}

	void feedBackward(vector<float> targets) {
		if (targets.size() != BATCH_SIZE) {
			cout << "Targets size must be equal to batch size! Target size: " << targets.size() << endl;
			exit(1);
		}

		// allocate device memory for targets
		float* d_targets;
		HANDLE_ERROR(cudaMalloc(&d_targets, BATCH_SIZE * sizeof(float)));

		// copy host targets to device targets
		HANDLE_ERROR(cudaMemcpy(d_targets, targets.data(), BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice));

		backwardOutputKernel << < BATCH_SIZE, 1 >> > (
			d_targets, d_hiddenActivations, d_outputDeltas, d_outputActivations, d_outputWeightedInputs, d_outputWeightGradients, d_outputBiasGradient);
		checkKernelErrors("backwardOutputKernel");

		backwardHiddenKernel << < BATCH_SIZE, HIDDEN_NEURONS >> > (
			d_hiddenInput, d_outputDeltas, d_outputWeights, d_hiddenWeightedInputs, d_hiddenWeightGradients, d_hiddenBiasGradients);
		checkKernelErrors("backwardHiddenKernel");

		// free device memory
		cudaFree(d_targets);
	}

	void updateWeightsAndBiases() {
		updateWeightsAndBiasesKernel << <HIDDEN_NEURONS, INPUT_NEURONS >> > (
			d_outputWeights, d_outputBias, d_outputWeightGradients, d_outputBiasGradient,
			d_hiddenWeights, d_hiddenBiases, d_hiddenWeightGradients, d_hiddenBiasGradients, LEARNING_RATE);
		checkKernelErrors("updateWeightsAndBiasesKernel");
	}

	void saveWeights(const string fileName) {
		copyWeightsToHost();

		ofstream file(fileName, ios::out | ios::binary);
		if (file.is_open()) {
			file.write((char*)h_hiddenWeights, HIDDEN_NEURONS * INPUT_NEURONS * sizeof(float));
			file.write((char*)h_outputWeights, HIDDEN_NEURONS * sizeof(float));
			file.write((char*)h_hiddenBiases, HIDDEN_NEURONS * sizeof(float));
			file.write((char*)h_outputBias, OUTPUT_NEURONS * sizeof(float));

			file.close();
		}
		else {
			cout << "Error: Could not open file for writing!" << endl;
		}
	}

	void loadWeights(const string fileName) {
		ifstream fileCheck(fileName);
		if (!fileCheck.good()) {
			cout << "Error: File does not exist!" << endl;
			exit(1);
		}

		ifstream file(fileName, ios::in | ios::binary);
		if (file.is_open()) {
			file.read((char*)h_hiddenWeights, HIDDEN_NEURONS * INPUT_NEURONS * sizeof(float));
			file.read((char*)h_outputWeights, HIDDEN_NEURONS * sizeof(float));
			file.read((char*)h_hiddenBiases, HIDDEN_NEURONS * sizeof(float));
			file.read((char*)h_outputBias, OUTPUT_NEURONS * sizeof(float));

			file.close();
		}
		else {
			cout << "Error: Could not open file for reading!" << endl;
		}

		copyWeightsToDevice();
	}

	float evaluatePosition(string& fen) {
		vector<float> input = fenToInput(fen);
		float* netInput = new float[NUM_FEATURES];

		copy(input.begin(), input.end(), netInput);
		return feedForward(netInput) * 250 - 125;
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
				feedForward(batchData);

				// feed backward
				vector<float> batchTargets;
				for (auto& d : batchData) {
					batchTargets.push_back(d.target);
				}
				feedBackward(batchTargets);

				// update weights and biases
				updateWeightsAndBiases();
			}

			if (epoch % 3 == 0) {
				cout << setw(6) << epoch << " | " << getLoss(valData) << endl;
			}

			if (epoch % 5 == 0) {
				// save weights after each epoch
				saveWeights(getWeightsFilePath(epoch));
			}
		}

		auto endTime = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime);
		cout << "\nTraining Neural Network done! (" << duration.count() << " seconds)\n" << endl;
	}

private:
	// hidden layer
	float* d_hiddenInput, * d_hiddenActivations, * d_hiddenWeightedInputs;
	float* d_hiddenWeights, * d_hiddenWeightGradients;
	float* d_hiddenBiases, * d_hiddenBiasGradients;

	// output layer
	float* d_outputActivations, * d_outputWeightedInputs, * d_outputDeltas;
	float* d_outputWeights, * d_outputWeightGradients;
	float* d_outputBias, * d_outputBiasGradient;

	// host variables
	float* h_hiddenWeights, * h_hiddenBiases;
	float* h_outputWeights, * h_outputBias;

	// host functions
	float getLoss(const vector<NetInput>& data)
	{
		copyWeightsToHost();

		float totalCost = 0;
		for (auto& d : data)
		{
			float prediction = feedForward(getSparseInput(d));
			float error = prediction - d.target;
			totalCost += error * error;
		}

		return totalCost / data.size();
	}

	void allocateHiddenLayer() {
		HANDLE_ERROR(cudaMalloc(&d_hiddenInput, BATCH_SIZE * INPUT_NEURONS * sizeof(float)));

		size_t hiddenSize = BATCH_SIZE * HIDDEN_NEURONS * sizeof(float);
		HANDLE_ERROR(cudaMalloc(&d_hiddenActivations, hiddenSize));
		HANDLE_ERROR(cudaMalloc(&d_hiddenWeightedInputs, hiddenSize));

		size_t hiddenWeightsSize = HIDDEN_NEURONS * INPUT_NEURONS * sizeof(float);
		HANDLE_ERROR(cudaMalloc(&d_hiddenWeights, hiddenWeightsSize));
		HANDLE_ERROR(cudaMalloc(&d_hiddenWeightGradients, hiddenWeightsSize));

		size_t hiddenBiasesSize = HIDDEN_NEURONS * sizeof(float);
		HANDLE_ERROR(cudaMalloc(&d_hiddenBiases, hiddenBiasesSize));
		HANDLE_ERROR(cudaMalloc(&d_hiddenBiasGradients, hiddenBiasesSize));

		// init biases to zero
		HANDLE_ERROR(cudaMemset(d_hiddenBiases, 0, hiddenBiasesSize));

		// init gradients to zero
		HANDLE_ERROR(cudaMemset(d_hiddenWeightGradients, 0, hiddenWeightsSize));
		HANDLE_ERROR(cudaMemset(d_hiddenBiasGradients, 0, hiddenBiasesSize));
	}

	void allocateOutputLayer() {
		size_t outputSize = BATCH_SIZE * OUTPUT_NEURONS * sizeof(float);
		HANDLE_ERROR(cudaMalloc(&d_outputActivations, outputSize));
		HANDLE_ERROR(cudaMalloc(&d_outputWeightedInputs, outputSize));
		HANDLE_ERROR(cudaMalloc(&d_outputDeltas, outputSize));

		size_t outputWeightsSize = OUTPUT_NEURONS * HIDDEN_NEURONS * sizeof(float);
		HANDLE_ERROR(cudaMalloc(&d_outputWeights, outputWeightsSize));
		HANDLE_ERROR(cudaMalloc(&d_outputWeightGradients, outputWeightsSize));

		size_t outputBiasSize = OUTPUT_NEURONS * sizeof(float);
		HANDLE_ERROR(cudaMalloc(&d_outputBias, outputBiasSize));
		HANDLE_ERROR(cudaMalloc(&d_outputBiasGradient, outputBiasSize));

		// init biases to zero
		HANDLE_ERROR(cudaMemset(d_outputBias, 0, outputBiasSize));

		// init gradients to zero
		HANDLE_ERROR(cudaMemset(d_outputWeightGradients, 0, outputWeightsSize));
		HANDLE_ERROR(cudaMemset(d_outputBiasGradient, 0, outputBiasSize));
	}

	void copyWeightsToDevice() {
		HANDLE_ERROR(cudaMemcpy(d_hiddenWeights, h_hiddenWeights, HIDDEN_NEURONS * INPUT_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_hiddenBiases, h_hiddenBiases, HIDDEN_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_outputWeights, h_outputWeights, HIDDEN_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_outputBias, h_outputBias, OUTPUT_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
	}

	void copyWeightsToHost() {
		HANDLE_ERROR(cudaMemcpy(h_hiddenWeights, d_hiddenWeights, HIDDEN_NEURONS * INPUT_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(h_hiddenBiases, d_hiddenBiases, HIDDEN_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(h_outputWeights, d_outputWeights, HIDDEN_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(h_outputBias, d_outputBias, OUTPUT_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));
	}
};
