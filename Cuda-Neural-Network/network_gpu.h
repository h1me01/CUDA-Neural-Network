#include "device_launch_parameters.h"

#include "layer.h"
#include "kernels.h"
#include "misc.h"
#include <curand_kernel.h>

/*
 * HYPER PARAMETERS
 */
constexpr int INPUT_SIZE = 768;
constexpr int HIDDEN_SIZE = 64;
constexpr int OUTPUT_SIZE = 1;

constexpr float LR = 0.01f;
constexpr int BATCH_SIZE = 128;

// Adam Parameters
constexpr float BETA1 = 0.95f;
constexpr float BETA2 = 0.999f;
constexpr float EPSILON = 1e-8f;

/*
 * NETWORK CLASS
 */
class Network_GPU {
public:
	Network_GPU(const string& fileName = "") :
		hidden_layer(BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE),
		output_layer(BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
	{
		h_hidden_input = new float[BATCH_SIZE * INPUT_SIZE];

		h_hidden_weights = new float[HIDDEN_SIZE * INPUT_SIZE];
		h_output_weights = new float[HIDDEN_SIZE];

		h_hidden_biases = new float[HIDDEN_SIZE];
		h_output_biases = new float[OUTPUT_SIZE];

		// init the weights using he init
		heInit(h_hidden_weights, INPUT_SIZE, HIDDEN_SIZE);
		heInit(h_output_weights, HIDDEN_SIZE, OUTPUT_SIZE);

		// copy from host to device
		copyToDev(hidden_layer.weights, h_hidden_weights, HIDDEN_SIZE * INPUT_SIZE);
		copyToDev(output_layer.weights, h_output_weights, OUTPUT_SIZE * HIDDEN_SIZE);

		if (fileName != "")
			loadWeights(fileName);
	}

	~Network_GPU() {
		delete[] h_hidden_input;
		delete[] h_hidden_weights;
		delete[] h_hidden_biases;
		delete[] h_output_weights;
		delete[] h_output_biases;
	}

	float feed(float* input)
	{
		// hidden layer
		float hiddenOutput[HIDDEN_SIZE];
		for (int i = 0; i < HIDDEN_SIZE; ++i)
		{
			float dot = 0;
			for (int j = 0; j < INPUT_SIZE; ++j) {
				dot += input[j] * h_hidden_weights[INPUT_SIZE * i + j];
			}
			dot += h_hidden_biases[i];

			hiddenOutput[i] = dot > 0 ? dot : 0; // relu activation
		}

		// output layer
		float prediction = 0;
		for (int i = 0; i < HIDDEN_SIZE; ++i) {
			prediction += hiddenOutput[i] * h_output_weights[i];
		}
		prediction += h_output_biases[0];

		// don't forget to deallocate input
		delete[] input;

		return 1.0f / (1.0f + expf(-SIGMOID_SCALAR * prediction)); // sigmoid activation
	}

	void feed(vector<NetInput>& batchInput) {
		// prepare sparse input
		for (int batch = 0; batch < BATCH_SIZE; ++batch) {
			float* currentInput = getSparseInput(batchInput[batch]);

			for (int i = 0; i < INPUT_SIZE; ++i) {
				h_hidden_input[INPUT_SIZE * batch + i] = currentInput[i];
			}

			delete[] currentInput;
		}

		// copy host input to device input
		copyToDev(hidden_layer.inputs, h_hidden_input, BATCH_SIZE * INPUT_SIZE);

		// forward pass hidden layer
		feedKernel << <BATCH_SIZE, HIDDEN_SIZE >> > (
			hidden_layer.inputs,
			hidden_layer.weights,
			hidden_layer.biases,
			hidden_layer.weighted_inputs,
			hidden_layer.activations,
			INPUT_SIZE, HIDDEN_SIZE);
		checkKernelErrors("feedKernel - Hidden Layer");

		// forward pass output layer
		feedKernel << <BATCH_SIZE, OUTPUT_SIZE >> > (
			hidden_layer.activations, // hidden activations is input of output layer 
			output_layer.weights,
			output_layer.biases,
			output_layer.weighted_inputs,
			output_layer.activations,
			HIDDEN_SIZE, OUTPUT_SIZE);
		checkKernelErrors("feedKernel - Output Layer");
	}

	void backprop(vector<float> targets) {
		if (targets.size() != BATCH_SIZE) {
			cout << "Targets size must be equal to batch size! Target size: " << targets.size() << endl;
			exit(1);
		}

		// copy host targets to device targets
		copyToDev(output_layer.targets, targets.data(), BATCH_SIZE);

		// backward pass output layer
		backpropOutputKernel << <BATCH_SIZE, OUTPUT_SIZE >> > (
			output_layer.targets,
			output_layer.activations,
			hidden_layer.activations, // hidden activations is input of output layer
			output_layer.weighted_inputs,
			output_layer.weights_grad,
			output_layer.biases_grad,
			output_layer.deltas,
			HIDDEN_SIZE, OUTPUT_SIZE);
		checkKernelErrors("backpropOutputKernel");

		// backward pass hidden layer
		backpropHiddenKernel << <BATCH_SIZE, HIDDEN_SIZE >> > (
			hidden_layer.inputs,
			hidden_layer.weighted_inputs,
			hidden_layer.weights_grad,
			hidden_layer.biases_grad,
			output_layer.weights,
			output_layer.deltas,
			INPUT_SIZE, HIDDEN_SIZE);
		checkKernelErrors("backpropHiddenKernel");
	}

	void update(bool use_adam) {
		if (use_adam) {
			// update output layer
			adamKernel << <OUTPUT_SIZE, HIDDEN_SIZE >> > (
				output_layer.weights,
				output_layer.biases,
				output_layer.weights_grad,
				output_layer.biases_grad,
				output_layer.m_weights, output_layer.v_weights,
				output_layer.m_biases, output_layer.v_biases,
				BETA1, BETA2, EPSILON, LR,
				HIDDEN_SIZE, OUTPUT_SIZE);
			checkKernelErrors("Adam - Output Layer");

			// update hidden layer
			adamKernel << <HIDDEN_SIZE, INPUT_SIZE >> > (
				hidden_layer.weights,
				hidden_layer.biases,
				hidden_layer.weights_grad,
				hidden_layer.biases_grad,
				hidden_layer.m_weights, hidden_layer.v_weights,
				hidden_layer.m_biases, hidden_layer.v_biases,
				BETA1, BETA2, EPSILON, LR,
				INPUT_SIZE, HIDDEN_SIZE);
			checkKernelErrors("Adam - Hidden Layer");
		}
		else {
			// update output layer
			updateKernel << <OUTPUT_SIZE, HIDDEN_SIZE >> > (
				output_layer.weights,
				output_layer.biases,
				output_layer.weights_grad,
				output_layer.biases_grad,
				LR, HIDDEN_SIZE, OUTPUT_SIZE);
			checkKernelErrors("Update - Output Layer");

			// update hidden layer
			updateKernel << <HIDDEN_SIZE, INPUT_SIZE >> > (
				hidden_layer.weights,
				hidden_layer.biases,
				hidden_layer.weights_grad,
				hidden_layer.biases_grad,
				LR, INPUT_SIZE, HIDDEN_SIZE);
			checkKernelErrors("Update - Hidden Layer");
		}
	}

	void saveWeights(const string fileName) {
		copyWeightsToHost();

		ofstream file(fileName, ios::out | ios::binary);
		if (file.is_open()) {
			file.write((char*)h_hidden_weights, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
			file.write((char*)h_output_weights, HIDDEN_SIZE * sizeof(float));
			file.write((char*)h_hidden_biases, HIDDEN_SIZE * sizeof(float));
			file.write((char*)h_output_biases, OUTPUT_SIZE * sizeof(float));

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
			file.read((char*)h_hidden_weights, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
			file.read((char*)h_output_weights, HIDDEN_SIZE * sizeof(float));
			file.read((char*)h_hidden_biases, HIDDEN_SIZE * sizeof(float));
			file.read((char*)h_output_biases, OUTPUT_SIZE * sizeof(float));

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
		return feed(netInput) * 250 - 125;
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
				feed(batchData);

				// feed backward
				vector<float> batchTargets;
				for (auto& d : batchData) {
					batchTargets.push_back(d.target);
				}
				backprop(batchTargets);

				// update weights and biases
				update(true); 
			}

			if (epoch % 3 == 0) {
				cout << setw(6) << epoch << " | " << getLoss(valData) << endl;
			}

			if (epoch % 5 == 0) {
				//saveWeights(getWeightsFilePath(epoch));
			}
		}

		auto endTime = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime);
		cout << "\nTraining Neural Network done! (" << duration.count() << " seconds)\n" << endl;
	}

private:
	Layer hidden_layer, output_layer;

	// host variables
	float* h_hidden_input, * h_hidden_weights, * h_hidden_biases;
	float* h_output_weights, * h_output_biases;

	// host functions
	float getLoss(const vector<NetInput>& data)
	{
		copyWeightsToHost();

		float totalCost = 0;
		for (auto& d : data)
		{
			float prediction = feed(getSparseInput(d));
			float error = prediction - d.target;
			totalCost += error * error;
		}

		return totalCost / data.size();
	}

	void copyWeightsToDevice() {
		copyToDev(hidden_layer.weights, h_hidden_weights, HIDDEN_SIZE * INPUT_SIZE);
		copyToDev(hidden_layer.biases, h_hidden_biases, HIDDEN_SIZE);
		copyToDev(output_layer.weights, h_output_weights, HIDDEN_SIZE);
		copyToDev(output_layer.biases, h_output_biases, OUTPUT_SIZE);
	}

	void copyWeightsToHost() {
		copyFromDev(h_hidden_weights, hidden_layer.weights, HIDDEN_SIZE * INPUT_SIZE);
		copyFromDev(h_hidden_biases, hidden_layer.biases, HIDDEN_SIZE);
		copyFromDev(h_output_weights, output_layer.weights, HIDDEN_SIZE);
		copyFromDev(h_output_biases, output_layer.biases, OUTPUT_SIZE);
	}

	string getWeightsFilePath(int epoch) {
		return "C:/Users/semio/Downloads/Astra_Weights/astra_weights_" + to_string(epoch) + "_768-64-1.nnue";
	}
};
