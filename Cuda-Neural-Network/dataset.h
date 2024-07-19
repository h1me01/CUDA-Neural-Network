#pragma once

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <iomanip>

#define NUM_FEATURES (12 * 64)

using namespace std;

enum Color : int {
    WHITE, BLACK, NUM_COLORS = 2
};

struct NetInput {
    uint64_t pieces[NUM_COLORS][6]{};
    float target;
    Color stm;

    NetInput() {
        target = 0.0f;
    }
};

vector<NetInput> getNetData(const string& filePath, int dataSize);

float* getSparseInput(const NetInput& netInput);

vector<float> fenToInput(string& fen);

vector<float> normalizeTargets(vector<float>& targetValues, float minValue = -125, float maxValue = 125);
