#include "dataset.h"

/*
 * HELPER FUNCTIONS
 */
int pieceIndex(char c) {
    const string pieces = "pnbrqk";
    return pieces.find(tolower(c));
}

int mirrorVertically(int sq) {
    return sq ^ 56;
}

int index(int psq, char p, Color view) {
    if (view != WHITE)
        psq = mirrorVertically(psq);

    Color pc = isupper(p) ? WHITE : BLACK;
    return psq + 64 * pieceIndex(p) + (pc != view) * 64 * 6;
}

int index(int psq, int pt, Color pc, Color view) {
    if (view != WHITE)
        psq = mirrorVertically(psq);

    return psq + 64 * pt + (pc != view) * 64 * 6;
}

const int DEBRUIJN64[64] = {
        0, 47, 1, 56, 48, 27, 2, 60,
        57, 49, 41, 37, 28, 16, 3, 61,
        54, 58, 35, 52, 50, 42, 21, 44,
        38, 32, 29, 23, 17, 11, 4, 62,
        46, 55, 26, 59, 40, 36, 15, 53,
        34, 51, 20, 43, 31, 22, 10, 45,
        25, 39, 14, 33, 19, 30, 9, 24,
        13, 18, 8, 12, 7, 6, 5, 63
};

int popLsb(uint64_t& b) {
    int lsb = DEBRUIJN64[0x03f79d71b4cb0a89 * (b ^ b - 1) >> 58];
    b &= b - 1;
    return lsb;
}

/*
 * PUBLIC FUNCTIONS
 */
vector<NetInput> getNetData(const string& filePath, int dataSize) {
    vector<NetInput> netData;
    ifstream file(filePath, ios::binary);

    if (!file) {
        cerr << "Error: Unable to open file for reading.\n";
        return netData;
    }

    NetInput netInput;
    while (file.read(reinterpret_cast<char*>(&netInput), sizeof(NetInput))) {
        netData.push_back(netInput);
        if (netData.size() >= dataSize) break;
    }

    return netData;
}

float* getSparseInput(const NetInput& netInput) {
    auto* sparseInput = new float[NUM_FEATURES] {};

    for (int i = 0; i < NUM_COLORS; ++i) {
        for (int j = 0; j < 6; ++j) {
            uint64_t piece = netInput.pieces[i][j];
            while (piece) {
                int sq = popLsb(piece);
                int idx = index(sq, j, (Color)i, netInput.stm);

                sparseInput[idx] = 1.0f;
            }
        }
    }

    return sparseInput;
}

vector<float> fenToInput(string& fen) {
    vector<float> input(NUM_FEATURES, 0);
    Color stm = fen.find('w') != string::npos ? WHITE : BLACK;

    int rank = 7, file = -1;
    for (char c : fen) {
        if (c == ' ') break;
        if (c == '/') {
            rank--;
            file = -1;
        }
        else if (isdigit(c)) {
            file += c - '0';
        }
        else {
            file++;
            int sq = 8 * rank + file;
            int idx = index(sq, c, stm);

            input[idx] = 1;
        }
    }

    return input;
}

vector<float> normalizeTargets(vector<float>& targetValues, float minValue, float maxValue) {
    vector<float> normalizedValues;
    for (float value : targetValues) {
        float normalizedValue = (value - minValue) / (maxValue - minValue);
        normalizedValues.push_back(normalizedValue);
    }

    return normalizedValues;
}
