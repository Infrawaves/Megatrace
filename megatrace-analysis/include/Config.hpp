#ifndef CONFIG_LOG_CONFIG
#define CONFIG_LOG_CONFIG
#include <iostream>
#include <vector>
struct TrainingProcess
{
    std::string name;
    size_t iteration;
    size_t startIdx;
    size_t endIdx;
    TrainingProcess(std::string name, size_t iteration, size_t startIdx, size_t endIdx);
};

struct TrainingConfig
{
    std::string inputFilePath;
    std::string outputDicPath;
    bool isSP;
    int layers;
    int ppSize;
    int tpSize;
    int dpSize;
    int GBS;
    int headers;
    int numRanks;
    size_t iterations;
    int tpGroupSize;
    int ppGroupSize;
    int dpGroupSize;
    double slowThreshold;
};

std::vector<std::vector<TrainingProcess>> gen_training_pattern(TrainingConfig config);

void _gen_training_pattern_withSP(int layerNumPerRank, std::vector<std::vector<TrainingProcess>> &trainingPatterns);

void _gen_training_pattern_withoutSP(int layerNumPerRank, std::vector<std::vector<TrainingProcess>> &trainingPatterns);

#endif