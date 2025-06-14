#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include "LogParser.hpp"
#include "Rank.hpp"
#include "Config.hpp"
using namespace std;

// 64rank training set
// const int TP_SIZE = 2;
// const int PP_SIZE = 4;
// const int LAYERS = 32;
// const int GBS = 512;
// const int HEADS = 32;
// const int ITERATIONS = 50;
// const int N_WORKERS = 512;
// const bool isSP = false;

// pareseYamlFile function to read and parse the YAML configuration file
std::unordered_map<std::string, std::string> parseYamlFile(const std::string& filepath) {
    std::unordered_map<std::string, std::string> config;
    std::ifstream file(filepath);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file: " << filepath 
                  << ", using default values" << std::endl;
        return config;
    }

    while (std::getline(file, line)) {
        
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (line.empty() || line[0] == '#') continue;

        
        size_t colonPos = line.find(':');
        if (colonPos == std::string::npos) continue;

        std::string key = line.substr(0, colonPos);
        std::string value = line.substr(colonPos + 1);
       
        value.erase(std::remove(value.begin(), value.end(), '\"'), value.end());
        value.erase(std::remove(value.begin(), value.end(), '\''), value.end());

        config[key] = value;
    }

    return config;
}

// getConfigValue function template to retrieve values from the config map
template<typename T>
T getConfigValue(const std::unordered_map<std::string, std::string>& config, 
                const std::string& key, T defaultValue) {
    auto it = config.find(key);
    if (it != config.end()) {
        try {
            if constexpr (std::is_same_v<T, bool>) {
                std::string val = it->second;
                std::transform(val.begin(), val.end(), val.begin(), ::tolower);
                return val == "true" || val == "1";
            } else if constexpr (std::is_integral_v<T>) {
                return std::stoi(it->second);
            } else if constexpr (std::is_floating_point_v<T>) {
                return std::stod(it->second);
            } else if constexpr (std::is_same_v<T, std::string>) {
                return it->second;
            }
        } catch (...) {
            return defaultValue;
        }
    }
    return defaultValue;
}

void checkRanks(Rank *ranks, TrainingConfig config)
{
    for (int i = 0; i < config.numRanks; i++)
    {
        ranks[i].printRankInfo();
        cout << "===================" << endl;
    }
}

void printTrainingProcess(const TrainingProcess &tp)
{
    std::cout << "Training Process Information:" << std::endl;
    std::cout << "Name: " << tp.name << std::endl;
    std::cout << "Iteration: " << tp.iteration << std::endl;
    std::cout << "Start Index: " << tp.startIdx << std::endl;
    std::cout << "End Index: " << tp.endIdx << std::endl;
}

int main(int argc, char *argv[])
{
    string configFile = "config.yaml";

    if (argc != 3)
    {
        cerr << "Usage: " << argv[0] << " <log_file_path> " << "<output_file_path> " << endl;
        return 1;
    }

    string inputFilePath = argv[1];
    string outputDicPath = argv[2];
   // double slowThreshold = stod(argv[3]);

    Rank *ranks = nullptr;

    auto yamlConfig = parseYamlFile(configFile);
    TrainingConfig config = {
        .inputFilePath = inputFilePath,
        .outputDicPath = outputDicPath,
        .isSP = getConfigValue(yamlConfig, "isSP", false),
        .layers = getConfigValue(yamlConfig, "layers", 32),
        .ppSize = getConfigValue(yamlConfig, "ppSize", 4),
        .tpSize = getConfigValue(yamlConfig, "tpSize", 2),
        .GBS = getConfigValue(yamlConfig, "GBS", 512),
        .headers = getConfigValue(yamlConfig, "headers", 32),
        .numRanks = getConfigValue(yamlConfig, "numRanks", 512),
        .iterations = getConfigValue(yamlConfig, "iterations", 50),
        .slowThreshold = getConfigValue(yamlConfig, "slowThreshold", 1) 
    };
    // cout<<config.isSP<<endl;
    // cout<<config.layers<<endl;
    // cout<<config.ppSize<<endl;
    // cout<<config.tpSize<<endl;
    // cout<<config.GBS<<endl;
    // cout<<config.headers<<endl;
    // cout<<config.numRanks<<endl;
    // cout<<config.iterations<<endl;
    // cout<<config.slowThreshold<<endl;


    ranks = initRanks(config);

    std::vector<std::vector<TrainingProcess>> trainingPatterns = gen_training_pattern(config);

    initParser(ranks, config);

    releaseRanks(ranks, config);

    return 0;
}