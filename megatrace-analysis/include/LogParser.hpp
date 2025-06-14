#ifndef CONFIG_NCCL_LOG
#define CONFIG_NCCL_LOG
#include <vector>
#include <string>
#include <unordered_map>
#include "Config.hpp"
struct NCCLLog
{
    double timestamp;
    std::string streamID;
    double latency;
    int rankID;
    int size;
    int iteration;
    std::string ncclFunction;
    std::string process;
    NCCLLog(double ts, std::string &sid, double lat, int rank, int sz, int iter, const std::string &func, const std::string &proc);

    NCCLLog();
};

int fetchLog(const std::string &filePath, std::streampos &lastPosition, NCCLLog &log);

int parseLogs(const std::vector<std::string> &logs);

NCCLLog parseLog(const std::string &log);

std::unordered_map<std::string, std::vector<NCCLLog>> groupLogsByStream(const std::vector<NCCLLog> &logs);

void printLogs(const std::vector<NCCLLog> &logs);

void printGroupedLogs(const std::unordered_map<std::string, std::vector<NCCLLog>> &groupedLogs);

void writeLogsToFile(const std::string& filename, const std::vector<NCCLLog> &logs);

std::vector<std::string> readLogsFromFile(const std::string &filePath);

std::string getTPStreamID(const std::string& filePath);

std::string getDPStreamID(const std::string& filePath);

#include "Rank.hpp"
void initParser(Rank *ranks, const TrainingConfig& config);
void worker_withoutSP(const std::string& filePath, int workerID, 
                        Rank rank, const TrainingConfig& config, 
                        std::vector<TrainingProcess> trainingPattern);
void worker_withSP(const std::string& filePath, int workerID, 
                    Rank rank, const TrainingConfig& config, 
                    std::vector<TrainingProcess> trainingPattern);
void manager(const TrainingConfig& config);
#endif
