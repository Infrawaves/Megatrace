#include "LogParser.hpp"
#include "Rank.hpp"
#include "GraphNode.hpp"
#include "Semaphore.hpp"
#include <iostream>
#include <fstream>
#include <regex>
#include <algorithm>
#include <mutex>
#include <thread>
#include <limits.h>
#include <climits>
#include <condition_variable>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <queue>
#include <sys/time.h>
#include <unistd.h>
#include <iomanip> 

NCCLLog::NCCLLog(double ts,
                 std::string &sid,
                 double lat,
                 int rank,
                 int sz,
                 int iter,
                 const std::string &func,
                 const std::string &proc)
    : timestamp(ts),
      streamID(sid),
      latency(lat),
      rankID(rank),
      size(sz),
      iteration(iter),
      ncclFunction(func),
      process(proc) {}

NCCLLog::NCCLLog()
    : timestamp(0),
      streamID("-1"),
      latency(0),
      rankID(0),
      size(0),
      iteration(-1),
      ncclFunction(""),
      process("") {}

std::mutex mtx;
bool terminateFlag = false;
std::atomic<bool> timeoutFlag(false);
std::atomic<int> terminatingNum(0);
std::vector<Iteration> iterations;
std::vector<int> iter_finished_state;
Semaphore sm(0);
Semaphore tm(0);
std::string subtractFromTimestamp(const std::string& timestampStr, const std::string& subtractStr) {
    // Find the decimal point positions
    size_t decimalPos1 = timestampStr.find('.');
    size_t decimalPos2 = subtractStr.find('.');
    
    // Split into integer and decimal parts
    std::string intPart1 = timestampStr.substr(0, decimalPos1);
    std::string decPart1 = (decimalPos1 != std::string::npos) ? timestampStr.substr(decimalPos1 + 1) : "0";
    
    // Pad decimal parts to same length
    while (decPart1.length() < 6) decPart1 += "0";
    
    // Perform subtraction on integer part
    std::string result;
    int carry = 0;
    int i = intPart1.length() - 1;
    int j = subtractStr.length() - 1;
    
    while (i >= 0 || j >= 0 || carry) {
        int digit1 = (i >= 0) ? intPart1[i] - '0' : 0;
        int digit2 = (j >= 0) ? subtractStr[j] - '0' : 0;
        
        int diff = digit1 - digit2 - carry;
        if (diff < 0) {
            diff += 10;
            carry = 1;
        } else {
            carry = 0;
        }
        
        result = std::to_string(diff) + result;
        i--;
        j--;
    }
    
    // Remove leading zeros
    result.erase(0, result.find_first_not_of('0'));
    if (result.empty()) result = "0";
    
    // Add decimal part back
    result += "." + decPart1;
    
    return result;
}


int parseLogs(const std::vector<std::string> &logs, std::vector<NCCLLog> &parsedLogs)
{
    std::regex logPattern(R"(\[(\d+\.?\d*)\]\s\[Rank\s(\d+)\]\sFun\s(\w+)\sData\s(\d+)\sstream\s(\w+))"); //  v3

    for (const std::string &log : logs)
    {
        std::smatch match;
        if (regex_search(log, match, logPattern))
        {
            NCCLLog entry;
            //entry.timestamp = stod(match[1].str());
            std::string timestampStr = match[1].str();
            std::string adjustedTimestamp = subtractFromTimestamp(timestampStr , "1735689600");
            entry.timestamp = std::stod(adjustedTimestamp);
            entry.rankID = stoi(match[2].str());
            entry.ncclFunction = "nccl" + match[3].str();

            entry.streamID = match[5].str();
            parsedLogs.push_back(entry);
        }
        else
        {
            std::cout << "parseLogs: log format error" << std::endl;
            return -1;
        }
    }
    return 0;
}

NCCLLog parseLog(const std::string &log)
{

    std::regex logPattern(R"(\[(\d+\.?\d*)\]\s\[Rank\s(\d+)\]\sFun\s(\w+)\sData\s(\d+)\sstream\s(\w+))"); //  v3
    std::smatch match;
    NCCLLog entry;
    if (regex_search(log, match, logPattern))
    {
        //entry.timestamp = stod(match[1].str());
        std::string timestampStr = match[1].str();
        std::string adjustedTimestamp = subtractFromTimestamp(timestampStr , "1735689600");
        entry.timestamp = std::stod(adjustedTimestamp);
        entry.rankID = stoi(match[2].str());
        entry.ncclFunction = "nccl" + match[3].str();
        entry.streamID = match[5].str();
    }
    else
    {
        std::cout << "parseLog: log format error" << std::endl;
    }
    return entry;
}

std::vector<std::string> readLogsFromFile(const std::string &filePath)
{
    std::vector<std::string> logs;
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filePath << std::endl;
        return logs;
    }

    std::string line;
    while (getline(file, line))
    {
        if (!line.empty())
        {
            logs.push_back(line);
        }
    }

    file.close();
    return logs;
}

std::unordered_map<std::string, std::vector<NCCLLog>> groupLogsByStream(const std::vector<NCCLLog> &logs)
{
    std::unordered_map<std::string, std::vector<NCCLLog>> groupedLogs;
    for (const auto &log : logs)
    {
        groupedLogs[log.streamID].push_back(log);
    }
    return groupedLogs;
}

std::string getTPStreamID(const std::string &filePath)
{
    std::vector<NCCLLog> parsedLogs;
    parseLogs(readLogsFromFile(filePath), parsedLogs);
    auto &&groupedLogs = groupLogsByStream(parsedLogs);
    size_t maxSize = 0;
    std::string maxKey = "-1";
    if (groupedLogs.size() == 0)
        return "-1";
    for (const auto &entry : groupedLogs)
    {
        if (entry.second.size() > maxSize)
        {
            maxSize = entry.second.size();
            maxKey = entry.first;
        }
    }
    return groupedLogs[maxKey][0].streamID;
}

std::string getDPStreamID(std::string filePath)
{
    std::vector<NCCLLog> parsedLogs;
    parseLogs(readLogsFromFile(filePath), parsedLogs);
    auto &&groupedLogs = groupLogsByStream(parsedLogs);
    size_t minSize = INT_MAX;
    std::string minKey = "-1";
    if (groupedLogs.size() == 0)
        return "-1";
    for (const auto &entry : groupedLogs)
    {
        if (entry.second.size() < minSize)
        {
            minSize = entry.second.size();
            minKey = entry.first;
        }
    }
    return groupedLogs[minKey][0].streamID;
}

void printLogs(const std::vector<NCCLLog> &logs)
{
    for (const auto &log : logs)
    {
        std::cout << "Timestamp: " << log.timestamp
                  << ", RankID: " << log.rankID
                  << ", NCCL Function: " << log.ncclFunction
                  << ", Size: " << log.size
                  << ", StreamID: " << log.streamID
                  << ", Iteration: " << log.iteration
                  << ", Process: " << log.process
                  << ", Latency: " << log.latency
                  << std::endl;
    }
}

void printGroupedLogs(const std::unordered_map<std::string, std::vector<NCCLLog>> &groupedLogs)
{
    for (const auto &[streamID, logs] : groupedLogs)
    {
        std::cout << "StreamID: " << streamID << " lines:" << logs.size() << std::endl;
        for (const auto &log : logs)
        {
            std::cout << "  Timestamp: " << log.timestamp
                      << ", RankID: " << log.rankID
                      << ", NCCL Function: " << log.ncclFunction
                      << ", Size: " << log.size
                      << ", Iteration: " << log.iteration
                      << std::endl;
        }
    }
}

int fetchLog(const std::string& filePath, std::streampos &lastPosition, NCCLLog &log)
{
    std::ifstream file;
    int count = 0;
    file.open(filePath, std::ios::in);
    while (true)
    {
        if (count++ == 1e5)
        {
            break;
        }
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file " << filePath << std::endl;
            continue;
        }

        file.seekg(lastPosition);
        std::string line;

        if (getline(file, line))
        {
            count = 0;
            lastPosition = file.tellg();
            log = parseLog(line);
            file.close();
            return 0;
        }
    }
    file.close();
    return -1;
}

void worker_withoutSP(const std::string &filePath, int workerID,
                      Rank rank, const TrainingConfig &config,
                      std::vector<TrainingProcess> trainingPattern)
{

    size_t iterCnt = 1;
    size_t processCnt = 0;

    std::streampos lastLogPosition = 0;
    std::vector<NCCLLog> logs;
    NCCLLog log;

    while (iterCnt <= config.iterations)
    {
        if (processCnt == trainingPattern.size())
            break;
        TrainingProcess cur_process = trainingPattern[processCnt];

        {
            std::lock_guard<std::mutex> lock(mtx);
            if (iterations.size() < iterCnt)
            {
                iterations.emplace_back(iterCnt, config.tpGroupSize, config.ppGroupSize, config.dpGroupSize, config.GBS, config.layers / config.ppSize, config.tpSize, config.ppSize, config.dpSize, config.numRanks);
            }
        }

        if (fetchLog(filePath, lastLogPosition, log))
        {
            break;
        }
        else
        {
            iterations[iterCnt - 1].historyLogs[workerID].push_back(log);
            logs.push_back(log);
        }
        logs.back().iteration = iterCnt;
        if (logs.size() > 1)
            logs[logs.size() - 2].latency = logs.back().timestamp - logs[logs.size() - 2].timestamp;
        else
            logs.back().latency = 0;

        if (iterCnt != cur_process.iteration)
        {
            std::lock_guard<std::mutex> lck(mtx);
            iter_finished_state[iterCnt - 1]++;
            if (iter_finished_state[iterCnt - 1] == config.numRanks)
            {
                iter_finished_state[iterCnt - 1] = -1;
            }
            iterCnt++;
        }

        if (logs.size() < cur_process.startIdx && processCnt != 0)
        { // 识别DP

            if (logs.back().ncclFunction == "ncclReduceScatter")
                iterations[iterCnt - 2].DP_info[rank.getDpGroup()].Rank_rs_time[rank.getDp()] = logs.back().timestamp;
            else if (logs.back().ncclFunction == "ncclAllGather" && iterations[iterCnt - 2].DP_info[rank.getDpGroup()].Rank_ag_time[rank.getDp()] == 0)
            {
                iterations[iterCnt - 2].DP_info[rank.getDpGroup()].Rank_ag_time[rank.getDp()] = logs.back().timestamp;
            }

            continue;
        }

        if (logs.size() >= cur_process.startIdx && logs.size() <= cur_process.endIdx)
        {
            logs.back().iteration = cur_process.iteration;
            logs.back().process = cur_process.name;

            if (logs.size() == cur_process.startIdx)
                iterations[iterCnt - 1].PP_info[rank.getPpGroup()].nodes[rank.getPp()].emplace_back(rank, logs.back().process, logs.back().iteration, logs.back().timestamp, 0);
            else if (logs.size() == cur_process.endIdx)
            {
                iterations[iterCnt - 1].PP_info[rank.getPpGroup()].nodes[rank.getPp()].back().endTime = logs.back().timestamp;
                double duration = iterations[iterCnt - 1].PP_info[rank.getPpGroup()].nodes[rank.getPp()].back().calDuration();
                processCnt++;
                iterations[iterCnt - 1].PP_info[rank.getPpGroup()].timecost_sum += duration;
            }
        }
    }
    std::cout << "rank:" << workerID << " finish" << std::endl;
    std::cout << config.outputDicPath << "/" << "ncclLog-rank-" << std::to_string(workerID) << ".txt" << std::endl;
    writeLogsToFile(config.outputDicPath + "/" + "ncclLog-rank-" + std::to_string(workerID) + ".txt", logs);

    if (terminatingNum.fetch_add(1, std::memory_order_acq_rel) + 1 == config.numRanks)
    {
        sm.Signal();
    }

    return;
}

void worker_withSP(const std::string &filePath, int workerID,
                   Rank rank, const TrainingConfig &config,
                   std::vector<TrainingProcess> trainingPattern)
{

    int iterCnt = 1;
    int processCnt = 0;

    std::streampos lastLogPosition = 0;
    std::vector<NCCLLog> logs;
    NCCLLog log;

    while (iterCnt <= config.iterations)
    {
        if (processCnt == trainingPattern.size())
            break;
        TrainingProcess cur_process = trainingPattern[processCnt];
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (iterations.size() < iterCnt)
            {
                iterations.emplace_back(iterCnt, config.tpGroupSize, config.ppGroupSize, config.dpGroupSize, config.GBS, config.layers / config.ppSize, config.tpSize, config.ppSize, config.dpSize, config.numRanks);
            }
        }

        if (fetchLog(filePath, lastLogPosition, log))
        {
            break;
        }

        else
        {
            iterations[iterCnt - 1].historyLogs[workerID].push_back(log);
            logs.push_back(log);
        }
        logs.back().iteration = iterCnt;
        if (logs.size() > 1)
            logs.back().latency = logs.back().timestamp - logs[logs.size() - 2].timestamp;
        else
            logs.back().latency = 0;

        if (iterCnt != cur_process.iteration)
        {
            std::lock_guard<std::mutex> lck(mtx);
            iter_finished_state[iterCnt - 1]++;
            if (iter_finished_state[iterCnt - 1] == config.numRanks)
            {
                // sm.Signal();
                iter_finished_state[iterCnt - 1] = -1;
            }
            iterCnt++;
        }

        if (logs.size() < cur_process.startIdx && processCnt != 0)
        { // 识别DP
            if (rank.id == 6)
            {
            }
            if (logs.back().ncclFunction == "ncclReduceScatter")
                iterations[iterCnt - 2].DP_info[rank.getDpGroup()].Rank_rs_time[rank.getDp()] = logs.back().timestamp;
            else if (logs.back().ncclFunction == "ncclAllGather" && iterations[iterCnt - 2].DP_info[rank.getDpGroup()].Rank_ag_time[rank.getDp()] == 0)
            {
                iterations[iterCnt - 2].DP_info[rank.getDpGroup()].Rank_ag_time[rank.getDp()] = logs.back().timestamp;
            }

            continue;
        }

        if (logs.size() >= cur_process.startIdx && logs.size() <= cur_process.endIdx)
        {
            logs.back().iteration = cur_process.iteration;
            logs.back().process = cur_process.name;

            if (logs.size() == cur_process.startIdx)
                iterations[iterCnt - 1].PP_info[rank.getPpGroup()].nodes[rank.getPp()].emplace_back(rank, logs.back().process, logs.back().iteration, logs.back().timestamp, 0);
            else if (logs.size() == cur_process.endIdx)
            {
                iterations[iterCnt - 1].PP_info[rank.getPpGroup()].nodes[rank.getPp()].back().endTime = logs.back().timestamp;
                iterations[iterCnt - 1].PP_info[rank.getPpGroup()].nodes[rank.getPp()].back().calDuration();
                processCnt++;
            }
        }
    }
    std::cout << config.outputDicPath << "/" << "ncclLog-rank-" << std::to_string(workerID) << ".txt" << std::endl;
    writeLogsToFile(config.outputDicPath + "/" + "ncclLog-rank-" + std::to_string(workerID) + ".txt", logs);

    if (terminatingNum.fetch_add(1, std::memory_order_acq_rel) + 1 == config.numRanks)
    {
        sm.Signal();
    }

    return;
}

void manager(const TrainingConfig &config)
{
    int count = 0;
    int microBatchNum = config.GBS / (config.numRanks / (config.tpSize * config.ppSize));
    PPTimeTable timetable(config.ppSize, microBatchNum);
    std::chrono::duration<double> total_duration = std::chrono::duration<double>::zero(); // 总时间
    int iteration_count = 0;
    sm.Wait();
    while (count != config.iterations)
    {
        //sm.Wait();
        Iteration &iteration = iterations[count]; //
        bool isHang = false;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iteration.PP_info.size(); i++)
        {
            Graph graph(iteration.iter, i);
            isHang |= graph.buildComputationGraph(config.slowThreshold, iteration.PP_info[i], iteration.DP_info, iteration.historyLogs, timetable, config.ppSize * microBatchNum * 2);
            if (!isHang)
            {
                graph.calculateCriticalPath();
                graph.checkSlow(iteration.historyLogs);
            }
            std::string path = config.outputDicPath + "/" + "graph-iteration" + std::to_string(iteration.iter) + "-ppGroup" + std::to_string(graph.groupID);

            graph.graphVisualization(path);
        }
        count++;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = end_time - start_time; // iteration duration
        total_duration += duration;            // total duration
        iteration_count++;                     

        if (iterations.size() <= count || isHang)
            break;
    }

    if (iteration_count > 0)
    {
        double average_duration = std::chrono::duration<double>(total_duration / iteration_count).count();
        std::cout << "Average iteration time: " << average_duration << " seconds" << std::endl;
        std::cout << "iteration_count: " << iteration_count << std::endl;
    }
    for (int i = 1; i <= config.numRanks; i++)
        tm.Signal();
}

void initParser(Rank *ranks, const TrainingConfig &config)
{
    std::unordered_map<int, std::thread> workerThread_map;
    iter_finished_state.resize(config.iterations);

    std::vector<std::vector<TrainingProcess>> trainingPatterns = gen_training_pattern(config);

    for (int i = 0; i < config.numRanks; i++)
    {
        std::string filePath = config.inputFilePath + "/" + "rank_" + std::to_string(i) + ".log";
        if (!config.isSP)
            workerThread_map[i] = std::move(std::thread(worker_withoutSP, filePath, i, ranks[i], config, trainingPatterns[ranks[i].getPp()]));
        else
            workerThread_map[i] = std::move(std::thread(worker_withSP, filePath, i, ranks[i], config, trainingPatterns[ranks[i].getPp()]));
    }

    std::thread managerThread(manager, config);

    for (auto &t : workerThread_map)
    {
        if (t.second.joinable())
        {
            t.second.join();
        }
    }

    if (managerThread.joinable())
    {
        managerThread.join();
    }
}

void writeLogsToFile(const std::string &filename, const std::vector<NCCLLog> &logs)
{

    std::ofstream outFile(filename, std::ios::out);

    if (!outFile.is_open())
    {
        outFile.open("." + filename);
        if (!outFile.is_open())
        {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
    }
    outFile << std::fixed << std::setprecision(9);
    for (const auto &log : logs)
    {
        outFile << "Timestamp: " << log.timestamp
                << ", RankID: " << log.rankID
                << ", NCCL Function: " << log.ncclFunction
                << ", Size: " << log.size
                << ", StreamID: " << log.streamID
                << ", Iteration: " << log.iteration
                << ", Process: " << log.process
                << ", Latency: " << log.latency
                << std::endl;
    }

    outFile.close();
}