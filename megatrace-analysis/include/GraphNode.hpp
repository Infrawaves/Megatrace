#ifndef CONFIG_LOG_GRAPH
#define CONFIG_LOG_GRAPH
#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include "Rank.hpp"
#include "LogParser.hpp"

struct TP_Rank_SP_info
{
    std::vector<std::vector<double>> Rank_FW_ag_time;
    std::vector<std::vector<double>> Rank_FW_rs_time;
    std::vector<std::vector<double>> Rank_BW_ag1_time;
    std::vector<std::vector<double>> Rank_BW_ag2_time;
    std::vector<std::vector<double>> Rank_BW_rs_time;

    TP_Rank_SP_info(int batch_size, int tp_index);
};

struct TP_Rank_info
{
    std::vector<std::vector<std::vector<double>>> Rank_FW_time;
    std::vector<std::vector<std::vector<double>>> Rank_BW_time;

    TP_Rank_info(int batch_size, int layer, int tp_index);
};

struct DP_Rank_info
{
    std::vector<double> Rank_ag_time;
    std::vector<double> Rank_rs_time;

    DP_Rank_info(int dp_size);
};

struct Node
{
    Rank rank;
    std::string processID;
    int groupID;
    int ppIndex;
    int batchIndex;
    int iteration;
    double duration;
    double startTime;
    double endTime;
    bool isCriticalNode;
    bool isSlowNode;
    bool isHangNode;
    std::vector<std::string> causalDependencies;

    double calDuration();

    void addCausalDependency(std::string id);

    Node(Rank r, const std::string &id, int iter,
         double start, double end);

    Node();
};

struct PP_Rank_info
{
    std::vector<std::vector<Node>> nodes;
    double timecost_sum;
    PP_Rank_info(int pp_size, int batch_size);
};

struct Iteration
{
    int iter;
    std::vector<TP_Rank_info> TP_info;
    std::vector<PP_Rank_info> PP_info;
    std::vector<DP_Rank_info> DP_info;
    std::vector<std::vector<NCCLLog>> historyLogs;

    Iteration(int iter_val, int TP_group_size, int PP_group_size, int DP_group_size,
              int batch_size, int layer, int tp_size, int pp_size, int dp_size, int numRank);
};

struct PPTimeTable
{
    int ppGroup;
    std::vector<std::vector<double>> expectation;
    std::vector<std::vector<int>> data_num;
    PPTimeTable(int pp_size, int batch_num);

    PPTimeTable();

    bool isSlow(int ppIndex, int batchIndex, double timeCost, double threshold);

    void updateTimeTable(int ppIndex, int batchIndex, double timeCost, int iterationNum);
};

struct Graph
{
    int iteration;
    int groupID;
    int nodeNum;
    int edgeNum;
    std::unordered_map<std::string, Node> nodes;
    std::map<std::pair<std::string, std::string>, double> edges;

    Graph(int iteration, int groupID, int nodeNum, int edgeNum);

    Graph(int iteration, int groupID);

    Graph();

    void addNode(Node node);

    void addEdge(Node &startNode, Node &endNode);

    double resizeNode(double duration, double minDuration, double maxDuration);

    void graphVisualization(std::string &outputFileName);

    bool buildComputationGraph(double threshold, PP_Rank_info pp_rank_info, std::vector<DP_Rank_info> dp_Info, std::vector<std::vector<NCCLLog>> historyLogs, PPTimeTable &timetable, int expectNodeNum);

    std::string incrementNodeID(const std::string &nodeID);

    std::string extractPrefixRegex(const std::string &input);

    void calculateCriticalPath();

    void checkSlow(std::vector<std::vector<NCCLLog>> historyLogs);
};

#endif