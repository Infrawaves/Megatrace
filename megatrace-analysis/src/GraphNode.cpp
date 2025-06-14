#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <math.h>
#include <regex>
#include <stack>
#include <iomanip>
#include <algorithm>
#include <climits>
#include "Rank.hpp"
#include "LogParser.hpp"
#include "GraphNode.hpp"

Node::Node()
    : rank(),
      processID(""),
      groupID(0),
      ppIndex(0),
      batchIndex(0),
      iteration(0),
      duration(0),
      startTime(0),
      endTime(0),
      isCriticalNode(false),
      isSlowNode(false) {}

Node::Node(Rank r, const std::string &id, int iter, double start, double end)
    : rank(r),
      processID(id),
      groupID(0),
      ppIndex(0),
      batchIndex(0),
      iteration(iter),
      duration(0),
      startTime(start),
      endTime(end),
      isCriticalNode(false),
      isSlowNode(false),
      isHangNode(false) {};

double Node::calDuration()
{
    duration = endTime - startTime < 0 ? 0 : endTime - startTime;
    return duration;
}

void Node::addCausalDependency(std::string id)
{
    causalDependencies.push_back(id);
}

TP_Rank_SP_info::TP_Rank_SP_info(int batch_size, int tp_index)
    : Rank_FW_ag_time(batch_size + 1, std::vector<double>(tp_index, 0)),
      Rank_FW_rs_time(batch_size + 1, std::vector<double>(tp_index, 0)),
      Rank_BW_ag1_time(batch_size + 1, std::vector<double>(tp_index, 0)),
      Rank_BW_ag2_time(batch_size + 1, std::vector<double>(tp_index, 0)),
      Rank_BW_rs_time(batch_size + 1, std::vector<double>(tp_index, 0)) {}

TP_Rank_info::TP_Rank_info(int batch_size, int layer, int tp_index)
    : Rank_FW_time(batch_size + 1,
                   std::vector<std::vector<double>>(layer * 2 + 1,
                                                    std::vector<double>(tp_index, 0))),
      Rank_BW_time(batch_size + 1,
                   std::vector<std::vector<double>>(layer * 2 + 1,
                                                    std::vector<double>(tp_index, 0))) {}

DP_Rank_info::DP_Rank_info(int dp_size)
    : Rank_ag_time(dp_size, 0),
      Rank_rs_time(dp_size, 0) {}

PP_Rank_info::PP_Rank_info(int pp_size, int batch_size)
    : nodes(pp_size),
      timecost_sum(0) {}

Iteration::Iteration(int iter_val, int TP_group_size, int PP_group_size, int DP_group_size,
                     int batch_size, int layer, int tp_size, int pp_size, int dp_size, int numRank)
    : iter(iter_val),
      TP_info(TP_group_size, TP_Rank_info(batch_size, layer, tp_size)),
      PP_info(PP_group_size, PP_Rank_info(pp_size, batch_size)),
      DP_info(DP_group_size, DP_Rank_info(dp_size)),
      historyLogs(numRank) {}

PPTimeTable::PPTimeTable(int pp_size, int batch_num)
    : ppGroup(pp_size),
      expectation(pp_size, std::vector<double>(batch_num * 2, 0)),
      data_num(pp_size, std::vector<int>(batch_num * 2, 0)) {}

PPTimeTable::PPTimeTable()
    : ppGroup(-1),
      expectation(),
      data_num() {}

bool PPTimeTable::isSlow(int ppIndex, int batchIndex, double timeCost, double threshold)
{
    double e = expectation[ppIndex][batchIndex];
    double val = (timeCost - e) / e;
    bool ret = val > threshold;
    return ret;
}

void PPTimeTable::updateTimeTable(int ppIndex, int batchIndex, double timeCost, int iterationNum)
{

    double &expectationValue = expectation[ppIndex][batchIndex];
    int &cnt = data_num[ppIndex][batchIndex];
    double X = timeCost;
    if (iterationNum == 2)
    {
        cnt++;
        expectationValue = X;
    }
    else if (iterationNum > 2)
    {
        cnt++;
        double oldExpectation = expectationValue;
        expectationValue += (X - oldExpectation) / cnt;
    }
}

Graph::Graph(int iteration, int groupID, int nodeNum, int edgeNum)
    : iteration(iteration), groupID(groupID), nodeNum(nodeNum), edgeNum(edgeNum) {}

Graph::Graph(int iteration, int groupID)
    : iteration(iteration), groupID(groupID), nodeNum(0), edgeNum(0) {}

Graph::Graph()
    : iteration(0), groupID(0), nodeNum(0), edgeNum(0) {}

void Graph::addNode(Node node)
{
    nodes[node.processID] = node;
    nodeNum++;
}

void Graph::addEdge(Node &startNode, Node &endNode)
{
    std::pair<std::string, std::string> key = std::make_pair(startNode.processID, endNode.processID);
    edges[key] = endNode.startTime - startNode.endTime;
}

double Graph::resizeNode(double duration, double minDuration, double maxDuration)
{
    if (minDuration == maxDuration)
        return 1.0;
    double logMin = log(minDuration + 1);
    double logMax = log(maxDuration + 1);
    double logDuration = log(duration + 1);

    return 3.0 + (logDuration - logMin) / (logMax - logMin) * (5.0 - 3.0);
}

void Graph::graphVisualization(std::string &outputFileName)
{
    std::ofstream dotFile(outputFileName + ".dot");

    if (!dotFile.is_open())
    {
        std::cerr << "Error: Could not open file for writing: " << outputFileName << std::endl;
        return;
    }
    double minDuration = LLONG_MAX;
    double maxDuration = LLONG_MIN;
    for (const auto &node : nodes)
    {
        if (node.second.duration < minDuration)
            minDuration = node.second.duration;
        if (node.second.duration > maxDuration)
            maxDuration = node.second.duration;
    }

    dotFile << "digraph G {" << std::endl;
    dotFile << "    rankdir=TB;" << std::endl;
    dotFile << "    node [style=filled];" << std::endl;
    dotFile << "    edge [minlen=4, penwidth=5.0];" << std::endl;

    std::map<int, std::vector<Node>> pp_index_map;
    for (auto node : nodes)
    {
        pp_index_map[node.second.ppIndex].push_back(node.second);
    }
    for (auto pp_nodes : pp_index_map)
    {
        for (const auto &node : pp_nodes.second)
        {

            std::string color = node.isCriticalNode ? "lightcyan" : "white";
            if (color == "lightcyan")
                color = node.isSlowNode ? "lightblue1" : color;
            else
                color = node.isSlowNode ? "grey" : color;

            if (node.processID == "endNode")
            {
                dotFile << "    \"" << node.processID << "\"" << ";" << std::endl;
                continue;
            }

            double size = resizeNode(node.duration, minDuration, maxDuration);
            int fontSize = (int)size * 12;

            std::string shape = node.isHangNode ? "doublecircle" : "circle";
            std::string penwidth = node.isHangNode ? "0.5" : "5.0";

            dotFile << "    \"" << node.processID << "\" [fillcolor=" << color
                    << ", width=" << size << ", height=" << size
                    << ", fontsize=" << fontSize
                    << ", penwidth=" << penwidth
                    << ", shape=\"" << shape << "\""
                    << ", fixedsize=true, label=\"" << node.processID
                    << "\\nduration: " << std::setiosflags(std::ios::fixed) << std::setprecision(3) << node.duration << "\\n"
                    << "rank: " << node.rank.id << "\"];" << std::endl;
        }
    }

    for (const auto &ppList : pp_index_map)
    {
        dotFile << "    { rank=same; ";
        for (const auto &node : ppList.second)
        {
            dotFile << "\"" << node.processID << "\"; ";
        }
        dotFile << "}" << std::endl;
    }

    for (const auto &edge : edges)
    {
        const auto &startNodeID = edge.first.first;
        const auto &endNodeID = edge.first.second;

        dotFile << "    \"" << startNodeID << "\" -> \"" << endNodeID << "\"";
        dotFile << ";" << std::endl;
    }

    dotFile << "}" << std::endl;
    dotFile.close();

    std::string command1 = "dot -Tsvg " + outputFileName + ".dot" + " -o " + outputFileName + ".svg";
    std::string command2 = "dot -Tpng " + outputFileName + ".dot" + " -o " + outputFileName + ".png";
    // system(command1.c_str());    //
    // system(command2.c_str());
}

bool Graph::buildComputationGraph(double threshold, PP_Rank_info pp_rank_info, std::vector<DP_Rank_info> dp_Info, std::vector<std::vector<NCCLLog>> historyLogs, PPTimeTable &timetable, int expectNodeNum)
{
    int m = pp_rank_info.nodes.size();
    timetable.ppGroup = groupID;
    std::unordered_set<std::string> st;
    bool isHang = false;
    for (int i = 0; i < m; i++)
    {
        int n = pp_rank_info.nodes[i].size();
        for (int j = 0; j < n; j++)
        {
            if (pp_rank_info.nodes[i].size() == 0)
                continue;
            Node &node = pp_rank_info.nodes[i][j];
            node.groupID = groupID;
            node.ppIndex = i;
            node.batchIndex = j;

            if (iteration > 3 && timetable.isSlow(node.ppIndex, node.batchIndex, node.duration, threshold))
            {
                node.isSlowNode = true;
                nodes[node.processID].isSlowNode = true;
            }
            if (!node.isSlowNode)
                timetable.updateTimeTable(node.ppIndex, node.batchIndex, node.duration, iteration);
            addNode(node);
            st.insert(nodes[node.processID].processID);
            if (i != m - 1 && nodes[node.processID].processID.find('B') == std::string::npos && pp_rank_info.nodes[i + 1].size() != 0)
            {
                if (j + 1 < pp_rank_info.nodes[i + 1].size() && pp_rank_info.nodes[i + 1][j].processID.find('B') != std::string::npos)
                {
                    nodes[node.processID].addCausalDependency(pp_rank_info.nodes[i + 1][j + 1].processID);
                }
                else if (pp_rank_info.nodes[i + 1][j].processID.find('B') == std::string::npos && j < pp_rank_info.nodes[i + 1].size())
                {
                    nodes[node.processID].addCausalDependency(pp_rank_info.nodes[i + 1][j].processID);
                }
            }
            if (j != n - 1)
            {
                nodes[node.processID].addCausalDependency(pp_rank_info.nodes[i][j + 1].processID);
            }
            if (i != 0 && nodes[node.processID].processID.find('B') != std::string::npos)
            {
                auto &&nextBackwardID = incrementNodeID(nodes[node.processID].processID);
                if (st.find(nextBackwardID) != st.end())
                {
                    nodes[node.processID].addCausalDependency(nextBackwardID);
                }
            }
        }
    }

    if (nodeNum == expectNodeNum)
    {
        Node endNode;
        endNode.processID = "endNode";
        addNode(endNode);
        for (int i = 0; i < m; i++)
        {
            Node node = pp_rank_info.nodes[i].back();
            Rank rk = node.rank;
            double rs_time = dp_Info[rk.getDpGroup()].Rank_rs_time[rk.getDp()];
            double ag_time = dp_Info[rk.getDpGroup()].Rank_ag_time[rk.getDp()];
            if (rk.id == 6)
            {
            }
            if (rs_time != 0)
            {
                Node dpNode(rk, "DP" + std::to_string(i), iteration, rs_time, ag_time);
                dpNode.ppIndex = node.rank.getPp();
                nodes[node.processID].addCausalDependency(dpNode.processID);
                if (ag_time != 0)
                {
                    dpNode.addCausalDependency(endNode.processID);
                }
                else
                {
                    isHang = true;
                }
                dpNode.addCausalDependency(endNode.processID);
                dpNode.calDuration();
                dpNode.duration = 0;
                addNode(dpNode);
                pp_rank_info.nodes[i].push_back(dpNode);
            }
        }
        expectNodeNum += pp_rank_info.nodes.size() + 1;
    }
    if (nodeNum != expectNodeNum)
        isHang = true;
    if (isHang)
    {

        Rank hangRank;
        std::string hangProcess;
        for (auto &it : nodes)
        {
            if (it.second.processID != "endNode" && it.second.duration == 0)
            {
                it.second.isHangNode = true;
                hangRank = it.second.rank;
                hangProcess = it.second.processID;
                break;
            }
        }
        if (hangProcess != "")
        {
            std::vector<NCCLLog> logs = historyLogs[hangRank.id];
            std::cout << "TYPE: hang, RANK: " << hangRank.id << ", " << "ITERATION: " << iteration << ", " << "PROCESS: " << hangProcess << ", "
                      << "FUNCTION: " << logs.back().ncclFunction << ", LATENCY: -1" << ", ISCRITICAL: 0" << std::endl;
        }
    }

    for (auto it : nodes)
    {
        for (auto endNodeID : it.second.causalDependencies)
            addEdge(it.second, nodes[endNodeID]);
    }
    return isHang;
}

std::string Graph::incrementNodeID(const std::string &nodeID)
{
    size_t pos = nodeID.find('B');

    std::string prefix = nodeID.substr(0, pos + 1);
    std::string suffix = nodeID.substr(pos + 1);

    int num = stoi(suffix);
    num++;
    return prefix + std::to_string(num);
}

void Graph::calculateCriticalPath()
{
    std::unordered_map<std::string, double> earliestStart, latestStart;
    std::unordered_map<std::string, int> inDegree;
    std::queue<std::string> topoQueue;
    std::stack<std::string> reverseTopoOrder;
    double maxDuration = 0;

    // initialize earliest start times
    for (const auto &[id, node] : nodes)
    {
        inDegree[id] = 0;
    }
    for (const auto &[id, node] : nodes)
    {
        for (const auto &dep : node.causalDependencies)
        {
            inDegree[dep]++;
        }
    }

    // sort nodes 
    for (const auto &[id, node] : nodes)
    {
        if (inDegree[id] == 0)
        {
            topoQueue.push(id);
            break;
        }
    }

    while (!topoQueue.empty())
    {
        std::string current = topoQueue.front();
        topoQueue.pop();
        reverseTopoOrder.push(current);

        for (const auto &neighbor : nodes[current].causalDependencies)
        {
            inDegree[neighbor]--;
            if (inDegree[neighbor] == 0)
            {
                topoQueue.push(neighbor);
            }
            // update earliest start time
            earliestStart[neighbor] = std::max(earliestStart[neighbor], earliestStart[current] + nodes[current].duration);
        }
    }

    // get maximum duration from earliest start times
    for (const auto &[id, time] : earliestStart)
    {
        maxDuration = std::max(maxDuration, time + nodes[id].duration);
    }

    // initialize latest start times
    for (const auto &[id, node] : nodes)
    {
        latestStart[id] = maxDuration - node.duration;
    }

    // sort nodes in reverse topological order
    while (!reverseTopoOrder.empty())
    {
        std::string current = reverseTopoOrder.top();
        reverseTopoOrder.pop();

        for (const auto &neighbor : nodes[current].causalDependencies)
        {
            latestStart[current] = std::min(latestStart[current], latestStart[neighbor] - nodes[current].duration);
        }
    }

    // mark critical nodes
    for (const auto &[id, node] : nodes)
    {
        if (earliestStart[id] == latestStart[id])
        {
            nodes[id].isCriticalNode = true;
        }
    }
}

void Graph::checkSlow(std::vector<std::vector<NCCLLog>> historyLogs)
{
    for (auto &it : nodes)
    {
        if (it.second.isSlowNode)
        {
            std::string ncclFunction = "";
            double maxSize = LONG_MIN;
            std::vector<NCCLLog> &logs = historyLogs[it.second.rank.id];
            for (int i = 1; i < logs.size(); i++)
            {
                double latency = logs[i].timestamp - logs[i - 1].timestamp;
                if (maxSize < latency)
                {
                    maxSize = latency;
                    ncclFunction = logs[i - 1].ncclFunction;
                }
            }
            std::cout << "TYPE: slow, RANK: " << it.second.rank.id << ", " << "ITERATION: " << iteration << ", " << "PROCESS: " << it.second.processID << ", FUNCTION: " << ncclFunction << ", LATENCY: " << maxSize << ", ISCRITICAL: " << it.second.isCriticalNode << std::endl;
        }
    }
}

std::string Graph::extractPrefixRegex(const std::string &input)
{
    std::regex pattern(R"(^(\d*)([A-Za-z]+))");
    std::smatch match;
    regex_search(input, match, pattern);

    if (match[2] == "DP")
        return "DataParalle";
    std::string batch = match[1].str();
    std::string op;

    if (match[2] == "F")
        op = "Forward";
    else if (match[2] == "B")
        op = "Backward";

    return "Batch: " + batch + "\n" + op;
}