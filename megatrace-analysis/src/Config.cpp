#include <iostream>
#include <vector>
#include "Config.hpp"

TrainingProcess::TrainingProcess(std::string name, size_t iteration, size_t startIdx, size_t endIdx)
    : name(name),
      iteration(iteration),
      startIdx(startIdx),
      endIdx(endIdx) {}

std::vector<std::vector<TrainingProcess>> gen_training_pattern(TrainingConfig config)
{
    int microbsz = config.GBS / ((config.numRanks) / (config.tpSize * config.ppSize));
    int processNum = microbsz * 2;
    int layerNumPerRank = config.layers / config.ppSize;

    std::vector<std::vector<TrainingProcess>> trainingPatterns(config.ppSize);


    for (int ppIdx = 0; ppIdx < config.ppSize; ppIdx++)
    {
        int warmUpSize = config.ppSize - ppIdx;

        int iterCnt = 1;
        while (iterCnt <= config.iterations)
        {
            int forwardProcessIdx = 1;
            int backwardProcessIdx = 1;

            while (forwardProcessIdx <= warmUpSize)
            {
                trainingPatterns[ppIdx].emplace_back(std::to_string(forwardProcessIdx) + "F" + std::to_string(ppIdx), iterCnt, -1, -1);
                forwardProcessIdx++;
            }

            for (int i = 1; i <= processNum / 2 - warmUpSize; i++)
            {
                trainingPatterns[ppIdx].emplace_back(std::to_string(backwardProcessIdx) + "B" + std::to_string(config.ppSize - ppIdx - 1), iterCnt, -1, -1);
                backwardProcessIdx++;
                trainingPatterns[ppIdx].emplace_back(std::to_string(forwardProcessIdx) + "F" + std::to_string(ppIdx), iterCnt, -1, -1);
                forwardProcessIdx++;
            }

            while (backwardProcessIdx <= processNum / 2)
            {
                trainingPatterns[ppIdx].emplace_back(std::to_string(backwardProcessIdx) + "B" + std::to_string(config.ppSize - ppIdx - 1), iterCnt, -1, -1);
                backwardProcessIdx++;
            }

            iterCnt++;
        }
    }

    if (config.isSP)
        _gen_training_pattern_withSP(layerNumPerRank, trainingPatterns);
    else
        _gen_training_pattern_withoutSP(layerNumPerRank, trainingPatterns);

    return trainingPatterns;
}

void _gen_training_pattern_withSP(int layerNumPerRank, std::vector<std::vector<TrainingProcess>> &trainingPatterns)
{
    int uselessLogEndIdx = 16;  // hard code
    int batchFinishLogSize = 8; // hard code 

    int first_pp_forward_log_size = 5 + layerNumPerRank * 4; // 3 broadcast + 1 allreduce + layerNum*2 allgather&reducescatter + send
    int normal_pp_forward_log_size = 2 + layerNumPerRank * 4;
    int last_pp_forward_log_size = 8 + layerNumPerRank * 4; //  recv + 3 broadcast + layerNum*2 allgather&reducescatter + 1 allgather + 3 allreduce

    int first_pp_backward_log_size = 2 + layerNumPerRank * 6; //  recv + layerNum*2 allgather&allgather&reducescatter  + 1 allgather
    int normal_pp_backward_log_size = 2 + layerNumPerRank * 6;
    int last_pp_backward_log_size = 3 + layerNumPerRank * 6; //  1 allgather + 1 reducescatter + layerNum*2 allgather&allgather&reducescatter + send

    for (size_t ppIdx = 0; ppIdx < trainingPatterns.size(); ppIdx++)
    {
        int startIdx = uselessLogEndIdx + 1;
        int iterCnt = 1;
        for (size_t i = 0; i < trainingPatterns[ppIdx].size(); i++)
        {
            int forward_log_size = normal_pp_forward_log_size;
            int backward_log_size = normal_pp_backward_log_size;

            if (ppIdx == 0)
            {
                forward_log_size = first_pp_forward_log_size;
                backward_log_size = first_pp_backward_log_size;
            }
            else if (ppIdx == trainingPatterns.size() - 1)
            {
                forward_log_size = last_pp_forward_log_size;
                backward_log_size = last_pp_backward_log_size;
            }

            if (iterCnt != trainingPatterns[ppIdx][i].iteration)
            {
                startIdx += batchFinishLogSize;
                iterCnt++;
            }
            trainingPatterns[ppIdx][i].startIdx = startIdx;

            if (trainingPatterns[ppIdx][i].name.find("F") != std::string::npos)
            {
                trainingPatterns[ppIdx][i].endIdx = startIdx + forward_log_size - 1;
            }
            else if (trainingPatterns[ppIdx][i].name.find("B") != std::string::npos)
            {
                trainingPatterns[ppIdx][i].endIdx = startIdx + backward_log_size - 1;
            }
            else
            {
                std::cout << "error in gen_training_pattern_withSP" << std::endl;
            }

            startIdx = trainingPatterns[ppIdx][i].endIdx + 1;
        }
    }
}

void _gen_training_pattern_withoutSP(int layerNumPerRank, std::vector<std::vector<TrainingProcess>> &trainingPatterns)
{
    int uselessLogEndIdx = 16; //  hard code
    int batchFinishLogSize = 7; //  hard code

    int first_pp_forward_log_size = 5 + layerNumPerRank * 2; // 3 broadcast + 1 allreduce + layerNum*2 allreduce + 1 send
    int normal_pp_forward_log_size = 2 + layerNumPerRank * 2;
    int last_pp_forward_log_size = 8 + layerNumPerRank * 2; //  1 recv + 3 broadcast + layerNum*2 allreduce + 4 allreduce

    int first_pp_backward_log_size = 1 + layerNumPerRank * 2; //  1 recv + layerNum*2 allreduce
    int normal_pp_backward_log_size = 2 + layerNumPerRank * 2;
    int last_pp_backward_log_size = 2 + layerNumPerRank * 2; //  1 allreduce + layerNum*2 allreduce +  1 send

    for (size_t ppIdx = 0; ppIdx < trainingPatterns.size(); ppIdx++)
    {
        int startIdx = uselessLogEndIdx + 1;
        int iterCnt = 1;
        for (size_t i = 0; i < trainingPatterns[ppIdx].size(); i++)
        {
            int forward_log_size = normal_pp_forward_log_size;
            int backward_log_size = normal_pp_backward_log_size;

            if (ppIdx == 0)
            {
                forward_log_size = first_pp_forward_log_size;
                backward_log_size = first_pp_backward_log_size;
            }
            else if (ppIdx == trainingPatterns.size() - 1)
            {
                forward_log_size = last_pp_forward_log_size;
                backward_log_size = last_pp_backward_log_size;
            }

            if (iterCnt != trainingPatterns[ppIdx][i].iteration)
            {
                startIdx += batchFinishLogSize;
                iterCnt++;
            }
            trainingPatterns[ppIdx][i].startIdx = startIdx;

            if (trainingPatterns[ppIdx][i].name.find("F") != std::string::npos)
            {
                if (ppIdx != 0) //  skip the send recv
                    trainingPatterns[ppIdx][i].startIdx++;

                trainingPatterns[ppIdx][i].endIdx = startIdx + forward_log_size - 1;
                // if(ppIdx != trainingPatterns.size() - 1)
                trainingPatterns[ppIdx][i].endIdx--; //  skip the send recv

                startIdx += forward_log_size;
                continue;
            }
            else if (trainingPatterns[ppIdx][i].name.find("B") != std::string::npos)
            {
                if (ppIdx != trainingPatterns.size() - 1)
                    trainingPatterns[ppIdx][i].startIdx++; //   

                trainingPatterns[ppIdx][i].endIdx = startIdx + backward_log_size - 1;
                if (ppIdx != 0)
                    trainingPatterns[ppIdx][i].endIdx--; //  

                startIdx += backward_log_size;
                continue;
            }
            else
            {
                std::cout << "error in gen_training_pattern_withSP" << std::endl;
            }
        }
    }
}