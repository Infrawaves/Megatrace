#include <iostream>
#include "Rank.hpp"
#include "LogParser.hpp"
#include "Config.hpp"
using namespace std;

Rank *initRanks(TrainingConfig &config)
{
    Rank *ranks;

    config.dpSize = config.numRanks / (config.tpSize * config.ppSize);

    ranks = new Rank[config.numRanks];

    config.tpGroupSize = config.numRanks / config.tpSize;
    config.ppGroupSize = config.numRanks / config.ppSize;
    config.dpGroupSize = config.numRanks / config.dpSize;

    for (int i = 0; i < config.numRanks; ++i)
    {
        ranks[i].id = i;
        ranks[i].setNRank(config.numRanks);
        ranks[i].setTp(i % config.tpSize);
        ranks[i].setTpGroup(i / config.tpSize);
        ranks[i].setPp(i / (config.numRanks / config.ppSize));
        ranks[i].setPpGroup(i % (config.numRanks / config.ppSize));

        if (i / (config.numRanks / config.ppSize) == 0)
        {
            ranks[i].setIsFirstPp(true);
        }
        else
        {
            ranks[i].setIsFirstPp(false);
        }

        if (i / (config.numRanks / config.ppSize) == config.ppSize - 1)
        {
            ranks[i].setIsLastPp(true);
        }
        else
        {
            ranks[i].setIsLastPp(false);
        }
        ranks[i].setDp((i / (config.tpSize)) % config.dpSize);
        ranks[i].setDpGroup((i / (config.tpSize * config.dpSize)) * config.tpSize + i % config.tpSize);
    }
    return ranks;
}

void releaseRanks(Rank *ranks, TrainingConfig config)
{
    delete[] ranks;
}

void Rank::printRankInfo()
{
    std::cout << "Rank ID: " << id << std::endl;
    std::cout << "TP: " << tp << std::endl;
    std::cout << "TP Group: " << tp_group << std::endl;
    std::cout << "PP: " << pp << std::endl;
    std::cout << "PP Group: " << pp_group << std::endl;
    std::cout << "DP: " << dp << std::endl;
    std::cout << "DP Group: " << dp_group << std::endl;
    std::cout << "N Rank: " << n_rank << std::endl;
    std::cout << "Next PP: " << next_pp << std::endl;
    std::cout << "Is First PP: " << (is_first_pp ? "true" : "false") << std::endl;
    std::cout << "Is Last PP: " << (is_last_pp ? "true" : "false") << std::endl;
}