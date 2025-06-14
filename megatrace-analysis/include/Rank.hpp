#ifndef RANK_H
#define RANK_H
#include "Config.hpp"
class Rank
{
public:
  int id;
  int tp;
  int tp_group;
  int pp;
  int pp_group;
  int dp;
  int dp_group;
  int n_rank;
  int next_pp;
  bool is_first_pp;
  bool is_last_pp;

  Rank(int id = 0, int tp = 0, int tp_group = 0, int pp = 0, int pp_group = 0,
       int dp = 0, int dp_group = 0, int n_rank = 0,
       bool is_first_pp = false, bool is_last_pp = false)
      : id(id), tp(tp), tp_group(tp_group), pp(pp), pp_group(pp_group),
        dp(dp), dp_group(dp_group), n_rank(n_rank),
        is_first_pp(is_first_pp), is_last_pp(is_last_pp) {}

  ~Rank() {}

  int getTp() const { return tp; }
  int getTpGroup() const { return tp_group; }
  int getPp() const { return pp; }
  int getPpGroup() const { return pp_group; }
  int getDp() const { return dp; }
  int getDpGroup() const { return dp_group; }
  int getNRank() const { return n_rank; }
  bool getIsFirstPp() const { return is_first_pp; }
  bool getIsLastPp() const { return is_last_pp; }

  void setTp(int value) { tp = value; }
  void setTpGroup(int value) { tp_group = value; }
  void setPp(int value) { pp = value; }
  void setPpGroup(int value) { pp_group = value; }
  void setDp(int value) { dp = value; }
  void setDpGroup(int value) { dp_group = value; }
  void setNRank(int value) { n_rank = value; }
  void setIsFirstPp(bool value) { is_first_pp = value; }
  void setIsLastPp(bool value) { is_last_pp = value; }
  void genTrace();
  void printRankInfo();
};

Rank *initRanks(TrainingConfig &config);

void releaseRanks(Rank *ranks, TrainingConfig config);
#endif
