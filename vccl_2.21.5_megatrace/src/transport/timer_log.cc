#include "timer_log.h"
#include "nccl.h"
#include "core.h"
#include <sys/un.h>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <map>
#include <chrono>

const int nccl_telemetry_enable = ncclGetEnv("NCCL_TELEMETRY_ENABLE") ? atoi(ncclGetEnv("NCCL_TELEMETRY_ENABLE")) : 0;
const char* nccl_telemetry_log_path = ncclGetEnv("NCCL_TELEMETRY_LOG_PATH");
using Clock = std::chrono::steady_clock;

std::string getCurrentTimeString() {
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);

    std::ostringstream oss;
    oss << std::put_time(localTime, "%Y%m%d%H%M%S");
    return oss.str();
}

void printLogInfo(struct timer_log log){
  INFO(NCCL_NET, "%d.%d.%d.%d->%d.%d.%d.%d send %d Bits used %lld nsec", 
             log.srcIp[0],log.srcIp[1],log.srcIp[2],log.srcIp[3],
             log.dscIp[0],log.dscIp[1],log.dscIp[2],log.dscIp[3],
             log.size,log.diff
  );
}

struct PortLogs {
  std::ofstream files[2];                 // 两个日志文件
  int currentFile = 0;                    // 当前活动的文件索引（0或1）
  bool headerWritten[2] = {false, false}; // 是否已写入标题行
  std::string filenames[2];               // 两个文件的路径
  Clock::time_point            startTime[2];
};
std::map<std::string, std::map<int, PortLogs>> logFilesMap; // 网卡名 -> 端口 -> 日志文件

void* timerLogService(void *args){
  // signal(SIGPIPE, sigpipe_handler);
  //setupTelemetry();//set up environment variables
  struct sockaddr_un server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sun_family = AF_UNIX;
  strncpy(server_addr.sun_path, SOCK_PATH, sizeof(server_addr.sun_path) - 1);
  //WARN("------------NCCL_TELEMETRY_ENABLE = %d-------------", nccl_telemetry_enable);

  if(TIMER_LOG_NCCL_TELEMETRY){
    static std::map<std::string, std::map<int, PortLogs>> logFilesMap; // new

    //std::string baseName = global_timer_log.log[i].NetworkCardName;
    std::string timestamp = getCurrentTimeString();

    while(!global_timer_log.stop){
      global_timer_log.collect = 1;

      __sync_synchronize();
      if(!global_timer_log.log.empty()){

        pthread_mutex_lock(&global_timer_log.lock);
        __sync_synchronize();
        if(global_timer_log.log.empty()){
          pthread_mutex_unlock(&global_timer_log.lock);
          continue;
        }
        timer_log log = global_timer_log.pop();
        
        // update slide window
        global_timer_log.pushSlideWindow(log, log.devIndex);
        pthread_mutex_unlock(&global_timer_log.lock);
        if (global_timer_log.slideWindow[log.devIndex].size() < maxWindowSize) {
          continue;
        }
        
        std::string ncName = log.NetworkCardName;
        auto& nicMap = logFilesMap[ncName];
        if (!nicMap.count(log.devIndex)) {
          auto& portLogs = nicMap[log.devIndex];
          char hostname[1024];
          getHostName(hostname, 1024, '.');
          for (int i = 0; i < 2; i++) {
            std::string filename = std::string(nccl_telemetry_log_path) + "/" +
                                   hostname + "_" + ncName + "_Port" + std::to_string(log.devIndex)+
                                   (i == 0 ? "_A.log" : "_B.log");
            portLogs.filenames[i] = filename;
            portLogs.files[i].open(filename, std::ios::trunc);
            portLogs.files[i] << "Time,Group,FromRank,ToRank,DevIndex,Func,FuncTimes,SrcIP,DstIP,Bandwidth,SendWrCounter,RemainWrDataSize,Timestamp\n";
            portLogs.headerWritten[i] = true;
            // portLogs.startTime[i] = Clock::now();
          }
          portLogs.currentFile = 0; // 初始化当前文件索引
        }
        PortLogs& portLogs = nicMap[log.devIndex];
        std::ofstream* pFile = &portLogs.files[portLogs.currentFile];

        // static bool first10MBPrinted[2] = {false, false};
        if (static_cast<size_t>(pFile->tellp()) >= 10 * 1024 * 1024) {
          // long long nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(
          //   Clock::now() - portLogs.startTime[portLogs.currentFile]).count();

          // if (!first10MBPrinted[log.devIndex]) {
          //   printf("[NCCL][Telemetry] %s first reached 10MiB in %.3f s (%lld ns)\n",
          //           portLogs.filenames[portLogs.currentFile].c_str(), nsec / 1e9, nsec);
          //   first10MBPrinted[log.devIndex] = true;
          // }
          pFile->close();
          portLogs.currentFile ^= 1;
          pFile = &portLogs.files[portLogs.currentFile];
          
          pFile->open(portLogs.filenames[portLogs.currentFile], std::ios::trunc);
          *pFile << "Time,Group,FromRank,ToRank,DevIndex,Func,FuncTimes,SrcIP,DstIP,Bandwidth,SendWrCounter,RemainWrDataSize,Timestamp\n";
          portLogs.headerWritten[portLogs.currentFile] = true;
          // 为下一次轮转重新记起点
          // portLogs.startTime[portLogs.currentFile] = Clock::now();
        }
        int bandWidths = global_timer_log.getBandWidths(log.devIndex);
        char dataBuffer[512];
        sprintf(dataBuffer, "%s,%lu,%d,%d,%d,%u,%lld,%d.%d.%d.%d,%d.%d.%d.%d,%d,%d,%d,%lld",
                getCurrentTimeString().c_str(), log.groupHash, log.rank, log.peerRank, log.devIndex,
                log.func, log.ncclFuncTimes,
                log.srcIp[0], log.srcIp[1], log.srcIp[2], log.srcIp[3],
                log.dscIp[0], log.dscIp[1], log.dscIp[2], log.dscIp[3],
                bandWidths, log.sendWrCounter, log.remainWrDataSize, log.diff);
        (*pFile) << dataBuffer << std::endl;
      }
    }
    for (auto& nic : logFilesMap) {
      for (auto& port : nic.second) {
          port.second.files[0].close();
          port.second.files[1].close();
      }
    }
  }
  return 0;
}