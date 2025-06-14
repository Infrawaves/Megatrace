#include "nccl.h"
#include "ring_log.h"
#include "core.h"
#include <sys/un.h>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>


const int nccl_megatrace_enable = ncclGetEnv("NCCL_MEGATRACE_ENABLE") ? atoi(ncclGetEnv("NCCL_MEGATRACE_ENABLE")) : 0;
const char* nccl_megatrace_log_path = ncclGetEnv("NCCL_MEGATRACE_ENABLE")?ncclGetEnv("NCCL_MEGATRACE_LOG_PATH"):"./logs";


// 初始化环形缓冲区
void ring_buffer_init(ring_buffer_t *rb) {
    rb->head.store(0);
    rb->tail.store(0);
    ring_nccl_log.live = 1 ;

}
/*  * 获取环形缓冲区中当前未消费的日志数量  */
int ring_buffer_count(ring_buffer_t *rb) {
    int tail = rb->tail.load(std::memory_order_acquire);
    int head = rb->head.load(std::memory_order_acquire);
    if (head >= tail) {
        return head - tail;
    } else {
        return RING_BUFFER_SIZE - tail + head;
    }
}
/*
* 向环形缓冲区中写入一条日志消息
* 返回 0 表示写入成功，-1 表示缓冲区已满（日志丢弃）
*/
int ring_buffer_push(ring_buffer_t *rb, const char *msg) {
    int head = rb->head.load(std::memory_order_relaxed);
    int next_head = (head + 1) % RING_BUFFER_SIZE;
    int tail = rb->tail.load(std::memory_order_acquire);
    if (next_head == tail) {         // 缓冲区满
        return -1;
    }
    std::string msg_str(msg);  // 将 msg 转换为 std::string
    if (msg_str.length() < LOG_MAX_LEN) {
        std::strcpy(rb->buffer[head].msg, msg_str.c_str());
    } else {
        std::strncpy(rb->buffer[head].msg, msg_str.c_str(), LOG_MAX_LEN - 1);
        rb->buffer[head].msg[LOG_MAX_LEN - 1] = '\0';  // 确保终止符
    }
    rb->head.store(next_head,std::memory_order_release);
    return 0;
}
/*
* 批量从环形缓冲区中读取日志条目
* 参数 max_entries 表示最多读取的条数，将日志存入 out_entries 数组中，
* 返回实际读取的日志条数。
*/
int ring_buffer_pop_batch(ring_buffer_t *rb, log_entry_t *out_entries, int max_entries) {
    int tail = rb->tail.load(std::memory_order_relaxed);
    int head = rb->head.load(std::memory_order_acquire);
    int count;
         if (head >= tail) {
             count = head - tail;
         } else {
             count = RING_BUFFER_SIZE - tail + head;
         }
         if (count > max_entries) {
             count = max_entries;
         }
         for (int i = 0; i < count; i++) {
             int index = (tail + i) % RING_BUFFER_SIZE;
             out_entries[i] = rb->buffer[index];
         }
         rb->tail.store((tail + count) % RING_BUFFER_SIZE,std::memory_order_release);
    	 return count;
}
 /*
 * 日志写入线程：负责将环形缓冲区中的日志写入到文件中。
 * 刷新策略：  * 1. 如果缓冲区中日志数量达到 BATCH_SIZE，则立即写入。
 * 2. 如果日志数量不足，但距离上次刷新超过 FLUSH_INTERVAL_US，则写入所有已有日志。  */
void *log_writer_thread(void *arg) {
    const char *rank_str = getenv("OMPI_COMM_WORLD_RANK");
    if (rank_str == NULL) {
        fprintf(stderr, "Error: Environment variable 'OMPI_COMM_RANK' not found.\n");
        return NULL;
    }
    int rank = atoi(rank_str); // 将 rank 从字符串转换为整数
    // 定义文件路径
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/rank_%d.log", nccl_megatrace_log_path, rank);

    // 打开文件
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("[Megatrace] open file error,file path not exist.\n");
        return NULL;
    }
    if(rank == 0){ 
	 INFO(NCCL_INIT,"[Megatrace] start log thread.\n");
    }
    log_entry_t logs[BATCH_SIZE];
    struct timespec last_flush_time, now;
    clock_gettime(CLOCK_MONOTONIC, &last_flush_time);
    while (1) {
        int available = ring_buffer_count(&ring_nccl_log);  // 如果缓冲区中日志数量达到 BATCH_SIZE，则批量写入

	//	printf("%d log number \n",available);
	if (available >= BATCH_SIZE) {
		//printf("%d log number \n",available);
                int num_logs = ring_buffer_pop_batch(&ring_nccl_log, logs, BATCH_SIZE);
                for (int i = 0; i < num_logs; i++) {
                    //fprintf(fp, "%s\n", logs[i].msg);
                }
                fflush(fp);
                clock_gettime(CLOCK_MONOTONIC, &last_flush_time);  // 重置刷新时间
            } else {             // 检查距离上次刷新是否超过设定时间间隔
                clock_gettime(CLOCK_MONOTONIC, &now);
                long long elapsed_us = (now.tv_sec - last_flush_time.tv_sec) * 1000000LL +(now.tv_nsec - last_flush_time.tv_nsec) / 1000;
        if (elapsed_us >= FLUSH_INTERVAL_US && available > 0) {                 // 时间到且缓冲区中有日志，则写入所有已有日志（不足 BATCH_SIZE 的情况）
            int num_logs = ring_buffer_pop_batch(&ring_nccl_log, logs, BATCH_SIZE);
            for (int i = 0; i < num_logs; i++) {
                fprintf(fp, "%s\n", logs[i].msg);
            }
            fflush(fp);
            clock_gettime(CLOCK_MONOTONIC, &last_flush_time); // 重置刷新时间
        } else { // 如果未达到条件，则稍作等待，避免忙等待
            usleep(1000); // 睡眠 1ms
            }
        }
    }
    fclose(fp);
    return NULL;
}

void log_event(struct timespec time_api, size_t count, const char* opName, cudaStream_t stream) {
    // 用于格式化日志信息
    char log_msg[LOG_MAX_LEN];

    // 获取当前时间戳字符串（格式化为秒.纳秒的形式）
    char time_str[64];
    snprintf(time_str, sizeof(time_str), "%ld.%09ld", time_api.tv_sec, time_api.tv_nsec);
    const char *rank_str = getenv("OMPI_COMM_WORLD_RANK");
    int rank = atoi(rank_str);
    // 格式化日志内容
    snprintf(log_msg, sizeof(log_msg), "[%s] [Rank %d] Fun %s Data %zu stream %p",
             time_str,rank ,opName, count, (void*)stream);
    int num = ring_buffer_count(&ring_nccl_log);
    // 将格式化后的日志写入环形缓冲区
    if (ring_buffer_push(&ring_nccl_log, log_msg) != 0) {
        // 若环形缓冲区满，可以选择丢弃日志或其他策略

        fprintf(stderr, "%d number log  Failed to push log into ring buffer: %s\n", num,log_msg);
    }
}
