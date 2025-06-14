import argparse
import os
import re
import sys
from collections import defaultdict
import logging
import copy

# Define enumeration mappings for func values
FUNC_ENUM = {
    0: "ncclFuncBroadcast",
    1: "ncclFuncReduce",
    2: "ncclFuncAllGather",
    3: "ncclFuncReduceScatter",
    4: "ncclFuncAllReduce",
    5: "ncclFuncSendRecv",
    6: "ncclFuncSend",
    7: "ncclFuncRecv",
    8: "ncclNumFuncs"
}
logging.basicConfig(filename='debug.log', level=logging.DEBUG, 
                     format='%(asctime)s - %(levelname)s - %(message)s')


def detect_cycles(data, output_file_path):
    """
    Detects loops in the direction of the rank flow and writes the results to an output file.
    """
    
    def find_cycles(graph):
        index = 0
        stack = []
        indices = {}
        lowlink = {}
        on_stack = set()
        cycles = []

        def strongconnect(v):
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)

            for w in graph[v]:
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                cycle = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    cycle.append(w)
                    if w == v:
                        break
                cycles.append(cycle)

        graph_copy = copy.deepcopy(graph)
        for v in graph_copy:
            if v not in indices:
                strongconnect(v)


        return cycles

    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(output_file_path, 'w') as output_file:
            for group, funcs in data.items():
                for func_description, func_timess in funcs.items():
                    for func_times, channels in func_timess.items():
                        for channel, flows in channels.items():
                            graph = defaultdict(list)
                            flow_keys = list(flows.keys()) 
                            for flow in flow_keys:
                                from_rank, to_rank = map(int, flow.split('->'))
                                graph[from_rank].append(to_rank)
                                
                            cycles = find_cycles(graph)
                            for cycle in cycles:
                                output_file.write(f"Detected cycle in group {group}, func {func_description}, func_time {func_times}, channel {channel}: {' -> '.join(map(str, cycle))}\n")

                                
        logging.info(f"Cycle detection results have been written to {output_file_path}")
        print(f"Cycle detection results have been written to {output_file_path}")
    except Exception as e:
        logging.error(f"Error writing to output file {output_file_path}: {e}")
        print(f"Error writing to output file {output_file_path}: {e}")


def process_logs(log_folder_path, output_file_path, cycle_output_file_path):
    """
    Processes the log file to calculate the transfer rate and writes the result to the output file.
    """
    log_pattern = re.compile(
        r'(\S+):(\S+):(\d+): Group:(\d+), from rank (\d+) to rank (\d+), channel_id:(\d+), func:(\d+),FuncTimes:(\d+), (\d+\.\d+\.\d+\.\d+)->(\d+\.\d+\.\d+\.\d+) send (\d+) Bytes used (\d+) nsec'
    )

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "bytes_sent": [],
        "time_used": [],
        "node": [],
        "cards": [],
        "timestamp": [],
        "src_ip": [],
        "dst_ip": [],
    }))))))

    for root, _, files in os.walk(log_folder_path):
        for log_file in files:
            if log_file.endswith('.log'):  # .log only
                log_file_path = os.path.join(root, log_file)
                if os.path.isfile(log_file_path):
                    try:
                        with open(log_file_path, 'r') as file:
                            for line in file:
                                match = log_pattern.match(line.strip())
                                if match:
                                    groups = match.groups()
                                    if len(groups) != 13:
                                        logging.error(f"Unexpected number of groups: {len(groups)}")
                                        return  
                                    node, cards, timestamp, group, from_rank, to_rank, channel, func, func_times, src_ip, dst_ip, bytes_sent, time_used = groups
                                    bytes_sent = int(bytes_sent)
                                    time_used = int(time_used)
                                    func_description = FUNC_ENUM.get(int(func), "Unknown")
                                    flow = f"{from_rank}->{to_rank}"
                                    ip_flow = f"{src_ip}->{dst_ip}"
                                    data[group][func_description][func_times][channel][flow][ip_flow]["bytes_sent"].append(bytes_sent)
                                    data[group][func_description][func_times][channel][flow][ip_flow]["time_used"].append(time_used)
                                    data[group][func_description][func_times][channel][flow][ip_flow]["node"].append(node)
                                    data[group][func_description][func_times][channel][flow][ip_flow]["cards"].append(cards)
                                    data[group][func_description][func_times][channel][flow][ip_flow]["timestamp"].append(timestamp)
                                    #data[group][func_description][flow][ip_flow]["channel"].append(channel)
                                    data[group][func_description][func_times][channel][flow][ip_flow]["src_ip"].append(src_ip)
                                    data[group][func_description][func_times][channel][flow][ip_flow]["dst_ip"].append(dst_ip)
                                    #data[group][func_description][func_times][channel][flow][ip_flow]["func_times"].append(func_times)
                                    if group not in data :
                                        logging.error(f"Missing keys: [group] in data structure for group {group}, func {func_description}, func_times {func_times}, channel {channel}, flow {flow}, ip_flow {ip_flow}")
                                        continue
                                    if func_description not in data[group]:
                                        logging.error(f"Missing keys: [func_description] in data structure for group {group}, func {func_description}, func_times {func_times}, channel {channel}, flow {flow}, ip_flow {ip_flow}")
                                        continue
                                    if func_times not in data[group][func_description]:
                                        logging.error(f"Missing keys: [func_times] in data structure for group {group}, func {func_description}, func_times {func_times}, channel {channel}, flow {flow}, ip_flow {ip_flow}")
                                        continue
                                    if  channel not in data[group][func_description][func_times]:
                                        logging.error(f"Missing keys: [channel] in data structure for group {group}, func {func_description}, func_times {func_times}, channel {channel}, flow {flow}, ip_flow {ip_flow}")
                                        continue
                                    if  ip_flow not in data[group][func_description][func_times][channel][flow] :
                                        logging.error(f"Missing keys: [ip_flow] in data structure for group {group}, func {func_description}, func_times {func_times}, channel {channel}, flow {flow}, ip_flow {ip_flow}")
                                        continue
                                else:
                                    logging.warning(f"Skipping line: {line.strip()}")
                    except PermissionError:
                        logging.error(f"PermissionError: Cannot read file {log_file_path}")
                    except Exception as e:
                        logging.error(f"Error processing file {log_file_path}: {e}")

    detect_cycles(data, cycle_output_file_path)
    
    file_results = defaultdict(lambda:defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))))

    for group, funcs in data.items():
        for func_description, func_timess in funcs.items():
            for func_times, channels in func_timess.items():
                for channel, flows in channels.items():
                    for flow, ip_flows in flows.items():
                        for ip_flow, metrics in ip_flows.items():
                            total_bytes = sum(metrics["bytes_sent"])
                            total_time = sum(metrics["time_used"])
                            if total_time > 0:
                                transmission_rate_gbps = (total_bytes * 8) / total_time
                            else:
                                transmission_rate_gbps = 0
                            file_results[group][func_description][func_times][channel][flow][ip_flow].append(transmission_rate_gbps)

    group_rates = defaultdict(list)
    lowest_rate_info = {"group": None, "func": None, "func_times": None, "channel": None, "flow": None, "ip_flow": None, "rate": float('inf')}
    for group, funcs in file_results.items():
        for func_description, func_timess in funcs.items():
            for func_times, channels in func_timess.items():
                for channel, flows in channels.items():
                    for flow, ip_flows in flows.items():
                        for ip_flow, rates in ip_flows.items():
                            if rates:
                                average_rate = sum(rates) / len(rates)
                                group_rates[group].append((group, func_description, func_times, channel, flow, ip_flow, average_rate))
                                if min(rates) < lowest_rate_info["rate"]:
                                    lowest_rate_info = {
                                        "group": group,
                                        "func": func_description,
                                        "func_times": func_times,
                                        "channel": channel,
                                        "flow": flow,
                                        "ip_flow": ip_flow,
                                        "rate": min(rates)
                                    }
    total_rates = [rate for group_rates in file_results.values() for funcs in group_rates.values() for func_timess in funcs.values() 
                    for channels in func_timess.values() for flows in channels.values() for ip_flows in flows.values() for rate in ip_flows]
    average_transmission_rate_gbps = sum(total_rates) / len(total_rates) if total_rates else 0

    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        with open(output_file_path, 'w') as output_file:
            for group, items in group_rates.items():
                average_rate = sum(item[6] for item in items) / len(items) if items else 0
                lowest_rate_info_group = min(items, key=lambda x: x[6]) if items else None
                output_file.write(f"Group: {group}\n")
                output_file.write(f"  Average Transmission Rate:\n")
                output_file.write(f"    Average Rate: {average_rate:.9f} Gb/s\n")
                if lowest_rate_info_group:
                    output_file.write(f"  Lowest Transmission Rate:\n")
                    output_file.write(
                        f"    Func: {lowest_rate_info_group[1]}, Func_times: {lowest_rate_info_group[2]}, Channel: {lowest_rate_info_group[3]}, Flow: {lowest_rate_info_group[4]}, IP Flow: {lowest_rate_info_group[5]}, Transmission Rate: {lowest_rate_info_group[6]:.9f} Gb/s\n")
                else:
                    output_file.write(f"  Lowest Transmission Rate: Not available\n")
                output_file.write("\n")

            output_file.write(
                f"Average Transmission Rate across the Network: {average_transmission_rate_gbps:.9f} Gb/s\n")
            output_file.write(f"Overall Lowest Transmission Rate across the Network:\n")
            output_file.write(
                f"  Group: {lowest_rate_info['group']}, Func: {lowest_rate_info['func']}, Func_times: {lowest_rate_info['func_times']}, Channel: {lowest_rate_info['channel']}, Flow: {lowest_rate_info['flow']}, IP Flow: {lowest_rate_info['ip_flow']}, Transmission Rate: {lowest_rate_info['rate']:.9f} Gb/s\n")

            output_file.write("\nDetailed Data Structure:\n")
            for group, funcs in data.items():
                output_file.write(f"Group: {group}\n")
                for func_description, func_timess in funcs.items():
                    for func_times, channels in func_timess.items():
                        for channel, flows in channels.items():
                            for flow, ip_flows in flows.items():
                                for ip_flow, metrics in ip_flows.items():
                                    avg_bytes = sum(metrics["bytes_sent"]) / len(metrics["bytes_sent"]) if metrics[
                                        "bytes_sent"] else 0
                                    avg_time = sum(metrics["time_used"]) / len(metrics["time_used"]) if metrics["time_used"] else 0
                                    avg_rate = (avg_bytes * 8) / avg_time if avg_time > 0 else 0
                                    output_file.write(f"    Func: {func_description}, Func_times: {func_times}, Channel: {channel}, Flow: {flow}, IP Flow: {ip_flow}\n")
                                    output_file.write(f"      Average Bytes Sent: {avg_bytes}\n")
                                    output_file.write(f"      Average Time Used: {avg_time}\n")
                                    output_file.write(f"      Average Transmission Rate: {avg_rate:.9f} Gb/s\n")

        print(f"Transmission rates by group have been calculated and written to {output_file_path}")
        print(f"Average transmission rate across the network: {average_transmission_rate_gbps:.9f} Gb/s")
        print(f"Overall lowest transmission rate across the network: {lowest_rate_info['rate']:.9f} Gb/s")
    except Exception as e:
        print(f"Error writing to output file {output_file_path}: {e}")
        logging.error(f"Error writing to output file {output_file_path}: {e}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process log files to calculate transmission rates.")
    parser.add_argument("log_folder_path", type=str, help="The path to the log folder.")
    parser.add_argument("output_file_path", type=str, help="The path to the output file.")
    parser.add_argument("cycle_output_file_path", type=str, help="The path to the output file for cycle detection results.")
    
    args = parser.parse_args()

    if not os.path.isdir(args.log_folder_path):
        logging.error(f"Error: {args.log_folder_path} is not a valid directory.")
        sys.exit(1)

    if os.path.exists(args.output_file_path) and not os.access(args.output_file_path, os.W_OK):
        logging.error(f"Error: No write permission for {args.output_file_path}.")
        sys.exit(1)
        
    process_logs(args.log_folder_path, args.output_file_path, args.cycle_output_file_path)
