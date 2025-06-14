# Megatrace


Megatrace is a lightweight system tool for training performance analysis and issue localization, implemented by enhancing the Nvidia collective communication library (referred to as VCCL). It primarily consists of two components: VCCL performance collection and Megatrace-analysis.

## VCCL 

### Build
```shell

```

### Environment
You can enable the performance collection feature using the following environment variables:
```shell
-x LD_LIBRARY_PATH=/path/vccl/build/lib:$LD_LIBRARY_PATH \
-x NCCL_MEGATRACE_ENABLE=1 \
-x NCCL_MEGATRACE_LOG_PATH=./mega_log \

```

## Megatrace-analysis

### Build
```shell
make 
```
### config
```yaml
isSP: false
layers: 32
ppSize: 4
tpSize: 2
GBS: 512
headers: 32
numRanks: 512
iterations: 50
slowThreshold: 1
```

### Run
```shell
./Trace  <log_file_path>  <output_file_path> 
```
### Graph

https://dreampuf.github.io/GraphvizOnline

### Example



 



