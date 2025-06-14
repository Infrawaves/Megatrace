CXX = g++
CXXFLAGS = -std=c++17 -Wall -g -Iinclude

TARGET = Trace
SRCS = src/main.cpp src/LogParser.cpp src/GraphNode.cpp src/rank.cpp src/Config.cpp
HDRS = include/Semaphore.hpp include/LogParser.hpp include/Rank.hpp include/GraphNode.hpp include/Config.hpp
TARGET_DIR = build
OUTPUT_DIR = output
DEPS = $(SRCS:.cpp=.d)

all: $(TARGET_DIR) $(TARGET_DIR)/$(TARGET)

# 创建目标目录
$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

# 生成目标可执行文件，依赖于源文件和头文件
$(TARGET_DIR)/$(TARGET): $(SRCS) $(HDRS) | $(TARGET_DIR)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@

# 自动生成依赖文件
%.d: %.cpp
	@$(CXX) $(CXXFLAGS) -MM -MT "$(@:.d=.o) $@" $< -MF $@

# 清理构建目录
clean:
	rm -rf $(TARGET_DIR) $(OUTPUT_DIR)/test/*

fclean: clean

re: fclean all

.PHONY: all clean fclean re