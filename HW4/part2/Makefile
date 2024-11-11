CXX = mpicxx
CXXFLAGS = -std=c++17 -Wall -Wextra -Werror -Ofast
# INCLUDE = -I./include

SRCS = $(wildcard *.cc)
OBJS = $(SRCS:.cc=.o)
TARGET = matmul

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $^ -o $@

# %.o: %.cc
# 	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

clean:
	-rm -f $(OBJS) $(TARGET)
