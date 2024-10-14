CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -Werror -O3 -lpthread -march=haswell

.PHONY: all clean

all: pi.out

pi.out: pi.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f pi.out	
