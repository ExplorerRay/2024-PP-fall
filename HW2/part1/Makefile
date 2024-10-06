CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -Werror -O3

.PHONY: all clean

all: pi.out

pi.out: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f pi.out	
