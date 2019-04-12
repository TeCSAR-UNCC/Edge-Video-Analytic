CC = g++
CFLAGS = -O3 -Wall -pthread
SOURCE = EdgeServer.cpp
EXECUTABLE= EdgeServer.bin

all: clean build

build:
	$(CC) $(SOURCE) $(CFLAGS) -o $(EXECUTABLE) `pkg-config --cflags --libs opencv`

clean:
	rm -rf $(EXECUTABLE)
	rm -rf *.o
