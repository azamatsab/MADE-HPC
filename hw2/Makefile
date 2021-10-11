CXX=g++
CFLAGS=-c
SOURCES=main.cpp matmul.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=speed_test

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
			$(CXX) $(OBJECTS) -o $(EXECUTABLE)

%.o : %.cpp
	$(CXX) $(CFLAGS) $(EXTFLAGS) $< -o $@

clean:
			rm -rf *o speed_test
