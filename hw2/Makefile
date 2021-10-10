CC=g++
CFLAGC=-c -Wall
SOURCES=main.cpp matmul.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=speed_test

all: $(EXECUTABLE)
$(EXECUTABLE): $(OBJECTS)
			$(CC) $(OPTIMIZATION) $(OBJECTS) -o $(EXECUTABLE)

.o:
	$(CC) $(OPTIMIZATION) $(CFLAGS) $< -o $@

clean:
			rm -rf *o speed_test

