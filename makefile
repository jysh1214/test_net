.PHONY: clean dirs

GPP=g++ -std=c++11 -Wall
SRC=src
OBJ=obj
EXAMPLE=example
CXXFLAG=-O3 -fopenmp -Wno-unused-result -Wno-unused-function -Isrc/ -Iinclude/

all: dirs ./nn

# nn
./nn: $(EXAMPLE)/nn.cpp\
$(OBJ)/connected_layer.o\
$(OBJ)/net.o
	$(GPP) $^ -o $@ $(CXXFLAG)

$(OBJ)/connected_layer.o: $(SRC)/connected_layer.cpp $(SRC)/connected_layer.h
	$(GPP) -c $< -o $@ $(CXXFLAG)

$(OBJ)/net.o: $(SRC)/net.cpp $(SRC)/net.h
	$(GPP) -c $< -o $@ $(CXXFLAG)

dirs:
	mkdir -p $(SRC) $(OBJ)

clean:
	rm -rf $(OBJ) ./nn .vscode

state:
	wc src/*
