SOURCES_ALL=$(shell find . -name "*.cpp")
EXCLUDE=./r_tree/benchmark_rstar.cpp ./r_tree/validar.cpp ./main.cpp ./benchmark_lims.cpp ./validar_lims.cpp

SOURCES_COMMON := $(filter-out $(EXCLUDE),$(SOURCES_ALL))
CXXFLAGS=-std=c++17 -Wall -O3 -ffast-math -march=native
OBJECTS=$(SOURCES_COMMON:%.cpp=%.o)
TARGET=main

.PHONY: all
all: $(TARGET) benchmark_lims validar_lims

$(TARGET): $(SOURCES_COMMON) main.cpp
	$(LINK.cpp) $^ $(LOADLIBES) $(LDLIBS) -o $@

benchmark_lims: $(SOURCES_COMMON) benchmark_lims.cpp
	$(LINK.cpp) $^ $(LOADLIBES) $(LDLIBS) -o $@

validar_lims: $(SOURCES_COMMON) validar_lims.cpp
	$(LINK.cpp) $^ $(LOADLIBES) $(LDLIBS) -o $@

.PHONY: clean
clean:
	rm -rf $(OBJECTS)
