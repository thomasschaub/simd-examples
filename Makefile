CXXFLAGS += -std=c++11 -ffast-math
#CXXFLAGS += -O0 -g -Wall -Werror
CXXFLAGS += -O3

ISPC := ispc

-include Makefile.local

all: matrix waveform osc

#matrix: CC=$(CXX)
#matrix: matrix_ispc.o matrix.o

#matrix.cpp: matrix_ispc.h

%.svg: %.r %.csv
	R --vanilla < $<

waveform: LDLIBS=-lsndfile

osc: LDLIBS=-lsndfile

#%_ispc.o: %.ispc
#	$(ISPC) -o $@ -h $(basename $@).h $<

#%_ispc.h: %.ispc
#	$(ISPC) -h $(basename $@).h $<

clean:
	rm -rf matrix waveform
