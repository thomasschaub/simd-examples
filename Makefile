CXXFLAGS += -std=c++11 -ffast-math
#CXXFLAGS += -O0 -g -Wall -Werror
CXXFLAGS += -O3

ISPC := ispc

-include Makefile.local

all: convolution convolution_test matrix osc waveform

%.pdf: %.r %.csv
	R --vanilla < $<

convolution: LDLIBS=-lsndfile

convolution_test: LDLIBS+=-lgtest_main -lpthread

osc: LDLIBS=-lsndfile

waveform: LDLIBS=-lsndfile

clean:
	rm -rf convolution convolution_test matrix osc waveform
