APPS=hello

all: ${APPS}

%: %.cu
	nvcc -O2 -arch=compute_50 -o $@ $<
clean:
	rm -f ${APPS}
