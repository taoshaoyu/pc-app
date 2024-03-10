APPS=hello

all: ${APPS}

%: %.cu
	nvcc -O2 -arch=compute_90 -o $@ $<
clean:
	rm -f ${APPS}
