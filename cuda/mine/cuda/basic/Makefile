CU_APPS=sumArraysOnGPU-timer \
        sumMatrixOnGPU-1D-grid-1D-block \
        sumArraysOnGPU-small-case \
        sumMatrixOnGPU
C_APPS=sumArraysOnHost

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
