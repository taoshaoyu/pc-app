CU_APPS=checkDeviceInfor checkThreadIndex \
        checkDimension \
        hello
C_APPS=sumArraysOnHost

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
