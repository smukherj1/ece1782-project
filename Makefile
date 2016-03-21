TARGET = sort
OBJS = sort.o gpu_sort.o cpu_sort.o utils.o
NVCFLAGS = -arch=sm_52 -std=c++11 --compiler-options "-fopenmp"
LDFLAGS =
LIBDIR =
LIBS =
INCLUDEDIR =
NVCC = nvcc
DEBUG = 0

ifeq ($(DEBUG), 1)
		NVCFLAGS += -G -g
else
        NVCFLAGS += -O3
endif

$(TARGET): $(OBJS)
	@echo Linking...
	@$(NVCC) $(NVCFLAGS) $(LDFLAGS) $(LIBDIR) $(OBJS) $(LIBS) -o $(TARGET)
	@echo Successfully built target "'"$(TARGET)"'"

%.o: %.cpp
	@echo Compiling C++ source $<...
	@$(NVCC) -M $(NVCFLAGS) $(INCLUDEDIR) $< > $*.d
	@$(NVCC) $(NVCFLAGS) $(INCLUDEDIR) $< -c

%.o: %.cu
	@echo Compiling CUDA source $<...
	@$(NVCC) -M $(NVCFLAGS) $(INCLUDEDIR) $< > $*.d
	@$(NVCC) $(NVCFLAGS) $(INCLUDEDIR) $< -c

clean:
	@-rm -rf *.o $(TARGET) *.d

run: $(TARGET)
	@./$(TARGET)

-include $(OBJS:.o=.d)