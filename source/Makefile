.PHONY: clean cmake-build

# Default compilers
CC ?= gcc
CXX ?= g++
AR ?= ar

CCV := $(shell $(CC) --version | head -n 1)
CXXV := $(shell $(CXX) --version | head -n 1)

#
# Cross-compilation detection
# Usage: make libbinding.a CC=x86_64-w64-mingw32-gcc CXX=x86_64-w64-mingw32-g++
#
ifneq (,$(findstring mingw,$(CC)))
    TARGET_OS := Windows
else
    TARGET_OS := Linux
endif

#
# Compile flags - portable x86_64 baseline
#
# x86-64-v3 includes: x86-64-v2 + AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE, XSAVE
# This targets CPUs from ~2013 onwards (Intel Haswell, AMD Excavator)
# AVX512 is handled via runtime detection by GGML
#
INCLUDES = -I./llama.cpp/include -I./llama.cpp/ggml/include -I./llama.cpp/common -I./llama.cpp/src -I./llama.cpp/vendor -I.
CFLAGS   = $(INCLUDES) -O3 -DNDEBUG -std=c11 -fPIC -march=x86-64-v3 -w
CXXFLAGS = $(INCLUDES) -O3 -DNDEBUG -std=c++17 -fPIC -march=x86-64-v3 -w
LDFLAGS  =

# Platform-specific flags
ifeq ($(TARGET_OS),Windows)
    CFLAGS   += -D_WIN32 -D_WIN32_WINNT=0x0A00
    CXXFLAGS += -D_WIN32 -D_WIN32_WINNT=0x0A00
    LDFLAGS  += -static-libgcc -static-libstdc++
else
    CFLAGS   += -pthread
    CXXFLAGS += -pthread
endif

#
# CMake configuration for llama.cpp
#
CMAKE_ARGS =

# Disable native optimization (we handle it with -march=x86-64-v3)
CMAKE_ARGS += -DGGML_NATIVE=OFF

# GPU backends - Vulkan enabled, others disabled
CMAKE_ARGS += -DGGML_CUDA=OFF
CMAKE_ARGS += -DGGML_VULKAN=ON
CMAKE_ARGS += -DGGML_HIP=OFF
CMAKE_ARGS += -DGGML_METAL=OFF
CMAKE_ARGS += -DGGML_SYCL=OFF

# Disable OpenMP for standalone executable (uses native thread pool instead)
CMAKE_ARGS += -DGGML_OPENMP=OFF

# Disable unnecessary features
CMAKE_ARGS += -DLLAMA_CURL=OFF
CMAKE_ARGS += -DLLAMA_HTTPLIB=OFF
CMAKE_ARGS += -DLLAMA_BUILD_TESTS=OFF
CMAKE_ARGS += -DLLAMA_BUILD_EXAMPLES=OFF
CMAKE_ARGS += -DLLAMA_BUILD_SERVER=OFF
CMAKE_ARGS += -DLLAMA_BUILD_TOOLS=OFF
CMAKE_ARGS += -DBUILD_SHARED_LIBS=OFF

# Cross-compilation CMake args
ifeq ($(TARGET_OS),Windows)
    CMAKE_ARGS += -DCMAKE_SYSTEM_NAME=Windows
    CMAKE_ARGS += -DCMAKE_C_COMPILER=$(CC)
    CMAKE_ARGS += -DCMAKE_CXX_COMPILER=$(CXX)
    CMAKE_ARGS += -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER
    CMAKE_ARGS += -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY
    CMAKE_ARGS += -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY
    # Vulkan paths for cross-compilation
    CMAKE_ARGS += -DCMAKE_FIND_ROOT_PATH=/opt/llvm-mingw/x86_64-w64-mingw32
    CMAKE_ARGS += -DVulkan_INCLUDE_DIR=/opt/llvm-mingw/x86_64-w64-mingw32/include
    CMAKE_ARGS += -DVulkan_LIBRARY=/opt/llvm-mingw/x86_64-w64-mingw32/lib/libvulkan-1.a
endif

#
# Print build information
#
$(info )
$(info zeus portable build)
$(info ===========================)
$(info TARGET_OS:  $(TARGET_OS))
$(info CC:         $(CCV))
$(info CXX:        $(CXXV))
$(info CFLAGS:     $(CFLAGS))
$(info CXXFLAGS:   $(CXXFLAGS))
$(info CMAKE_ARGS: $(CMAKE_ARGS))
$(info )

#
# Build targets
#

# Build llama.cpp via cmake
.PHONY: cmake-build
cmake-build: llama.cpp/include/llama.h
	@mkdir -p build
	cd build && CC="$(CC)" CXX="$(CXX)" cmake ../llama.cpp $(CMAKE_ARGS) && cmake --build . --config Release -j$$(nproc) || true
	@# Check that required libraries exist (either with or without lib prefix)
	@test -f build/src/libllama.a -o -f build/src/llama.a || (echo "ERROR: llama library not built"; exit 1)
	@test -f build/ggml/src/libggml-base.a -o -f build/ggml/src/ggml-base.a || (echo "ERROR: ggml-base library not built"; exit 1)

# Compile binding.o
binding.o: binding.cpp binding.h cmake-build
	$(CXX) $(CXXFLAGS) binding.cpp -o binding.o -c

# Create libbinding.a and copy libraries for CGO
libbinding.a: binding.o
	$(AR) rcs libbinding.a binding.o
	@# Copy libraries, handling both naming conventions
	@if [ -f build/src/libllama.a ]; then cp build/src/libllama.a ./; else cp build/src/llama.a ./libllama.a; fi
	@if [ -f build/ggml/src/libggml.a ]; then cp build/ggml/src/libggml.a ./; else cp build/ggml/src/ggml.a ./libggml.a; fi
	@if [ -f build/ggml/src/libggml-base.a ]; then cp build/ggml/src/libggml-base.a ./; else cp build/ggml/src/ggml-base.a ./libggml-base.a; fi
	@if [ -f build/ggml/src/libggml-cpu.a ]; then cp build/ggml/src/libggml-cpu.a ./; else cp build/ggml/src/ggml-cpu.a ./libggml-cpu.a; fi
	@if [ -f build/common/libcommon.a ]; then cp build/common/libcommon.a ./; else cp build/common/common.a ./libcommon.a; fi
	@if [ -f build/ggml/src/ggml-vulkan/libggml-vulkan.a ]; then cp build/ggml/src/ggml-vulkan/libggml-vulkan.a ./; \
	 elif [ -f build/ggml/src/ggml-vulkan/ggml-vulkan.a ]; then cp build/ggml/src/ggml-vulkan/ggml-vulkan.a ./libggml-vulkan.a; \
	 elif [ -f build/ggml/src/libggml-vulkan.a ]; then cp build/ggml/src/libggml-vulkan.a ./; fi

clean:
	rm -rf *.o
	rm -rf *.a
	rm -rf .objs
	rm -rf build
	rm -f prepare
