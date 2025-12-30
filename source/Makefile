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
#        make libbinding.a CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++
#
ifneq (,$(findstring mingw,$(CC)))
    TARGET_OS := Windows
    TARGET_ARCH := x86_64
else ifneq (,$(findstring aarch64,$(CC)))
    TARGET_OS := Linux
    TARGET_ARCH := aarch64
else
    TARGET_OS := Linux
    TARGET_ARCH := x86_64
endif

#
# Compile flags - architecture-specific baseline
#
# x86-64-v3 includes: x86-64-v2 + AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE, XSAVE
# This targets CPUs from ~2013 onwards (Intel Haswell, AMD Excavator)
# AVX512 is handled via runtime detection by GGML
#
# ARMv8-A baseline for Raspberry Pi 4 (Cortex-A72) compatibility
# GGML runtime detection handles DOTPROD, FP16, SVE for RPi 5+ (Cortex-A76)
#
INCLUDES = -I./llama.cpp/include -I./llama.cpp/ggml/include -I./llama.cpp/common -I./llama.cpp/src -I./llama.cpp/vendor -I.

ifeq ($(TARGET_ARCH),aarch64)
    ARCH_FLAGS = -march=armv8-a -mtune=generic -mlow-precision-recip-sqrt -mlow-precision-sqrt -mlow-precision-div
else
    ARCH_FLAGS = -march=x86-64-v3 -mtune=generic -mrecip=all
endif

ifdef DEBUG
    # Debug build - optimize for diagnostics
    CFLAGS   = $(INCLUDES) -O0 -g3 -DDEBUG -std=c11 -fPIC $(ARCH_FLAGS) -fstack-protector-strong -w
    CXXFLAGS = $(INCLUDES) -O0 -g3 -DDEBUG -std=c++17 -fPIC $(ARCH_FLAGS) -fstack-protector-strong -w
    LDFLAGS  = -g
else
    # Production build - optimize for performance
    CFLAGS   = $(INCLUDES) -Ofast -DNDEBUG -std=c11 -fPIC $(ARCH_FLAGS) -ffunction-sections -fdata-sections -fno-stack-protector -fvisibility=hidden -fmerge-all-constants -fno-ident -w
    CXXFLAGS = $(INCLUDES) -Ofast -DNDEBUG -std=c++17 -fPIC $(ARCH_FLAGS) -ffunction-sections -fdata-sections -fno-stack-protector -fvisibility=hidden -fmerge-all-constants -fno-ident -w
    LDFLAGS  = -Wl,--gc-sections -Wl,--as-needed -s
endif

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

# Disable build cache warning
CMAKE_ARGS += -DGGML_CCACHE=OFF

# LTO disabled - pre-built for distribution
CMAKE_ARGS += -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF

# Build type and LTO
ifdef DEBUG
    CMAKE_ARGS += -DCMAKE_BUILD_TYPE=Debug
else
    CMAKE_ARGS += -DCMAKE_BUILD_TYPE=Release
    CMAKE_ARGS += -DCMAKE_C_FLAGS="-ffunction-sections -fdata-sections -fvisibility=hidden -fmerge-all-constants -fno-ident"
    CMAKE_ARGS += -DCMAKE_CXX_FLAGS="-ffunction-sections -fdata-sections -fvisibility=hidden -fmerge-all-constants -fno-ident"
endif

# Disable native optimization (targets x86-64-v3, armv8-a)
CMAKE_ARGS += -DGGML_NATIVE=OFF

# GPU backends - Vulkan enabled, others disabled
CMAKE_ARGS += -DGGML_CUDA=OFF
CMAKE_ARGS += -DGGML_VULKAN=ON
CMAKE_ARGS += -DGGML_HIP=OFF
CMAKE_ARGS += -DGGML_METAL=OFF
CMAKE_ARGS += -DGGML_SYCL=OFF

# Vulkan debug options
ifdef DEBUG
    CMAKE_ARGS += -DGGML_VULKAN_DEBUG=ON
    CMAKE_ARGS += -DGGML_VULKAN_VALIDATE=ON
    CMAKE_ARGS += -DGGML_VULKAN_CHECK_RESULTS=OFF
    CMAKE_ARGS += -DGGML_VULKAN_SHADER_DEBUG_INFO=ON
    CMAKE_ARGS += -DGGML_VULKAN_MEMORY_DEBUG=OFF
else
    CMAKE_ARGS += -DGGML_VULKAN_DEBUG=OFF
    CMAKE_ARGS += -DGGML_VULKAN_VALIDATE=OFF
    CMAKE_ARGS += -DGGML_VULKAN_CHECK_RESULTS=OFF
    CMAKE_ARGS += -DGGML_VULKAN_SHADER_DEBUG_INFO=OFF
    CMAKE_ARGS += -DGGML_VULKAN_MEMORY_DEBUG=OFF
endif

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
CMAKE_ARGS += -DGGML_CPU_ALL_VARIANTS=OFF
CMAKE_ARGS += -DGGML_BACKEND_DL=OFF

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
else ifeq ($(TARGET_ARCH),aarch64)
    CMAKE_ARGS += -DCMAKE_SYSTEM_NAME=Linux
    CMAKE_ARGS += -DCMAKE_SYSTEM_PROCESSOR=aarch64
    CMAKE_ARGS += -DCMAKE_C_COMPILER=$(CC)
    CMAKE_ARGS += -DCMAKE_CXX_COMPILER=$(CXX)
    CMAKE_ARGS += -DGGML_CPU_ARM_ARCH=armv8-a
    # Vulkan paths for ARM64 cross-compilation
    CMAKE_ARGS += -DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu
    CMAKE_ARGS += -DVulkan_INCLUDE_DIR=/usr/include
    CMAKE_ARGS += -DVulkan_LIBRARY=/usr/lib/aarch64-linux-gnu/libvulkan.so
endif

#
# Print build information
#
$(info )
$(info zeus portable build)
$(info ===========================)
$(info TARGET_OS:   $(TARGET_OS))
$(info TARGET_ARCH: $(TARGET_ARCH))
$(info CC:          $(CCV))
$(info CXX:         $(CXXV))
$(info CFLAGS:      $(CFLAGS))
$(info CXXFLAGS:    $(CXXFLAGS))
$(info CMAKE_ARGS:  $(CMAKE_ARGS))
$(info )

#
# Build targets
#

# Build llama.cpp via cmake
.PHONY: cmake-build
cmake-build: llama.cpp/include/llama.h
	@mkdir -p build
ifdef DEBUG
	cd build && CC="$(CC)" CXX="$(CXX)" cmake ../llama.cpp $(CMAKE_ARGS) && cmake --build . --config Debug -j$$(nproc)
else
	cd build && CC="$(CC)" CXX="$(CXX)" cmake ../llama.cpp $(CMAKE_ARGS) && cmake --build . --config Release -j$$(nproc)
endif
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
