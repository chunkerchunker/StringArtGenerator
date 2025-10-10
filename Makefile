.PHONY: all build build-c build-fast build-wasm clean test test-c test-fast

CC = clang

# Detect architecture and set SIMD flags accordingly
ARCH := $(shell uname -m)
ifeq ($(ARCH),x86_64)
    # x86_64: Enable AVX if supported
    SIMD_FLAGS = -mavx
else ifeq ($(ARCH),aarch64)
    # ARM 64-bit: NEON is always available
    SIMD_FLAGS =
else ifeq ($(ARCH),arm64)
    # Apple Silicon: NEON is always available
    SIMD_FLAGS =
else
    SIMD_FLAGS =
endif

CFLAGS = -std=c99 -O3 -march=native -Wall -Wextra
# Enable SIMD optimizations for hot path (AVX on x86, NEON on ARM)
CFLAGS_FAST = -std=c99 -O3 -march=native $(SIMD_FLAGS) -funroll-loops -ffast-math -finline-functions -Wall -Wextra
LDFLAGS = -lm

all: build build-c build-fast build-wasm

build:
	go build -o stringart main.go

build-c: stringart-c

build-fast: stringart-fast

stringart-c: stringart.c stb_image.h stb_image_write.h
	$(CC) $(CFLAGS) stringart.c -o stringart-c $(LDFLAGS)

stringart_core.o: stringart_core.c stringart_core.h stb_image.h stb_image_write.h
	$(CC) $(CFLAGS_FAST) -c stringart_core.c -o stringart_core.o

stringart-fast: stringart_fast.c stringart_core.o stringart_core.h
	$(CC) $(CFLAGS_FAST) stringart_fast.c stringart_core.o -o stringart-fast $(LDFLAGS)

build-wasm: stringart.wasm.js

stringart.wasm.js: stringart_wasm.c stringart_core.c stringart_core.h stb_image.h stb_image_write.h
	emcc -O3 -msimd128 -s WASM=1 -s EXPORTED_RUNTIME_METHODS='["cwrap","ccall","HEAPU8","HEAP8","HEAP32","getValue","setValue","wasmMemory"]' \
	     -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s EXPORT_NAME="StringArtModule" \
	     -s EXPORTED_FUNCTIONS='["_malloc","_free"]' \
	     -s INITIAL_MEMORY=32MB -s MAXIMUM_MEMORY=1GB \
	     stringart_wasm.c stringart_core.c -o stringart.wasm.js

clean:
	rm -f stringart stringart-c stringart-fast stringart.wasm.js stringart.wasm stringart_core.o stringart.wasm.wasm

test: build
	./stringart -input ae300.jpg -pins 400 -lines 20000 -output-size 2000 -weight 4 -output-weight 15
	open output.png

test-c: build-c
	./stringart-c -input ae300.jpg -pins 400 -lines 20000 -output-size 2000 -weight 4 -output-weight 15
	open output.png

test-fast: build-fast
	./stringart-fast -input ae300.jpg -pins 400 -lines 20000 -output-size 2000 -weight 4 -output-weight 15
	open output.png
