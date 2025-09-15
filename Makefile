.PHONY: all build build-c build-fast clean test test-c test-fast

CC = clang
CFLAGS = -std=c99 -O3 -march=native -Wall -Wextra
CFLAGS_FAST = -std=c99 -O3 -march=native -funroll-loops -ffast-math -finline-functions -Wall -Wextra
LDFLAGS = -lm

all: build build-c build-fast

build:
	go build -o stringart main.go

build-c: stringart-c

build-fast: stringart-fast

stringart-c: stringart.c stb_image.h stb_image_write.h
	$(CC) $(CFLAGS) stringart.c -o stringart-c $(LDFLAGS)

stringart-fast: stringart_fast.c stb_image.h stb_image_write.h
	$(CC) $(CFLAGS_FAST) stringart_fast.c -o stringart-fast $(LDFLAGS)

clean:
	rm -f stringart stringart-c stringart-fast

test: build
	./stringart -input ae300.jpg -pins 400 -lines 20000 -output-size 2000 -weight 4 -output-weight 15
	open output.png

test-c: build-c
	./stringart-c -input ae300.jpg -pins 400 -lines 20000 -output-size 2000 -weight 4 -output-weight 15
	open output.png

test-fast: build-fast
	./stringart-fast -input ae300.jpg -pins 400 -lines 20000 -output-size 2000 -weight 4 -output-weight 15
	open output.png
