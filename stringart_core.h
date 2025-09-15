#ifndef STRINGART_CORE_H
#define STRINGART_CORE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "stb_image.h"
#include "stb_image_write.h"

typedef struct {
    float x;
    float y;
} Coord;

typedef struct {
    int pins;
    int maxLines;
    int targetSize;
    int outputSize;
    int lineWeight;
    int outputWeight;
    int minDistance;
} Config;

typedef struct {
    int* coords;
    int size;
    float* errors;
} FastLineCache;

typedef struct {
    Config config;
    int imgSize;
    int imgSizeSq;

    Coord* pinCoords;
    float* sourceImage;
    float* errorImage;
    FastLineCache* lineCache;

    int* validPins;
    int* validCounts;
} FastStringArtGenerator;

// Core algorithm functions (shared between all versions)
void initGenerator(FastStringArtGenerator* gen, Config* cfg);
void freeGenerator(FastStringArtGenerator* gen);
void simpleResize(unsigned char* src, int srcW, int srcH, int channels,
                 unsigned char* dst, int dstW, int dstH);
void processImageData(FastStringArtGenerator* gen, unsigned char* imageData,
                     int width, int height, int channels);
void calculatePinCoords(FastStringArtGenerator* gen);
void precalculateAllPotentialLines(FastStringArtGenerator* gen);
int* calculateLines(FastStringArtGenerator* gen, int* lineCount);
unsigned char* generateOutputImage(FastStringArtGenerator* gen, int* lineSequence, int lineCount);

// Drawing functions
void drawLineAlpha(unsigned char* img, int x0, int y0, int x1, int y1, int alpha, int size);
void drawFilledCircle(unsigned char* img, int cx, int cy, int r, int size);

// Drawing functions (extra - not in core)
void drawCircle(unsigned char* img, int cx, int cy, int r, int size);

#endif // STRINGART_CORE_H