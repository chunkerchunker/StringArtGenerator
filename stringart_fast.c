#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
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
    char inputFile[256];
    char outputFile[256];
} Config;

// Optimized line data structure
typedef struct {
    int* coords;        // Flattened 1D indices
    int size;
    float* errors;      // Pre-allocated for error calculations
} FastLineCache;

typedef struct {
    Config config;
    int imgSize;
    int imgSizeSq;

    Coord* pinCoords;
    float* sourceImage;
    float* errorImage;
    FastLineCache* lineCache;

    // Cache optimization: pre-compute valid pin ranges
    int* validPins;     // Array of valid next pins for each current pin
    int* validCounts;   // Count of valid pins for each current pin
} FastStringArtGenerator;

void initGenerator(FastStringArtGenerator* gen, Config* cfg) {
    gen->config = *cfg;
    gen->imgSize = cfg->targetSize;
    gen->imgSizeSq = gen->imgSize * gen->imgSize;

    gen->pinCoords = malloc(cfg->pins * sizeof(Coord));
    gen->sourceImage = malloc(gen->imgSizeSq * sizeof(float));
    gen->errorImage = malloc(gen->imgSizeSq * sizeof(float));

    // Pre-allocate line cache
    int maxLines = cfg->pins * cfg->pins;
    gen->lineCache = malloc(maxLines * sizeof(FastLineCache));

    // Pre-allocate valid pin lookup tables
    gen->validPins = malloc(cfg->pins * cfg->pins * sizeof(int));
    gen->validCounts = malloc(cfg->pins * sizeof(int));

    for (int i = 0; i < maxLines; i++) {
        gen->lineCache[i].coords = NULL;
        gen->lineCache[i].size = 0;
        gen->lineCache[i].errors = NULL;
    }
}

void freeGenerator(FastStringArtGenerator* gen) {
    free(gen->pinCoords);
    free(gen->sourceImage);
    free(gen->errorImage);
    free(gen->validPins);
    free(gen->validCounts);

    int maxLines = gen->config.pins * gen->config.pins;
    for (int i = 0; i < maxLines; i++) {
        if (gen->lineCache[i].coords) free(gen->lineCache[i].coords);
        if (gen->lineCache[i].errors) free(gen->lineCache[i].errors);
    }
    free(gen->lineCache);
}

void simpleResize(unsigned char* src, int srcW, int srcH, int channels,
                 unsigned char* dst, int dstW, int dstH) {
    // Optimized with fewer divisions
    int scaleX_fixed = (srcW << 16) / dstW;  // Fixed-point scaling
    int scaleY_fixed = (srcH << 16) / dstH;

    for (int y = 0; y < dstH; y++) {
        int srcY = (y * scaleY_fixed) >> 16;
        for (int x = 0; x < dstW; x++) {
            int srcX = (x * scaleX_fixed) >> 16;

            int srcIdx = (srcY * srcW + srcX) * channels;
            int dstIdx = (y * dstW + x) * channels;

            for (int c = 0; c < channels; c++) {
                dst[dstIdx + c] = src[srcIdx + c];
            }
        }
    }
}

void loadAndProcessImage(FastStringArtGenerator* gen) {
    int width, height, channels;
    unsigned char* img = stbi_load(gen->config.inputFile, &width, &height, &channels, 0);

    if (!img) {
        fprintf(stderr, "Error loading image: %s\n", gen->config.inputFile);
        exit(1);
    }

    int size = (width < height) ? width : height;
    int startX = (width - size) / 2;
    int startY = (height - size) / 2;

    unsigned char* cropped = malloc(size * size * channels);
    for (int y = 0; y < size; y++) {
        memcpy(&cropped[y * size * channels],
               &img[(startY + y) * width * channels + startX * channels],
               size * channels);
    }

    unsigned char* resized = malloc(gen->imgSize * gen->imgSize * channels);
    simpleResize(cropped, size, size, channels, resized, gen->imgSize, gen->imgSize);

    if (width != gen->config.targetSize || height != gen->config.targetSize) {
        printf("  Resized from %dx%d to %dx%d\n", width, height, gen->imgSize, gen->imgSize);
    }

    // Optimized luminosity conversion - unrolled
    const float r_coeff = 0.2126f;
    const float g_coeff = 0.7152f;
    const float b_coeff = 0.0722f;

    for (int i = 0; i < gen->imgSizeSq; i++) {
        int idx = i * channels;
        float r = resized[idx];
        float g = (channels > 1) ? resized[idx + 1] : r;
        float b = (channels > 2) ? resized[idx + 2] : r;

        float luminosity = r_coeff * r + g_coeff * g + b_coeff * b;
        gen->sourceImage[i] = luminosity;
        gen->errorImage[i] = 255.0f - luminosity;
    }

    stbi_image_free(img);
    free(cropped);
    free(resized);
}

void calculatePinCoords(FastStringArtGenerator* gen) {
    float center = gen->imgSize * 0.5f;
    float radius = center - 1.0f;
    float angleStep = 2.0f * M_PI / gen->config.pins;

    for (int i = 0; i < gen->config.pins; i++) {
        float angle = angleStep * i;
        gen->pinCoords[i].x = floorf(center + radius * cosf(angle));
        gen->pinCoords[i].y = floorf(center + radius * sinf(angle));
    }
}

void precalculateAllPotentialLines(FastStringArtGenerator* gen) {
    // Build valid pin lookup table first
    for (int i = 0; i < gen->config.pins; i++) {
        int count = 0;
        for (int offset = gen->config.minDistance; offset < gen->config.pins - gen->config.minDistance; offset++) {
            int j = (i + offset) % gen->config.pins;
            gen->validPins[i * gen->config.pins + count] = j;
            count++;
        }
        gen->validCounts[i] = count;
    }

    // Now precalculate lines
    for (int i = 0; i < gen->config.pins; i++) {
        for (int k = 0; k < gen->validCounts[i]; k++) {
            int j = gen->validPins[i * gen->config.pins + k];

            float x0 = gen->pinCoords[i].x;
            float y0 = gen->pinCoords[i].y;
            float x1 = gen->pinCoords[j].x;
            float y1 = gen->pinCoords[j].y;

            float dx = x1 - x0;
            float dy = y1 - y0;
            int steps = (int)sqrtf(dx * dx + dy * dy);

            if (steps > 0) {
                int* coords = malloc(steps * sizeof(int));
                float* errors = malloc(steps * sizeof(float));

                float invSteps = 1.0f / (steps - 1);
                for (int s = 0; s < steps; s++) {
                    float t = s * invSteps;
                    int x = (int)(x0 + t * dx);
                    int y = (int)(y0 + t * dy);
                    coords[s] = y * gen->imgSize + x;
                }

                int idx1 = j * gen->config.pins + i;
                int idx2 = i * gen->config.pins + j;

                gen->lineCache[idx1].coords = coords;
                gen->lineCache[idx1].size = steps;
                gen->lineCache[idx1].errors = errors;

                gen->lineCache[idx2].coords = coords;
                gen->lineCache[idx2].size = steps;
                gen->lineCache[idx2].errors = errors;
            }
        }
    }
}

// Ultra-fast line error calculation with unrolled loop
static inline float getLineErrorFast(float* error, int* coords, int size) {
    if (size == 0) return 0.0f;

    float sum = 0.0f;
    int i = 0;

    // Unroll loop by 4
    for (; i + 3 < size; i += 4) {
        sum += error[coords[i]] + error[coords[i+1]] +
               error[coords[i+2]] + error[coords[i+3]];
    }

    // Handle remaining elements
    for (; i < size; i++) {
        sum += error[coords[i]];
    }

    return sum / size;
}

// Fast error update with unrolled loop
static inline void updateErrorFast(float* error, int* coords, int size, float weight) {
    int i = 0;

    // Unroll loop by 4
    for (; i + 3 < size; i += 4) {
        error[coords[i]] -= weight;
        error[coords[i+1]] -= weight;
        error[coords[i+2]] -= weight;
        error[coords[i+3]] -= weight;
    }

    // Handle remaining elements
    for (; i < size; i++) {
        error[coords[i]] -= weight;
    }
}

int* calculateLines(FastStringArtGenerator* gen, int* lineCount) {
    printf("Calculating string art lines...\n");

    int* lineSequence = malloc((gen->config.maxLines + 1) * sizeof(int));
    lineSequence[0] = 0;
    *lineCount = 1;

    int currentPin = 0;
    int lastPins[20];
    for (int i = 0; i < 20; i++) lastPins[i] = -1;

    float lineWeight = (float)gen->config.lineWeight;

    for (int i = 0; i < gen->config.maxLines; i++) {
        if (i % 100 == 0) {
            printf("  Processing line %d/%d\r", i, gen->config.maxLines);
            fflush(stdout);
        }

        int bestPin = -1;
        float maxErr = 0.0f;
        int bestIndex = 0;

        // Use pre-computed valid pins
        for (int k = 0; k < gen->validCounts[currentPin]; k++) {
            int testPin = gen->validPins[currentPin * gen->config.pins + k];

            // Quick check if pin was recently used
            int skip = 0;
            for (int j = 0; j < 20; j++) {
                if (lastPins[j] == testPin) {
                    skip = 1;
                    break;
                }
            }
            if (skip) continue;

            int index = testPin * gen->config.pins + currentPin;
            float lineErr = getLineErrorFast(gen->errorImage,
                                           gen->lineCache[index].coords,
                                           gen->lineCache[index].size);

            if (lineErr > maxErr) {
                maxErr = lineErr;
                bestPin = testPin;
                bestIndex = index;
            }
        }

        if (bestPin == -1) break;

        lineSequence[*lineCount] = bestPin;
        (*lineCount)++;

        // Update error map
        updateErrorFast(gen->errorImage,
                       gen->lineCache[bestIndex].coords,
                       gen->lineCache[bestIndex].size,
                       lineWeight);

        // Update recent pins list efficiently
        memmove(lastPins, lastPins + 1, 19 * sizeof(int));
        lastPins[19] = bestPin;
        currentPin = bestPin;
    }

    printf("\n");
    return lineSequence;
}

void drawLineAlpha(unsigned char* img, int x0, int y0, int x1, int y1, int alpha, int size) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    float alphaF = alpha / 255.0f;
    float invAlpha = 1.0f - alphaF;

    while (1) {
        if (x0 >= 0 && x0 < size && y0 >= 0 && y0 < size) {
            int idx = (y0 * size + x0) * 4;
            img[idx] = (unsigned char)(img[idx] * invAlpha);
            img[idx + 1] = (unsigned char)(img[idx + 1] * invAlpha);
            img[idx + 2] = (unsigned char)(img[idx + 2] * invAlpha);
        }

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

void drawCircle(unsigned char* img, int cx, int cy, int r, int size) {
    for (float angle = 0; angle < 2 * M_PI; angle += 0.01f) {
        int x = cx + (int)(r * cosf(angle));
        int y = cy + (int)(r * sinf(angle));
        if (x >= 0 && x < size && y >= 0 && y < size) {
            int idx = (y * size + x) * 4;
            img[idx] = img[idx + 1] = img[idx + 2] = 0;
        }
    }
}

void drawFilledCircle(unsigned char* img, int cx, int cy, int r, int size) {
    for (int x = -r; x <= r; x++) {
        for (int y = -r; y <= r; y++) {
            if (x * x + y * y <= r * r) {
                int px = cx + x;
                int py = cy + y;
                if (px >= 0 && px < size && py >= 0 && py < size) {
                    int idx = (py * size + px) * 4;
                    img[idx] = img[idx + 1] = img[idx + 2] = 0;
                }
            }
        }
    }
}

void generateOutputImage(FastStringArtGenerator* gen, int* lineSequence, int lineCount) {
    int outSize = gen->config.outputSize;
    unsigned char* image = malloc(outSize * outSize * 4);

    // Fast initialization
    memset(image, 255, outSize * outSize * 4);
    for (int i = 3; i < outSize * outSize * 4; i += 4) {
        image[i] = 255;  // Alpha channel
    }

    float scale = (float)outSize / gen->imgSize;

    // Draw circle border
    int centerOut = outSize / 2;
    int radiusOut = outSize / 2 - 1;
    drawCircle(image, centerOut, centerOut, radiusOut, outSize);

    // Draw pins
    for (int i = 0; i < gen->config.pins; i++) {
        int scaledX = (int)(gen->pinCoords[i].x * scale);
        int scaledY = (int)(gen->pinCoords[i].y * scale);
        drawFilledCircle(image, scaledX, scaledY, 2, outSize);
    }

    // Draw lines
    for (int i = 0; i < lineCount - 1; i++) {
        Coord from = gen->pinCoords[lineSequence[i]];
        Coord to = gen->pinCoords[lineSequence[i + 1]];

        int x0 = (int)(from.x * scale);
        int y0 = (int)(from.y * scale);
        int x1 = (int)(to.x * scale);
        int y1 = (int)(to.y * scale);

        drawLineAlpha(image, x0, y0, x1, y1, gen->config.outputWeight, outSize);
    }

    stbi_write_png(gen->config.outputFile, outSize, outSize, 4, image, outSize * 4);
    free(image);
}

int main(int argc, char* argv[]) {
    Config config = {
        .pins = 300,
        .maxLines = 4000,
        .targetSize = 500,
        .outputSize = 0,
        .lineWeight = 8,
        .outputWeight = 0,
        .minDistance = 30,
        .inputFile = "",
        .outputFile = "output.png"
    };

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-input") == 0 && i + 1 < argc) {
            strcpy(config.inputFile, argv[++i]);
        } else if (strcmp(argv[i], "-output") == 0 && i + 1 < argc) {
            strcpy(config.outputFile, argv[++i]);
        } else if (strcmp(argv[i], "-pins") == 0 && i + 1 < argc) {
            config.pins = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-lines") == 0 && i + 1 < argc) {
            config.maxLines = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-size") == 0 && i + 1 < argc) {
            config.targetSize = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-output-size") == 0 && i + 1 < argc) {
            config.outputSize = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-weight") == 0 && i + 1 < argc) {
            config.lineWeight = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-output-weight") == 0 && i + 1 < argc) {
            config.outputWeight = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-min-distance") == 0 && i + 1 < argc) {
            config.minDistance = atoi(argv[++i]);
        }
    }

    if (config.outputSize == 0) {
        config.outputSize = config.targetSize;
    }
    if (config.outputWeight == 0) {
        config.outputWeight = config.lineWeight;
    }

    if (strlen(config.inputFile) == 0) {
        fprintf(stderr, "Please provide an input file using -input flag\n");
        return 1;
    }

    printf("Processing %s...\n", config.inputFile);
    printf("  Pins: %d, Max lines: %d, Processing size: %d, Output size: %d, Line weight: %d, Output weight: %d, Min distance: %d\n",
           config.pins, config.maxLines, config.targetSize, config.outputSize,
           config.lineWeight, config.outputWeight, config.minDistance);

    FastStringArtGenerator gen;
    initGenerator(&gen, &config);

    clock_t start = clock();

    loadAndProcessImage(&gen);
    calculatePinCoords(&gen);
    precalculateAllPotentialLines(&gen);

    int lineCount;
    int* lineSequence = calculateLines(&gen, &lineCount);

    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Processing took %.2f seconds\n", cpu_time_used);

    printf("Generating output image...\n");
    generateOutputImage(&gen, lineSequence, lineCount);
    printf("Output saved to %s\n", config.outputFile);

    free(lineSequence);
    freeGenerator(&gen);

    return 0;
}