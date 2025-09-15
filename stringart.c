#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
// Using simple nearest neighbor resize instead of stb_image_resize
// to avoid dependency issues

typedef struct {
    double x;
    double y;
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

typedef struct {
    Config config;
    int imgSize;
    double imgSizeF;
    int imgSizeSq;

    Coord* pinCoords;
    double* sourceImage;
    int** lineCacheX;
    int** lineCacheY;
    int* lineCacheSizes;
} StringArtGenerator;

void initGenerator(StringArtGenerator* gen, Config* cfg) {
    gen->config = *cfg;
    gen->imgSize = cfg->targetSize;
    gen->imgSizeF = (double)gen->imgSize;
    gen->imgSizeSq = gen->imgSize * gen->imgSize;

    gen->pinCoords = malloc(cfg->pins * sizeof(Coord));
    gen->sourceImage = malloc(gen->imgSizeSq * sizeof(double));

    int cacheSize = cfg->pins * cfg->pins;
    gen->lineCacheX = malloc(cacheSize * sizeof(int*));
    gen->lineCacheY = malloc(cacheSize * sizeof(int*));
    gen->lineCacheSizes = malloc(cacheSize * sizeof(int));

    for (int i = 0; i < cacheSize; i++) {
        gen->lineCacheX[i] = NULL;
        gen->lineCacheY[i] = NULL;
        gen->lineCacheSizes[i] = 0;
    }
}

void simpleResize(unsigned char* src, int srcW, int srcH, int channels,
                 unsigned char* dst, int dstW, int dstH) {
    for (int y = 0; y < dstH; y++) {
        for (int x = 0; x < dstW; x++) {
            int srcX = (x * srcW) / dstW;
            int srcY = (y * srcH) / dstH;

            for (int c = 0; c < channels; c++) {
                dst[(y * dstW + x) * channels + c] =
                    src[(srcY * srcW + srcX) * channels + c];
            }
        }
    }
}

void freeGenerator(StringArtGenerator* gen) {
    free(gen->pinCoords);
    free(gen->sourceImage);

    int cacheSize = gen->config.pins * gen->config.pins;
    for (int i = 0; i < cacheSize; i++) {
        if (gen->lineCacheX[i]) free(gen->lineCacheX[i]);
        if (gen->lineCacheY[i]) free(gen->lineCacheY[i]);
    }
    free(gen->lineCacheX);
    free(gen->lineCacheY);
    free(gen->lineCacheSizes);
}

void loadAndProcessImage(StringArtGenerator* gen) {
    int width, height, channels;
    unsigned char* img = stbi_load(gen->config.inputFile, &width, &height, &channels, 0);

    if (!img) {
        fprintf(stderr, "Error loading image: %s\n", gen->config.inputFile);
        exit(1);
    }

    // Find minimum dimension and crop to square
    int size = (width < height) ? width : height;
    int startX = (width - size) / 2;
    int startY = (height - size) / 2;

    // Create temporary cropped image
    unsigned char* cropped = malloc(size * size * channels);
    for (int y = 0; y < size; y++) {
        memcpy(&cropped[y * size * channels],
               &img[(startY + y) * width * channels + startX * channels],
               size * channels);
    }

    // Resize to target size
    unsigned char* resized = malloc(gen->imgSize * gen->imgSize * channels);
    simpleResize(cropped, size, size, channels, resized, gen->imgSize, gen->imgSize);

    if (width != gen->config.targetSize || height != gen->config.targetSize) {
        printf("  Resized from %dx%d to %dx%d\n", width, height, gen->imgSize, gen->imgSize);
    }

    // Convert to grayscale luminosity
    for (int i = 0; i < gen->imgSizeSq; i++) {
        int idx = i * channels;
        double r = resized[idx];
        double g = (channels > 1) ? resized[idx + 1] : r;
        double b = (channels > 2) ? resized[idx + 2] : r;

        // ITU-R BT.709 luminosity formula
        gen->sourceImage[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    stbi_image_free(img);
    free(cropped);
    free(resized);
}

void calculatePinCoords(StringArtGenerator* gen) {
    double center = gen->imgSize / 2.0;
    double radius = gen->imgSize / 2.0 - 1;

    for (int i = 0; i < gen->config.pins; i++) {
        double angle = 2 * M_PI * i / gen->config.pins;
        gen->pinCoords[i].x = floor(center + radius * cos(angle));
        gen->pinCoords[i].y = floor(center + radius * sin(angle));
    }
}

void precalculateAllPotentialLines(StringArtGenerator* gen) {
    for (int i = 0; i < gen->config.pins; i++) {
        for (int j = i + gen->config.minDistance; j < gen->config.pins; j++) {
            double x0 = gen->pinCoords[i].x;
            double y0 = gen->pinCoords[i].y;
            double x1 = gen->pinCoords[j].x;
            double y1 = gen->pinCoords[j].y;

            double dx = x1 - x0;
            double dy = y1 - y0;
            int steps = (int)sqrt(dx * dx + dy * dy);

            if (steps > 0) {
                int* xs = malloc(steps * sizeof(int));
                int* ys = malloc(steps * sizeof(int));

                for (int k = 0; k < steps; k++) {
                    double t = (double)k / (steps - 1);
                    xs[k] = (int)(x0 + t * dx);
                    ys[k] = (int)(y0 + t * dy);
                }

                int idx1 = j * gen->config.pins + i;
                int idx2 = i * gen->config.pins + j;

                gen->lineCacheX[idx1] = xs;
                gen->lineCacheX[idx2] = xs;
                gen->lineCacheY[idx1] = ys;
                gen->lineCacheY[idx2] = ys;
                gen->lineCacheSizes[idx1] = steps;
                gen->lineCacheSizes[idx2] = steps;
            }
        }
    }
}

double getLineError(StringArtGenerator* gen, double* error, int cacheIndex) {
    int* xs = gen->lineCacheX[cacheIndex];
    int* ys = gen->lineCacheY[cacheIndex];
    int size = gen->lineCacheSizes[cacheIndex];

    if (!xs || size == 0) return 0.0;

    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += error[ys[i] * gen->imgSize + xs[i]];
    }
    return sum / size;
}

int* calculateLines(StringArtGenerator* gen, int* lineCount) {
    printf("Calculating string art lines...\n");

    double* error = malloc(gen->imgSizeSq * sizeof(double));
    for (int i = 0; i < gen->imgSizeSq; i++) {
        error[i] = 255.0 - gen->sourceImage[i];
    }

    int* lineSequence = malloc((gen->config.maxLines + 1) * sizeof(int));
    lineSequence[0] = 0;
    *lineCount = 1;

    int currentPin = 0;
    int lastPins[20];
    for (int i = 0; i < 20; i++) lastPins[i] = -1;

    for (int i = 0; i < gen->config.maxLines; i++) {
        if (i % 100 == 0) {
            printf("  Processing line %d/%d\r", i, gen->config.maxLines);
            fflush(stdout);
        }

        int bestPin = -1;
        double maxErr = 0.0;
        int bestIndex = 0;

        // Find best next pin
        for (int offset = gen->config.minDistance; offset < gen->config.pins - gen->config.minDistance; offset++) {
            int testPin = (currentPin + offset) % gen->config.pins;

            // Check if pin was recently used
            int skip = 0;
            for (int j = 0; j < 20; j++) {
                if (lastPins[j] == testPin) {
                    skip = 1;
                    break;
                }
            }
            if (skip) continue;

            int index = testPin * gen->config.pins + currentPin;
            double lineErr = getLineError(gen, error, index);

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
        int* xs = gen->lineCacheX[bestIndex];
        int* ys = gen->lineCacheY[bestIndex];
        int size = gen->lineCacheSizes[bestIndex];
        for (int j = 0; j < size; j++) {
            int idx = ys[j] * gen->imgSize + xs[j];
            error[idx] -= gen->config.lineWeight;
        }

        // Update recent pins list
        for (int j = 0; j < 19; j++) {
            lastPins[j] = lastPins[j + 1];
        }
        lastPins[19] = bestPin;
        currentPin = bestPin;
    }

    printf("\n");
    free(error);
    return lineSequence;
}

void drawLineAlpha(unsigned char* img, int x0, int y0, int x1, int y1, int alpha, int size) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    double alphaF = alpha / 255.0;
    double invAlpha = 1.0 - alphaF;

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
    for (double angle = 0; angle < 2 * M_PI; angle += 0.01) {
        int x = cx + (int)(r * cos(angle));
        int y = cy + (int)(r * sin(angle));
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

void generateOutputImage(StringArtGenerator* gen, int* lineSequence, int lineCount) {
    int outSize = gen->config.outputSize;
    unsigned char* image = malloc(outSize * outSize * 4);

    // Initialize to white with full alpha
    for (int i = 0; i < outSize * outSize * 4; i += 4) {
        image[i] = image[i + 1] = image[i + 2] = 255;
        image[i + 3] = 255;
    }

    double scale = (double)outSize / gen->imgSize;

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
        Coord from, to;
        int x0, y0, x1, y1;

        from = gen->pinCoords[lineSequence[i]];
        to = gen->pinCoords[lineSequence[i + 1]];

        x0 = (int)(from.x * scale);
        y0 = (int)(from.y * scale);
        x1 = (int)(to.x * scale);
        y1 = (int)(to.y * scale);

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

    StringArtGenerator gen;
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