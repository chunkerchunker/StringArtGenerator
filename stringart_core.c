#include "stringart_core.h"

void initGenerator(FastStringArtGenerator* gen, Config* cfg) {
    gen->config = *cfg;
    gen->imgSize = cfg->targetSize;
    gen->imgSizeSq = gen->imgSize * gen->imgSize;

    gen->pinCoords = malloc(cfg->pins * sizeof(Coord));
    gen->sourceImage = malloc(gen->imgSizeSq * sizeof(float));
    gen->errorImage = malloc(gen->imgSizeSq * sizeof(float));

    int maxLines = cfg->pins * cfg->pins;
    gen->lineCache = malloc(maxLines * sizeof(FastLineCache));

    gen->validPins = malloc(cfg->pins * cfg->pins * sizeof(int));
    gen->validCounts = malloc(cfg->pins * sizeof(int));

    for (int i = 0; i < maxLines; i++) {
        gen->lineCache[i].coords = NULL;
        gen->lineCache[i].size = 0;
        gen->lineCache[i].errors = NULL;
    }
}

void freeGenerator(FastStringArtGenerator* gen) {
    if (!gen) return;

    free(gen->pinCoords);
    free(gen->sourceImage);
    free(gen->errorImage);
    free(gen->validPins);
    free(gen->validCounts);

    // Free line cache - be careful about shared pointers
    int maxLines = gen->config.pins * gen->config.pins;
    for (int i = 0; i < maxLines; i++) {
        if (gen->lineCache[i].coords) {
            free(gen->lineCache[i].coords);
            gen->lineCache[i].coords = NULL; // Prevent double-free
        }
        if (gen->lineCache[i].errors) {
            free(gen->lineCache[i].errors);
            gen->lineCache[i].errors = NULL; // Prevent double-free
        }
    }
    free(gen->lineCache);
}

void simpleResize(unsigned char* src, int srcW, int srcH, int channels,
                 unsigned char* dst, int dstW, int dstH) {
    int scaleX_fixed = (srcW << 16) / dstW;
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

void processImageData(FastStringArtGenerator* gen, unsigned char* imageData, int width, int height, int channels) {
    int size = (width < height) ? width : height;
    int startX = (width - size) / 2;
    int startY = (height - size) / 2;

    unsigned char* cropped = malloc(size * size * channels);
    for (int y = 0; y < size; y++) {
        memcpy(&cropped[y * size * channels],
               &imageData[(startY + y) * width * channels + startX * channels],
               size * channels);
    }

    unsigned char* resized = malloc(gen->imgSize * gen->imgSize * channels);
    simpleResize(cropped, size, size, channels, resized, gen->imgSize, gen->imgSize);

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
    for (int i = 0; i < gen->config.pins; i++) {
        int count = 0;
        for (int offset = gen->config.minDistance; offset < gen->config.pins - gen->config.minDistance; offset++) {
            int j = (i + offset) % gen->config.pins;
            gen->validPins[i * gen->config.pins + count] = j;
            count++;
        }
        gen->validCounts[i] = count;
    }

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

                // Only store the line once to avoid double-free issues
                // Always use the smaller index as the canonical entry
                int idx = (i < j) ? (i * gen->config.pins + j) : (j * gen->config.pins + i);

                gen->lineCache[idx].coords = coords;
                gen->lineCache[idx].size = steps;
                gen->lineCache[idx].errors = errors;
            }
        }
    }
}

static inline float getLineErrorFast(float* error, int* coords, int size) {
    if (size == 0) return 0.0f;

    float sum = 0.0f;
    int i = 0;

    for (; i + 3 < size; i += 4) {
        sum += error[coords[i]] + error[coords[i+1]] +
               error[coords[i+2]] + error[coords[i+3]];
    }

    for (; i < size; i++) {
        sum += error[coords[i]];
    }

    return sum / size;
}

static inline void updateErrorFast(float* error, int* coords, int size, float weight) {
    int i = 0;

    for (; i + 3 < size; i += 4) {
        error[coords[i]] -= weight;
        error[coords[i+1]] -= weight;
        error[coords[i+2]] -= weight;
        error[coords[i+3]] -= weight;
    }

    for (; i < size; i++) {
        error[coords[i]] -= weight;
    }
}

int* calculateLines(FastStringArtGenerator* gen, int* lineCount) {
    int* lineSequence = malloc((gen->config.maxLines + 1) * sizeof(int));
    lineSequence[0] = 0;
    *lineCount = 1;

    int currentPin = 0;
    int lastPins[20];
    for (int i = 0; i < 20; i++) lastPins[i] = -1;

    float lineWeight = (float)gen->config.lineWeight;

    for (int i = 0; i < gen->config.maxLines; i++) {
        int bestPin = -1;
        float maxErr = 0.0f;
        int bestIndex = 0;

        for (int k = 0; k < gen->validCounts[currentPin]; k++) {
            int testPin = gen->validPins[currentPin * gen->config.pins + k];

            int skip = 0;
            for (int j = 0; j < 20; j++) {
                if (lastPins[j] == testPin) {
                    skip = 1;
                    break;
                }
            }
            if (skip) continue;

            // Use consistent indexing (smaller pin first)
            int index = (currentPin < testPin) ?
                       (currentPin * gen->config.pins + testPin) :
                       (testPin * gen->config.pins + currentPin);
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

        updateErrorFast(gen->errorImage,
                       gen->lineCache[bestIndex].coords,
                       gen->lineCache[bestIndex].size,
                       lineWeight);

        memmove(lastPins, lastPins + 1, 19 * sizeof(int));
        lastPins[19] = bestPin;
        currentPin = bestPin;
    }

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

unsigned char* generateOutputImage(FastStringArtGenerator* gen, int* lineSequence, int lineCount) {
    int outSize = gen->config.outputSize;
    unsigned char* image = malloc(outSize * outSize * 4);

    memset(image, 255, outSize * outSize * 4);
    for (int i = 3; i < outSize * outSize * 4; i += 4) {
        image[i] = 255;
    }

    float scale = (float)outSize / gen->imgSize;

    for (int i = 0; i < gen->config.pins; i++) {
        int scaledX = (int)(gen->pinCoords[i].x * scale);
        int scaledY = (int)(gen->pinCoords[i].y * scale);
        drawFilledCircle(image, scaledX, scaledY, 2, outSize);
    }

    for (int i = 0; i < lineCount - 1; i++) {
        Coord from = gen->pinCoords[lineSequence[i]];
        Coord to = gen->pinCoords[lineSequence[i + 1]];

        int x0 = (int)(from.x * scale);
        int y0 = (int)(from.y * scale);
        int x1 = (int)(to.x * scale);
        int y1 = (int)(to.y * scale);

        drawLineAlpha(image, x0, y0, x1, y1, gen->config.outputWeight, outSize);
    }

    return image;
}