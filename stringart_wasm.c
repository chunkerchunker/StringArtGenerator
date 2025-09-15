#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stringart_core.h"
#include <emscripten.h>

static FastStringArtGenerator* g_gen = NULL;
static int* g_lineSequence = NULL;
static int g_lineCount = 0;
static Config g_lastConfig = {0};

EMSCRIPTEN_KEEPALIVE
int initStringArt(int pins, int maxLines, int targetSize, int outputSize,
                  int lineWeight, int outputWeight, int minDistance) {
    Config config = {
        .pins = pins,
        .maxLines = maxLines,
        .targetSize = targetSize,
        .outputSize = outputSize,
        .lineWeight = lineWeight,
        .outputWeight = outputWeight,
        .minDistance = minDistance
    };

    // Check if we can reuse existing pin/line cache
    int needsReinit = 0;
    if (!g_gen ||
        g_lastConfig.pins != config.pins ||
        g_lastConfig.targetSize != config.targetSize ||
        g_lastConfig.minDistance != config.minDistance) {
        needsReinit = 1;
    }

    if (needsReinit) {
        // Full reinitialization needed
        if (g_gen) {
            freeGenerator(g_gen);
            free(g_gen);
            g_gen = NULL;
        }

        g_gen = malloc(sizeof(FastStringArtGenerator));
        initGenerator(g_gen, &config);

        // Calculate pin coordinates and precalculate lines only when needed
        calculatePinCoords(g_gen);
        precalculateAllPotentialLines(g_gen);
    } else {
        // Just update the config parameters that don't affect pin/line cache
        g_gen->config.maxLines = config.maxLines;
        g_gen->config.outputSize = config.outputSize;
        g_gen->config.lineWeight = config.lineWeight;
        g_gen->config.outputWeight = config.outputWeight;
    }

    // Always clear line sequence from previous run
    if (g_lineSequence) {
        free(g_lineSequence);
        g_lineSequence = NULL;
    }

    g_lastConfig = config;
    return needsReinit ? 2 : 1; // Return 2 if reinitialized, 1 if reused cache
}

EMSCRIPTEN_KEEPALIVE
int processImage(unsigned char* imageData, int width, int height, int channels) {
    if (!g_gen) return 0;

    processImageData(g_gen, imageData, width, height, channels);

    if (g_lineSequence) {
        free(g_lineSequence);
    }
    g_lineSequence = calculateLines(g_gen, &g_lineCount);

    return g_lineCount;
}

EMSCRIPTEN_KEEPALIVE
unsigned char* getOutputImage() {
    if (!g_gen || !g_lineSequence) return NULL;
    return generateOutputImage(g_gen, g_lineSequence, g_lineCount);
}

EMSCRIPTEN_KEEPALIVE
int getOutputSize() {
    if (!g_gen) return 0;
    return g_gen->config.outputSize;
}

EMSCRIPTEN_KEEPALIVE
int getLineCount() {
    return g_lineCount;
}

EMSCRIPTEN_KEEPALIVE
int* getLineSequence() {
    return g_lineSequence;
}

EMSCRIPTEN_KEEPALIVE
void freeOutputImage(unsigned char* imageData) {
    if (imageData) {
        free(imageData);
    }
}

EMSCRIPTEN_KEEPALIVE
void cleanup() {
    if (g_gen) {
        freeGenerator(g_gen);
        free(g_gen);
        g_gen = NULL;
    }
    if (g_lineSequence) {
        free(g_lineSequence);
        g_lineSequence = NULL;
    }
    g_lineCount = 0;
    g_lastConfig = (Config){0}; // Reset last config
}