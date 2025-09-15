#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stringart_core.h"
#include <time.h>

typedef struct {
    char inputFile[256];
    char outputFile[256];
} FileConfig;

void loadAndProcessImage(FastStringArtGenerator* gen, const char* inputFile) {
    int width, height, channels;
    unsigned char* img = stbi_load(inputFile, &width, &height, &channels, 0);

    if (!img) {
        fprintf(stderr, "Error loading image: %s\n", inputFile);
        exit(1);
    }

    processImageData(gen, img, width, height, channels);

    if (width != gen->config.targetSize || height != gen->config.targetSize) {
        printf("  Resized from %dx%d to %dx%d\n", width, height, gen->imgSize, gen->imgSize);
    }

    stbi_image_free(img);
}

void generateOutputImageWithFile(FastStringArtGenerator* gen, int* lineSequence, int lineCount, const char* outputFile) {
    unsigned char* image = generateOutputImage(gen, lineSequence, lineCount);

    int outSize = gen->config.outputSize;
    int centerOut = outSize / 2;
    int radiusOut = outSize / 2 - 1;
    drawCircle(image, centerOut, centerOut, radiusOut, outSize);

    stbi_write_png(outputFile, outSize, outSize, 4, image, outSize * 4);
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
        .minDistance = 30
    };

    FileConfig fileConfig = {
        .inputFile = "",
        .outputFile = "output.png"
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-input") == 0 && i + 1 < argc) {
            strcpy(fileConfig.inputFile, argv[++i]);
        } else if (strcmp(argv[i], "-output") == 0 && i + 1 < argc) {
            strcpy(fileConfig.outputFile, argv[++i]);
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

    if (strlen(fileConfig.inputFile) == 0) {
        fprintf(stderr, "Please provide an input file using -input flag\n");
        return 1;
    }

    printf("Processing %s...\n", fileConfig.inputFile);
    printf("  Pins: %d, Max lines: %d, Processing size: %d, Output size: %d, Line weight: %d, Output weight: %d, Min distance: %d\n",
           config.pins, config.maxLines, config.targetSize, config.outputSize,
           config.lineWeight, config.outputWeight, config.minDistance);

    FastStringArtGenerator gen;
    initGenerator(&gen, &config);

    clock_t start = clock();

    loadAndProcessImage(&gen, fileConfig.inputFile);
    calculatePinCoords(&gen);
    precalculateAllPotentialLines(&gen);

    printf("Calculating string art lines...\n");
    int lineCount;
    int* lineSequence = calculateLines(&gen, &lineCount);

    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Processing took %.2f seconds\n", cpu_time_used);

    printf("Generating output image...\n");
    generateOutputImageWithFile(&gen, lineSequence, lineCount, fileConfig.outputFile);
    printf("Output saved to %s\n", fileConfig.outputFile);

    free(lineSequence);
    freeGenerator(&gen);

    return 0;
}