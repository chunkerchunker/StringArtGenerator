#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <time.h>
#include "stringart_core.h"

typedef struct {
  char inputFile[256];
  char outputFile[256];
  int autoWeight;
} FileConfig;

void generateOutputImageWithFile(FastStringArtGenerator* gen,
                                 int* lineSequence,
                                 int lineCount,
                                 const char* outputFile) {
  unsigned char* image = generateOutputImage(gen, lineSequence, lineCount);

  int outSize = gen->config.outputSize;
  int centerOut = outSize / 2;
  int radiusOut = outSize / 2 - 1;
  drawCircle(image, centerOut, centerOut, radiusOut, outSize);

  stbi_write_png(outputFile, outSize, outSize, 4, image, outSize * 4);
  free(image);
}

int main(int argc, char* argv[]) {
  Config config = {.pins = 288,
                   .maxLines = 4000,
                   .targetSize = 500,
                   .outputSize = 0,
                   .lineWeight = 8,
                   .outputWeight = 0,
                   .minDistance = 10};

  FileConfig fileConfig = {
      .inputFile = "", .outputFile = "output.png", .autoWeight = 0};

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
    } else if (strcmp(argv[i], "-auto-weight") == 0) {
      fileConfig.autoWeight = 1;
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
  if (fileConfig.autoWeight) {
    printf("  Auto-weight mode enabled\n");
    printf(
        "  Pins: %d, Max lines: %d, Processing size: %d, Output size: %d, Line "
        "weight: auto, Output weight: %d, Min distance: %d\n",
        config.pins, config.maxLines, config.targetSize, config.outputSize,
        config.outputWeight, config.minDistance);
  } else {
    printf(
        "  Pins: %d, Max lines: %d, Processing size: %d, Output size: %d, Line "
        "weight: %d, Output weight: %d, Min distance: %d\n",
        config.pins, config.maxLines, config.targetSize, config.outputSize,
        config.lineWeight, config.outputWeight, config.minDistance);
  }

  FastStringArtGenerator gen;
  initGenerator(&gen, &config);

  clock_t start = clock();

  // Load image data for potential reuse in auto weight search
  int width, height, channels;
  unsigned char* img =
      stbi_load(fileConfig.inputFile, &width, &height, &channels, 0);
  if (!img) {
    fprintf(stderr, "Error loading image: %s\n", fileConfig.inputFile);
    exit(1);
  }

  if (fileConfig.autoWeight) {
    // Find optimal weight using binary search
    int optimalWeight =
        findOptimalLineWeight(&gen, img, width, height, channels, 1);
    config.lineWeight = optimalWeight;
    config.outputWeight =
        (config.outputWeight == 0) ? optimalWeight : config.outputWeight;
    printf("  Found optimal weight: %d\n", optimalWeight);
  } else {
    // Normal processing
    processImageData(&gen, img, width, height, channels);
    calculatePinCoords(&gen);
    precalculateAllPotentialLines(&gen);
  }

  printf("Calculating string art lines...\n");
  int lineCount;
  int* lineSequence = calculateLines(&gen, &lineCount);

  stbi_image_free(img);

  clock_t end = clock();
  double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Processing took %.2f seconds\n", cpu_time_used);

  printf("Generating output image...\n");
  generateOutputImageWithFile(&gen, lineSequence, lineCount,
                              fileConfig.outputFile);
  printf("Output saved to %s\n", fileConfig.outputFile);

  free(lineSequence);
  freeGenerator(&gen);

  return 0;
}