#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <time.h>
#include "stringart_core.h"

typedef struct {
  char inputFile[256];
  char outputFile[256];
  char outputPinsFile[256];
  int autoWeight;
  int noImage;
} FileConfig;

void printHelp(const char* programName) {
  printf("Usage: %s [options]\n\n", programName);
  printf("Options:\n");
  printf("  -input <file>          Input image file (required)\n");
  printf("  -output <file>         Output image file (default: output.png)\n");
  printf("  -output-pins <file>    Output pins sequence to text file\n");
  printf(
      "  -pins <number>         Number of pins around circle (default: 300)\n");
  printf("  -lines <number>        Maximum number of lines (default: 4000)\n");
  printf("  -size <number>         Processing image size (default: 500)\n");
  printf(
      "  -output-size <number>  Output image size (default: same as -size)\n");
  printf("  -weight <number>       Line weight for algorithm (default: 8)\n");
  printf(
      "  -output-weight <number> Line weight for output image (default: same "
      "as -weight)\n");
  printf("  -min-distance <number> Minimum pin distance (default: 10)\n");
  printf("  -auto-weight           Automatically find optimal line weight\n");
  printf("  --no-quantize          Disable int16 quantization (slower, default: on)\n");
  printf("  --no-image             Skip generating output image file\n");
  printf("  -q, --quiet            Suppress non-error output\n");
  printf("  -h, --help             Show this help message\n");
}

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
  Config config = {.pins = 300,
                   .maxLines = 4000,
                   .targetSize = 500,
                   .outputSize = 0,
                   .lineWeight = 8,
                   .outputWeight = 0,
                   .minDistance = 10,
                   .quiet = 0,
                   .useQuantized = 1};  // Default to quantized (2x faster on NEON)

  FileConfig fileConfig = {.inputFile = "",
                           .outputFile = "output.png",
                           .outputPinsFile = "",
                           .autoWeight = 0,
                           .noImage = 0};

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      printHelp(argv[0]);
      return 0;
    } else if (strcmp(argv[i], "-input") == 0 && i + 1 < argc) {
      strcpy(fileConfig.inputFile, argv[++i]);
    } else if (strcmp(argv[i], "-output") == 0 && i + 1 < argc) {
      strcpy(fileConfig.outputFile, argv[++i]);
    } else if (strcmp(argv[i], "-output-pins") == 0 && i + 1 < argc) {
      strcpy(fileConfig.outputPinsFile, argv[++i]);
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
    } else if (strcmp(argv[i], "--no-quantize") == 0) {
      config.useQuantized = 0;
    } else if (strcmp(argv[i], "--no-image") == 0) {
      fileConfig.noImage = 1;
    } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
      config.quiet = 1;
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

  if (!config.quiet) {
    printf("Processing %s...\n", fileConfig.inputFile);
    if (fileConfig.autoWeight) {
      printf("  Auto-weight mode enabled\n");
      printf(
          "  Pins: %d, Max lines: %d, Processing size: %d, Output size: %d, "
          "Line "
          "weight: auto, Output weight: %d, Min distance: %d\n",
          config.pins, config.maxLines, config.targetSize, config.outputSize,
          config.outputWeight, config.minDistance);
    } else {
      printf(
          "  Pins: %d, Max lines: %d, Processing size: %d, Output size: %d, "
          "Line "
          "weight: %d, Output weight: %d, Min distance: %d\n",
          config.pins, config.maxLines, config.targetSize, config.outputSize,
          config.lineWeight, config.outputWeight, config.minDistance);
    }
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
    int optimalWeight = findOptimalLineWeight(&gen, img, width, height,
                                              channels, !config.quiet);
    config.lineWeight = optimalWeight;
    config.outputWeight =
        (config.outputWeight == 0) ? optimalWeight : config.outputWeight;
    if (!config.quiet) {
      printf("  Found optimal weight: %d\n", optimalWeight);
    }
  } else {
    // Normal processing
    processImageData(&gen, img, width, height, channels);
    calculatePinCoords(&gen);
    precalculateAllPotentialLines(&gen);
  }

  if (!config.quiet) {
    printf("Calculating string art lines...\n");
  }
  int lineCount;
  int* lineSequence = calculateLines(&gen, &lineCount);

  stbi_image_free(img);

  clock_t end = clock();
  double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  if (!config.quiet) {
    printf("Processing took %.2f seconds\n", cpu_time_used);
  }

  if (!fileConfig.noImage) {
    if (!config.quiet) {
      printf("Generating output image...\n");
    }
    generateOutputImageWithFile(&gen, lineSequence, lineCount,
                                fileConfig.outputFile);
    if (!config.quiet) {
      printf("Output saved to %s\n", fileConfig.outputFile);
    }
  }

  if (strlen(fileConfig.outputPinsFile) > 0) {
    FILE* pinsFile = fopen(fileConfig.outputPinsFile, "w");
    if (pinsFile) {
      for (int i = 0; i < lineCount; i++) {
        fprintf(pinsFile, "%d\n", lineSequence[i]);
      }
      fclose(pinsFile);
      if (!config.quiet) {
        printf("Pins sequence saved to %s\n", fileConfig.outputPinsFile);
      }
    } else {
      fprintf(stderr, "Error: Could not write to %s\n",
              fileConfig.outputPinsFile);
    }
  }

  free(lineSequence);
  freeGenerator(&gen);

  return 0;
}