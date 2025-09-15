#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

struct Coord {
    double x;
    double y;
};

struct Config {
    int pins = 300;
    int maxLines = 4000;
    int targetSize = 500;
    int outputSize = 500;
    int lineWeight = 8;
    int outputWeight = 8;
    int minDistance = 30;
    std::string inputFile;
    std::string outputFile = "output.png";
};

class StringArtGenerator {
private:
    Config config;
    int imgSize;
    double imgSizeF;
    int imgSizeSq;

    std::vector<Coord> pinCoords;
    std::vector<double> sourceImage;
    std::vector<std::vector<int>> lineCacheX;
    std::vector<std::vector<int>> lineCacheY;

public:
    StringArtGenerator(const Config& cfg) : config(cfg) {
        imgSize = config.targetSize;
        imgSizeF = static_cast<double>(imgSize);
        imgSizeSq = imgSize * imgSize;
    }

    void run() {
        std::cout << "Processing " << config.inputFile << "...\n";
        std::cout << "  Pins: " << config.pins << ", Max lines: " << config.maxLines
                  << ", Processing size: " << config.targetSize
                  << ", Output size: " << config.outputSize
                  << ", Line weight: " << config.lineWeight
                  << ", Output weight: " << config.outputWeight
                  << ", Min distance: " << config.minDistance << "\n";

        auto start = std::chrono::high_resolution_clock::now();

        loadAndProcessImage();
        calculatePinCoords();
        precalculateAllPotentialLines();
        auto lineSequence = calculateLines();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Processing took " << diff.count() << " seconds\n";

        std::cout << "Generating output image...\n";
        generateOutputImage(lineSequence);
        std::cout << "Output saved to " << config.outputFile << "\n";
    }

private:
    void loadAndProcessImage() {
        int width, height, channels;
        unsigned char* img = stbi_load(config.inputFile.c_str(), &width, &height, &channels, 0);

        if (!img) {
            std::cerr << "Error loading image: " << config.inputFile << "\n";
            exit(1);
        }

        // Find minimum dimension and crop to square
        int size = std::min(width, height);
        int startX = (width - size) / 2;
        int startY = (height - size) / 2;

        // Create temporary cropped image
        std::vector<unsigned char> cropped(size * size * channels);
        for (int y = 0; y < size; ++y) {
            std::memcpy(&cropped[y * size * channels],
                       &img[(startY + y) * width * channels + startX * channels],
                       size * channels);
        }

        // Resize to target size
        std::vector<unsigned char> resized(imgSize * imgSize * channels);
        stbir_resize_uint8(&cropped[0], size, size, 0,
                          &resized[0], imgSize, imgSize, 0, channels);

        if (width != config.targetSize || height != config.targetSize) {
            std::cout << "  Resized from " << width << "x" << height
                     << " to " << imgSize << "x" << imgSize << "\n";
        }

        // Convert to grayscale luminosity
        sourceImage.resize(imgSizeSq);

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < imgSizeSq; ++i) {
            int idx = i * channels;
            double r = resized[idx];
            double g = (channels > 1) ? resized[idx + 1] : r;
            double b = (channels > 2) ? resized[idx + 2] : r;

            // ITU-R BT.709 luminosity formula
            sourceImage[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        }

        stbi_image_free(img);
    }

    void calculatePinCoords() {
        pinCoords.resize(config.pins);
        double center = imgSize / 2.0;
        double radius = imgSize / 2.0 - 1;

        for (int i = 0; i < config.pins; ++i) {
            double angle = 2 * M_PI * i / config.pins;
            pinCoords[i].x = std::floor(center + radius * std::cos(angle));
            pinCoords[i].y = std::floor(center + radius * std::sin(angle));
        }
    }

    void precalculateAllPotentialLines() {
        int cacheSize = config.pins * config.pins;
        lineCacheX.resize(cacheSize);
        lineCacheY.resize(cacheSize);

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2)
        #endif
        for (int i = 0; i < config.pins; ++i) {
            for (int j = i + config.minDistance; j < config.pins; ++j) {
                double x0 = pinCoords[i].x;
                double y0 = pinCoords[i].y;
                double x1 = pinCoords[j].x;
                double y1 = pinCoords[j].y;

                double dx = x1 - x0;
                double dy = y1 - y0;
                int steps = static_cast<int>(std::sqrt(dx * dx + dy * dy));

                std::vector<int> xs(steps);
                std::vector<int> ys(steps);

                for (int k = 0; k < steps; ++k) {
                    double t = static_cast<double>(k) / (steps - 1);
                    xs[k] = static_cast<int>(x0 + t * dx);
                    ys[k] = static_cast<int>(y0 + t * dy);
                }

                int idx1 = j * config.pins + i;
                int idx2 = i * config.pins + j;

                lineCacheX[idx1] = xs;
                lineCacheX[idx2] = xs;
                lineCacheY[idx1] = ys;
                lineCacheY[idx2] = ys;
            }
        }
    }

    double getLineError(const std::vector<double>& error, int cacheIndex) {
        const auto& xs = lineCacheX[cacheIndex];
        const auto& ys = lineCacheY[cacheIndex];

        if (xs.empty()) return 0.0;

        double sum = 0.0;
        for (size_t i = 0; i < xs.size(); ++i) {
            sum += error[ys[i] * imgSize + xs[i]];
        }
        return sum / xs.size();
    }

    std::vector<int> calculateLines() {
        std::cout << "Calculating string art lines...\n";

        std::vector<double> error(imgSizeSq);
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < imgSizeSq; ++i) {
            error[i] = 255.0 - sourceImage[i];
        }

        std::vector<int> lineSequence;
        lineSequence.reserve(config.maxLines + 1);
        lineSequence.push_back(0);

        int currentPin = 0;
        std::vector<int> lastPins(20, -1);

        for (int i = 0; i < config.maxLines; ++i) {
            if (i % 100 == 0) {
                std::cout << "  Processing line " << i << "/" << config.maxLines << "\r" << std::flush;
            }

            int bestPin = -1;
            double maxErr = 0.0;
            int bestIndex = 0;

            // Find best next pin
            for (int offset = config.minDistance; offset < config.pins - config.minDistance; ++offset) {
                int testPin = (currentPin + offset) % config.pins;

                // Check if pin was recently used
                bool skip = false;
                for (int recentPin : lastPins) {
                    if (recentPin == testPin) {
                        skip = true;
                        break;
                    }
                }
                if (skip) continue;

                int index = testPin * config.pins + currentPin;
                double lineErr = getLineError(error, index);

                if (lineErr > maxErr) {
                    maxErr = lineErr;
                    bestPin = testPin;
                    bestIndex = index;
                }
            }

            if (bestPin == -1) break;

            lineSequence.push_back(bestPin);

            // Update error map
            const auto& xs = lineCacheX[bestIndex];
            const auto& ys = lineCacheY[bestIndex];
            for (size_t j = 0; j < xs.size(); ++j) {
                int idx = ys[j] * imgSize + xs[j];
                error[idx] -= config.lineWeight;
            }

            // Update recent pins list
            for (int j = 0; j < 19; ++j) {
                lastPins[j] = lastPins[j + 1];
            }
            lastPins[19] = bestPin;
            currentPin = bestPin;
        }

        std::cout << "\n";
        return lineSequence;
    }

    void generateOutputImage(const std::vector<int>& lineSequence) {
        int outSize = config.outputSize;
        std::vector<unsigned char> image(outSize * outSize * 4, 255);

        // Set alpha channel
        for (int i = 3; i < outSize * outSize * 4; i += 4) {
            image[i] = 255;
        }

        double scale = static_cast<double>(outSize) / imgSize;

        // Draw circle border
        int centerOut = outSize / 2;
        int radiusOut = outSize / 2 - 1;
        drawCircle(image, centerOut, centerOut, radiusOut, outSize);

        // Draw pins
        for (const auto& coord : pinCoords) {
            int scaledX = static_cast<int>(coord.x * scale);
            int scaledY = static_cast<int>(coord.y * scale);
            drawFilledCircle(image, scaledX, scaledY, 2, outSize);
        }

        // Draw lines with alpha blending
        for (size_t i = 0; i < lineSequence.size() - 1; ++i) {
            const auto& from = pinCoords[lineSequence[i]];
            const auto& to = pinCoords[lineSequence[i + 1]];

            int x0 = static_cast<int>(from.x * scale);
            int y0 = static_cast<int>(from.y * scale);
            int x1 = static_cast<int>(to.x * scale);
            int y1 = static_cast<int>(to.y * scale);

            drawLineAlpha(image, x0, y0, x1, y1, config.outputWeight, outSize);
        }

        stbi_write_png(config.outputFile.c_str(), outSize, outSize, 4, image.data(), outSize * 4);
    }

    void drawLineAlpha(std::vector<unsigned char>& img, int x0, int y0, int x1, int y1, int alpha, int size) {
        int dx = std::abs(x1 - x0);
        int dy = std::abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int err = dx - dy;

        double alphaF = alpha / 255.0;
        double invAlpha = 1.0 - alphaF;

        while (true) {
            if (x0 >= 0 && x0 < size && y0 >= 0 && y0 < size) {
                int idx = (y0 * size + x0) * 4;
                img[idx] = static_cast<unsigned char>(img[idx] * invAlpha);
                img[idx + 1] = static_cast<unsigned char>(img[idx + 1] * invAlpha);
                img[idx + 2] = static_cast<unsigned char>(img[idx + 2] * invAlpha);
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

    void drawCircle(std::vector<unsigned char>& img, int cx, int cy, int r, int size) {
        for (double angle = 0; angle < 2 * M_PI; angle += 0.01) {
            int x = cx + static_cast<int>(r * std::cos(angle));
            int y = cy + static_cast<int>(r * std::sin(angle));
            if (x >= 0 && x < size && y >= 0 && y < size) {
                int idx = (y * size + x) * 4;
                img[idx] = img[idx + 1] = img[idx + 2] = 0;
            }
        }
    }

    void drawFilledCircle(std::vector<unsigned char>& img, int cx, int cy, int r, int size) {
        for (int x = -r; x <= r; ++x) {
            for (int y = -r; y <= r; ++y) {
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
};

int main(int argc, char* argv[]) {
    Config config;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-input" && i + 1 < argc) {
            config.inputFile = argv[++i];
        } else if (arg == "-output" && i + 1 < argc) {
            config.outputFile = argv[++i];
        } else if (arg == "-pins" && i + 1 < argc) {
            config.pins = std::stoi(argv[++i]);
        } else if (arg == "-lines" && i + 1 < argc) {
            config.maxLines = std::stoi(argv[++i]);
        } else if (arg == "-size" && i + 1 < argc) {
            config.targetSize = std::stoi(argv[++i]);
        } else if (arg == "-output-size" && i + 1 < argc) {
            config.outputSize = std::stoi(argv[++i]);
        } else if (arg == "-weight" && i + 1 < argc) {
            config.lineWeight = std::stoi(argv[++i]);
        } else if (arg == "-output-weight" && i + 1 < argc) {
            config.outputWeight = std::stoi(argv[++i]);
        } else if (arg == "-min-distance" && i + 1 < argc) {
            config.minDistance = std::stoi(argv[++i]);
        }
    }

    if (config.outputSize == 0) {
        config.outputSize = config.targetSize;
    }
    if (config.outputWeight == 0) {
        config.outputWeight = config.lineWeight;
    }

    if (config.inputFile.empty()) {
        std::cerr << "Please provide an input file using -input flag\n";
        return 1;
    }

    StringArtGenerator generator(config);
    generator.run();

    return 0;
}