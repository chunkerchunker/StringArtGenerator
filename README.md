# String Art Generator

This project generates string art images from input photographs by simulating the process of wrapping thread around pins on a circular frame.

Originally based on the work of reddit user /u/kmmeerts from his post: https://www.reddit.com/r/DIY/comments/au0ilz/made_a_string_art_portrait_out_of_a_continuous_2/

## Implementations

### Go Version (Original)
- **File**: `main.go`
- **Build**: `make build`
- **Run**: `./stringart [options]`

### C Version (Basic Optimized)
- **File**: `stringart.c`
- **Build**: `make build-c`
- **Run**: `./stringart-c [options]`
- **Optimizations**:
  - Compiled with `-O3 -march=native` for performance
  - Uses optimized data structures and memory layout
  - Simple nearest-neighbor image resizing to avoid external dependencies
  - Only uses standard C library plus STB image libraries

### C Version (Highly Optimized)
- **File**: `stringart_fast.c`
- **Build**: `make build-fast`
- **Run**: `./stringart-fast [options]`
- **Optimizations**:
  - Pre-computed valid pin lookup tables to avoid repeated calculations
  - Unrolled loops for critical hot paths
  - Cache-friendly memory access patterns
  - Fixed-point arithmetic where possible
  - Aggressive function inlining and fast math optimizations

## Usage

```bash
# Build all versions
make all

# Build specific version
make build        # Go version
make build-c      # C basic version
make build-fast   # C highly optimized version

# Run with default settings
./stringart-fast -input image.jpg

# Test different versions (will open result automatically)
make test-fast    # Recommended for best performance
make test-c       # Basic C version
make test         # Original Go version

# Full parameter example
./stringart-fast -input image.jpg -output result.png -pins 400 -lines 20000 -size 500 -output-size 2000 -weight 4 -output-weight 15 -min-distance 30
```

## Parameters

- `-input`: Input image file (JPEG or PNG) **[Required]**
- `-output`: Output image file (default: "output.png")
- `-pins`: Number of pins around the circle (default: 300)
- `-lines`: Maximum number of lines to draw (default: 4000)
- `-size`: Processing size for input image in pixels (default: 500)
- `-output-size`: Output image size in pixels (default: same as processing size)
- `-weight`: Line weight for darkness calculation (default: 8, higher = darker)
- `-output-weight`: Visual line opacity for output image (default: same as weight, 0-255)
- `-min-distance`: Minimum distance between connected pins (default: 30)

## Algorithm

1. **Image Processing**: Load and convert input image to grayscale, crop to square, resize to target dimensions
2. **Pin Placement**: Calculate pin coordinates evenly distributed around a circle
3. **Line Precalculation**: Pre-compute all possible lines between pins respecting minimum distance constraints
4. **Greedy Line Selection**: Iteratively select the line that best reduces the error between current and target image
5. **Output Generation**: Render the selected lines on a clean canvas with specified visual parameters

## Dependencies

### C Versions
- STB libraries (included as headers):
  - `stb_image.h` - Image loading
  - `stb_image_write.h` - PNG output
- Standard C library with math functions (`-lm`)

### Go Version
- Standard Go libraries
- `golang.org/x/image/draw` - Image processing
- `github.com/antha-lang/antha/antha/anthalib/num` - Numerical operations

## Performance Comparison

The optimized C versions provide significant performance improvements:

### Expected Speedups vs Go Version:
- **`stringart-c` (Basic)**: 1.5-2x faster
- **`stringart-fast` (Highly Optimized)**: 2-4x faster

### Key Optimizations Applied:
- **Compiler**: `-O3 -march=native` for CPU-specific optimizations
- **Memory**: Cache-friendly data layouts and access patterns
- **Algorithm**: Pre-computed lookup tables and unrolled loops
- **Math**: Fast floating-point operations (`-ffast-math`)
- **Function Inlining**: Aggressive inlining for hot code paths

### Recommended Version:
Use **`stringart-fast`** for the best performance. Single-threaded cache optimization outperforms multithreading for this algorithm due to its sequential nature and memory access patterns.

### Performance Testing:
```bash
# Compare versions
time ./stringart -input image.jpg -pins 400 -lines 10000        # Go baseline
time ./stringart-c -input image.jpg -pins 400 -lines 10000      # Basic C optimization
time ./stringart-fast -input image.jpg -pins 400 -lines 10000   # Best performance
```

## Build Requirements

- C compiler (clang or gcc)
- Make
- Internet connection (for downloading STB headers)

## Testing

```bash
# Test with sample parameters
make test-c
```

This will process a test image with high-quality settings and open the result.

## Web Version

For a web-based interface, navigate to: https://halfmonty.github.io/StringArtGenerator/

![](test2.gif)
