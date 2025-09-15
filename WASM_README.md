# String Art Generator - WASM Version

This directory contains a WebAssembly (WASM) version of the string art generator that can run directly in web browsers.

## Prerequisites

To build the WASM version, you need:

1. **Emscripten SDK** installed and activated
   ```bash
   # Install Emscripten
   git clone https://github.com/emscripten-core/emsdk.git
   cd emsdk
   ./emsdk install latest
   ./emsdk activate latest
   source ./emsdk_env.sh
   ```

## Building

1. **Build the WASM module:**
   ```bash
   make build-wasm
   ```

   This will generate:
   - `stringart.wasm.js` - JavaScript wrapper for the WASM module
   - `stringart.wasm` - The actual WebAssembly binary

## Running the Demo

1. **Serve the files** using a local HTTP server (required due to CORS restrictions):
   ```bash
   # Using Python 3
   python3 -m http.server 8000

   # Or using Node.js
   npx serve .

   # Or using any other HTTP server
   ```

2. **Open the demo** in your browser:
   ```
   http://localhost:8000/stringart-demo.html
   ```

## Usage

The web interface allows you to:

1. **Upload an image** - Click "Select Image File" to choose your input image
2. **Adjust parameters** using the sliders:
   - **Pins**: Number of pins around the circle (100-500)
   - **Max Lines**: Maximum number of lines to draw (1000-10000)
   - **Processing Size**: Internal processing resolution (200-800px)
   - **Output Size**: Final output image size (400-2000px)
   - **Line Weight**: How much each line affects the algorithm (1-20)
   - **Output Weight**: Visual opacity of lines in output (1-30)
   - **Min Distance**: Minimum distance between consecutive pins (10-100)

3. **Generate string art** - Click the "Generate String Art" button
4. **Download result** - Use the download button under the generated image

## API Reference

The WASM module exports these functions:

### `initStringArt(pins, maxLines, targetSize, outputSize, lineWeight, outputWeight, minDistance)`
Initialize the string art generator with the given parameters.

### `processImage(imageDataPtr, width, height, channels)`
Process an image and calculate the string art lines. Returns the number of lines generated.

### `getOutputImage()`
Returns a pointer to the generated output image data (RGBA format).

### `getOutputSize()`
Returns the size of the output image.

### `getLineCount()`
Returns the number of lines in the generated string art.

### `cleanup()`
Free all allocated memory.

## Performance Notes

- **Processing time** varies based on parameters - more pins and lines take longer
- **Memory usage** scales with image size and number of pins
- **Recommended settings** for good balance of quality and speed:
  - Pins: 300-400
  - Max Lines: 3000-5000
  - Processing Size: 400-600px

## File Structure

- `stringart_wasm.c` - WASM-compatible C source code
- `stringart-demo.html` - Interactive web demo
- `Makefile` - Build configuration with WASM target
- `stb_image.h` / `stb_image_write.h` - Image loading/saving libraries

## Troubleshooting

**WASM module fails to load:**
- Ensure you're serving files over HTTP (not file://)
- Check browser console for detailed error messages
- Verify Emscripten compiled successfully

**Out of memory errors:**
- Reduce image size or processing parameters
- The algorithm is memory-intensive for large images

**Slow performance:**
- Reduce number of pins and max lines
- Use smaller processing size
- Close other browser tabs to free memory