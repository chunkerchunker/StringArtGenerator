# WebGPU String Art Generator

Real-time webcam string art generation using fully GPU-resident compute shaders.

## Overview

This implementation ports the string art algorithm from `main.go` to run entirely on the GPU using WebGPU. It processes webcam frames in real-time, applying the same greedy line selection algorithm to generate artistic string art representations.

### Architecture

- **One iteration per dispatch**: Each of the 4000 iterations runs as a separate GPU dispatch
- **GPU queue ordering**: Automatic memory visibility between dispatches without CPU synchronization
- **Parallel candidate evaluation**: 270 candidates evaluated simultaneously per iteration
- **Persistent line cache**: ~41 MB of precomputed line coordinates (generated once)

## Algorithm Fidelity

This implementation uses the **exact same algorithm** as the Go version:

1. Webcam frame capture and resize to square (200×200px default)
2. Convert to grayscale luminosity (ITU-R BT.709 weights)
3. Initialize error buffer (255.0 - luminosity)
4. Greedy line selection (4000 iterations):
   - Evaluate 270 candidate lines in parallel
   - Select line that maximizes error reduction
   - Update error buffer by subtracting line weight
5. Render selected lines to canvas

## Requirements

- **Browser**: Chrome/Edge 113+ or Safari 18+ (WebGPU support)
- **GPU**: Any modern GPU with WebGPU support
- **Webcam**: For real-time input

## Usage

### Local Testing

Since this uses ES6 modules, you need to serve via HTTP (not `file://`):

```bash
# Option 1: Python
cd webgpu
python3 -m http.server 8000

# Option 2: Node.js
npx serve webgpu

# Option 3: Any other local server
```

Then open `http://localhost:8000` in a WebGPU-enabled browser.

### Controls

1. Click **Start Webcam** to request camera access
2. Wait for WebGPU initialization (~2-5 seconds for line cache generation)
3. Processing begins automatically
4. Click **Pause/Resume** to control frame processing
5. FPS counter shows real-time performance

## Configuration

Edit `CONFIG` in `main.js`:

```javascript
const CONFIG = {
    IMG_SIZE: 200,        // Processing resolution (200×200px)
    OUTPUT_SIZE: 500,     // Canvas output size (500×500px)
    PINS: 300,            // Number of pins around circle
    MIN_DISTANCE: 30,     // Minimum distance between connected pins
    MAX_LINES: 4000,      // Number of lines to draw
    LINE_WEIGHT: 8.0      // Darkness of each line
};
```

## Performance

### Expected Performance (M2 GPU)

Performance varies based on configuration. With default settings (200×200px, 300 pins, 2000-4000 lines):

- **Typical**: 1-3 fps
- **Bottleneck**: 4000 separate GPU dispatches per frame

### Performance Breakdown

- **CPU preprocessing**: ~3-5ms (capture + luminosity conversion)
- **GPU compute**: Varies with MAX_LINES (each iteration is a separate dispatch)
- **GPU→CPU readback**: ~0.5ms (16 KB line sequence)
- **Canvas rendering**: ~2-5ms

### Performance Tips

1. **Reduce MAX_LINES**: Try 2000 instead of 4000 for 2x speedup
2. **Increase LINE_WEIGHT**: Higher values (12-16) can compensate for fewer lines
3. **Reduce IMG_SIZE**: Smaller processing resolution = faster
4. **Reduce PINS**: Fewer pins = less computation per iteration

### Comparison to C/WASM

The C/WASM version (achieving ~10 fps with SIMD) is faster due to:
- CPU cache-friendly memory access patterns
- Hand-optimized SIMD instructions
- No GPU dispatch overhead

The WebGPU version excels at:
- Parallel candidate evaluation (270 simultaneous evaluations vs sequential)
- GPU memory bandwidth for large data structures
- Offloading computation from CPU

## Files

```
webgpu/
├── index.html              Main HTML page with webcam + canvas
├── style.css               Styling
├── main.js                 WebGPU initialization and frame processing loop
├── shader-single-iter.wgsl Single-iteration compute shader
├── lineCache.js            Pre-compute line coordinates (CPU)
├── preprocessing.js        Webcam capture and luminosity conversion (CPU)
├── renderer.js             Canvas 2D rendering (CPU)
└── README.md               This file
```

## Implementation Details

### GPU Buffers

| Buffer | Size | Usage |
|--------|------|-------|
| errorBuffer | 160 KB | f32 per pixel, read/write each iteration |
| lineCoordBuffer | ~41 MB | All line coordinates (persistent, read-only) |
| lineMetadataBuffer | 540 KB | Offset/length per pin pair (persistent, read-only) |
| lineSequenceBuffer | 16 KB | Output pin sequence (write-only) |
| stateBuffer | 8 bytes | Current pin + iteration counter (read/write) |
| configBuffer | 32 bytes | Algorithm parameters (read-only) |

### Compute Shader Phases (per iteration)

1. **Phase 1**: Evaluate candidates (270 threads active)
   - Each thread evaluates one candidate line
   - Computes average error along line
   - Stores in shared memory

2. **Phase 2**: Find best candidate (256 threads, parallel reduction)
   - Log₂(256) = 8 barrier synchronizations
   - Finds maximum error and corresponding pin

3. **Phase 3**: Update error buffer (256 threads active)
   - Subtract LINE_WEIGHT from pixels along selected line
   - Advance current pin

### Browser Compatibility

**Supported:**
- Chrome/Edge 113+ (flag: `chrome://flags/#enable-unsafe-webgpu`)
- Safari 18+ (macOS Sonoma+)

**Not supported:**
- Firefox (WebGPU in development)
- Mobile browsers (limited WebGPU support)

## Troubleshooting

### "WebGPU not supported"
- Update browser to latest version
- Enable WebGPU flag in Chrome: `chrome://flags/#enable-unsafe-webgpu`
- Try Safari 18+ on macOS

### GPU timeout / no output
- Reduce `MAX_LINES` (try 2000 or 1000)
- GPU may kill long-running shaders (>2 seconds)
- Check browser console for errors

### Low FPS
- Expected! This is a compute-heavy algorithm
- Reduce `IMG_SIZE` (try 150 or 100)
- Reduce `PINS` or `MAX_LINES`

### Module loading errors
- Must serve via HTTP (not `file://`)
- Check console for CORS errors

## Known Issues & Limitations

1. **Performance**: Slower than C/WASM due to 4000 separate GPU dispatches per frame
2. **GPU timeout**: Very long shader runs (>2 seconds) may timeout on some systems
3. **Floating-point precision**: Minor differences from CPU version due to GPU precision

## Future Optimizations

1. **Batch iterations**: Process multiple iterations per dispatch to reduce overhead
2. **GPU preprocessing**: Move luminosity conversion to GPU (texture → compute shader)
3. **Reduced precision**: Use f16 instead of f32 for error buffer
4. **Indirect dispatch**: Use GPU-driven iteration loop to eliminate CPU dispatch overhead

## License

Same as parent project.
