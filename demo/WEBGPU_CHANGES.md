# WebGPU Migration - Demo Changes

This document summarizes the migration from WASM to WebGPU for the string art demo.

## Overview

The demo now uses **100% WebGPU** for string art generation, replacing the previous WASM implementation. This provides GPU-accelerated processing with the same algorithm.

## Key Changes

### 1. Configuration Change Detection

The processor now intelligently detects which parameters changed and responds accordingly:

#### Parameters that trigger full reinitialization:
- **imgSize** (processing resolution) - requires new error buffer size
- **pins** (number of pins) - requires line cache regeneration
- **minDistance** (minimum pin distance) - requires line cache regeneration
- **maxLines** (maximum lines) - requires new buffer sizes

When any of these change, the processor will:
1. Regenerate the line cache (CPU preprocessing)
2. Recreate all GPU buffers with new sizes
3. Recompile bind groups

#### Parameters that only require buffer updates:
- **lineWeight** - only updates config buffer on GPU
- **iterationsPerDispatch** - only updates config buffer on GPU

### 2. Concurrency Control

Added mutex/lock pattern to prevent concurrent GPU operations:

```javascript
// Only one processImage call can execute at a time
// Additional calls wait in queue
while (processingLock) {
    await processingLock;
}
```

This fixes the error: `"Buffer already has an outstanding map pending"`

### 3. Line Cache Optimization

Line cache is only regenerated when necessary:
- Cached when `pins`, `imgSize`, or `minDistance` haven't changed
- Logged to console when regeneration occurs
- Significant performance improvement for repeated generations with same settings

## Performance Characteristics

### Cache Hit (same pins/imgSize/minDistance):
- Line cache reused ✓
- GPU buffers reused ✓
- Only uploads: error buffer, state buffer
- **Fast**: ~50-100ms overhead

### Cache Miss (different pins/imgSize/minDistance):
- Line cache regenerated (CPU) - ~2-5 seconds for 300 pins
- All GPU buffers recreated
- **Slow first run**: ~3-7 seconds total
- Subsequent runs with same params: fast (cache hit)

### Auto Line Weight:
- Runs 8-10 iterations (binary search)
- Each iteration: full processing
- Total time: ~10-30 seconds depending on settings
- Shows intermediate results during search

## Browser Compatibility

- **Chrome/Edge 113+**: Full support
- **Safari 18+** (macOS Sonoma+): Full support
- **Firefox**: Not yet supported (WebGPU in development)

## Technical Details

### GPU Resource Management

**Buffers created per context:**
- `errorBuffer`: imgSize² × 4 bytes (f32 per pixel)
- `lineCoordBuffer`: ~41MB for 300 pins (all line coordinates)
- `lineMetadataBuffer`: pins² × 8 bytes (offset/length pairs)
- `lineSequenceBuffer`: (maxLines + 1) × 4 bytes
- `stateBuffer`: 8 bytes (current pin + iteration)
- `configBuffer`: 32 bytes (algorithm parameters)
- `readbackBuffer`: (maxLines + 1) × 4 bytes

**Total GPU memory**: ~42-50MB for typical settings (200px, 300 pins, 4000 lines)

### Processing Pipeline

1. **CPU Preprocessing** (once per config):
   - Generate pin coordinates
   - Precalculate all potential line coordinates
   - Store in line cache

2. **Per-Frame Processing**:
   - Convert image to luminosity (CPU)
   - Create error buffer: 255 - luminosity (CPU)
   - Upload error buffer to GPU
   - Dispatch batched iterations (GPU)
   - Readback line sequence (GPU→CPU)
   - Render to canvas (CPU)

### Compute Shader Details

- **Workgroup size**: 1024 threads
- **Iterations per dispatch**: 100 (configurable)
- **Total dispatches**: ceil(maxLines / iterationsPerDispatch)
- **Parallel candidate evaluation**: 270 candidates per iteration
- **Early termination**: Stops when no valid lines remain

## Debugging

Enable detailed logging by checking browser console:
- Line cache generation messages
- Reinitialization triggers
- Processing timings
- GPU memory usage (visible in Chrome DevTools → Performance Monitor)

## Known Limitations

1. **GPU timeout**: Very large settings (>10000 lines, >500 pins) may timeout on some GPUs
2. **Memory**: Each context uses ~50MB GPU memory
3. **Single-threaded**: Mutex ensures only one generation at a time (prevents errors but serializes requests)

## Future Optimizations

1. **Reuse buffers across reinitializations** when only size increases
2. **Async shader compilation** for faster startup
3. **Multiple readback buffers** to allow pipelined processing
4. **Shared line cache** across multiple contexts with same settings
