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
- Binary search over weight values 1–200 (up to 10 iterations)
- Finds the highest weight where the algorithm still fills the line budget (`lineCount >= maxLines`)
- Each iteration: full GPU processing
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
- `stateBuffer`: 16 bytes (current pin, iteration, stopped flag, padding)
- `configBuffer`: 32 bytes (algorithm parameters)
- `readbackBuffer`: (maxLines + 1) × 4 bytes
- `stateReadbackBuffer`: 16 bytes (for reading state back from GPU)

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
- **Early termination**: Shader sets a persistent stop flag (`state.z`) when no valid lines remain; skips work on subsequent iterations without using `break` (WGSL uniform control flow requirement). Actual line count is determined JS-side by scanning the line sequence buffer for sentinel values (`0xFFFFFFFF`)

## Design Decisions and Pitfalls

### Line Count Detection

The GPU shader writes pin indices into a fixed-size `lineSequenceBuffer`. When the algorithm stops early (error depleted), remaining slots are unwritten. Determining the actual line count is non-trivial:

- **Cannot rely on GPU state readback**: The shader tracks iteration count in `state.y`, but WGSL storage buffer writes by one thread are not guaranteed to be visible to other threads even after `workgroupBarrier()` (which only synchronizes workgroup memory, not storage memory). This means `state.y` may read back as stale/incorrect after processing.
- **Cannot use zero as empty marker**: Pin 0 is a valid pin index. If the buffer is zeroed and the algorithm legitimately draws a line to pin 0, it's indistinguishable from an empty slot.
- **Solution**: The buffer is filled with sentinel values (`0xFFFFFFFF`) before each run. After processing, JS scans forward from index 1 to find the first sentinel, giving the exact line count. This is reliable regardless of GPU memory visibility behavior.

### WGSL Uniform Control Flow

WGSL requires that `workgroupBarrier()` is only called from uniform control flow — all threads in a workgroup must reach the same barrier. The compiler enforces this statically:

- **Storage buffer reads are non-uniform**: `state.y`, `state.z` etc. are `read_write storage`, so any `if`/`break` depending on them is considered non-uniform.
- **Workgroup variables are also non-uniform**: Even though all threads read the same workgroup variable after a barrier, the WGSL validator treats them as potentially non-uniform.
- **Solution**: The shader uses NO `break` statements. All threads unconditionally execute every loop iteration and hit every barrier. Work is skipped via `if (isActive == 1u)` guards on the actual computation, while barriers remain outside any non-uniform branches. A workgroup variable `wgActive` is set by thread 0 and read by all threads (into local `let` bindings) to coordinate skipping.

### Buffer Reset Between Runs

Every call to `processImage` must reset:
- **Error buffer**: Re-uploaded from the source image (fresh luminosity data)
- **State buffer**: Reset to `[0, 0, 0, 0]` (pin 0, iteration 0, not stopped)
- **Line sequence buffer**: Filled with sentinel `0xFFFFFFFF`

Without resetting the line sequence buffer, stale line data from previous runs leaks through. This was especially visible when changing `lineWeight` between runs — old lines from a low-weight run would persist in the buffer and appear in a subsequent high-weight run's output.

### Auto Line Weight Binary Search

The binary search finds the optimal `lineWeight` for a given `maxLines` target. Key design points:

- **Search direction**: Find the **highest** weight where `lineCount >= targetMaxLines`. Higher weight = each line has more impact = fewer lines needed. The search maximizes per-line impact while ensuring the line budget is fully used.
- **Why not lowest weight**: Lower weight always produces more lines (up to maxLines cap), so searching for the lowest weight where `lineCount <= target` would trivially return weight=1. The useful question is: how high can weight go before lines start dropping off?
- **Convergence**: Range 1–200, up to 10 iterations of binary search. `log2(200) ≈ 7.6`, so 8 iterations suffice for exact convergence. `bestWeight` is only updated when `lineCount >= targetMaxLines`.
- **Final generation**: After the search, a final `processImage` call with `bestWeight` produces the displayed result.

## Debugging

Enable detailed logging by checking browser console:
- Line cache generation messages
- Reinitialization triggers
- Processing timings
- Binary search progress: each iteration logs `Weight N: M lines (target: T)` and `bounds: low=L, high=H, bestWeight=B`
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
