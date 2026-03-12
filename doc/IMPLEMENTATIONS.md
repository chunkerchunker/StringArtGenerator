# History of implementation alternatives explored

## Core Algorithm (Shared Across All Implementations)

All versions implement the same greedy algorithm:

1. **Preprocess**: Load image, crop to square, resize, convert to grayscale (ITU-R BT.709 luminosity)
2. **Pin placement**: Distribute N pins evenly around a circle
3. **Line precalculation**: Pre-compute pixel coordinates for all valid pin-to-pin lines (respecting minimum distance)
4. **Greedy iteration** (repeat up to MAX_LINES times):
   - From the current pin, evaluate all candidate pins (beyond min distance, not in recent-20 history)
   - Select the pin whose line maximizes error reduction (sum of error values along the line)
   - Subtract line weight from error buffer along that line
   - Stop early if no line provides positive error reduction
5. **Render**: Draw selected lines on a white canvas with specified opacity

## Implementations

### 1. Go (Original Reference)

- **Files**: `main.go`
- **Runtime**: CLI (`./stringart -input image.jpg`)
- **Details**: Straightforward single-threaded implementation. Uses `golang.org/x/image/draw` for bilinear resizing. Served as the baseline for correctness and performance comparison.
- **Speed**: 1x (baseline)

### 2. C (Basic Optimized)

- **Files**: `stringart.c`
- **Runtime**: CLI (`./stringart-c`)
- **Details**: Direct port from Go with C-level optimizations. Minimal dependencies (STB image headers only). Simple nearest-neighbor resizing to avoid library dependencies. Added `-auto-weight` (binary search for optimal line weight) and `-output-pins` features.
- **Speed**: ~1.5-2x vs Go
- **Build flags**: `-O3 -march=native`

### 3. C (Highly Optimized with SIMD)

- **Files**: `stringart_core.h`, `stringart_core.c`, `stringart_fast.c`
- **Runtime**: CLI (`./stringart-fast`)
- **Details**: Same algorithm with hardware-accelerated hot paths:
  - **SIMD error summation**: AVX (8 floats), SSE (4 floats), NEON (ARM/Apple Silicon), WASM SIMD128, with scalar fallback
  - Pre-computed valid-pin lookup tables to eliminate branching
  - Fixed-point arithmetic for image resizing
  - Cache-friendly memory layout
  - Aggressive inlining and loop unrolling
- **Speed**: ~2-4x vs Go
- **Build flags**: `-O3 -march=native -funroll-loops -ffast-math -finline-functions`
- **Key learning**: Single-threaded cache optimization outperformed multithreading for this algorithm due to its sequential nature and memory access patterns.

### 4. Python (NumPy + Numba JIT)

- **Files**: `string-art-python/string_art.py`, `string-art-python/main.py`
- **Runtime**: CLI (`uv run string_art.py --input image.jpg`)
- **Details**: NumPy for data handling, Numba `@jit(nopython=True)` for hot loops (line error evaluation, error buffer updates, main greedy loop). Uses Bresenham's algorithm for line rasterization and **aggdraw** for anti-aliased output rendering — the only implementation producing smooth, anti-aliased results.
- **Speed**: ~0.5-1x vs Go (after JIT warmup)
- **Tradeoff**: Slower execution but highest output quality (anti-aliased lines) and fastest development iteration.

### 5. WebAssembly (C compiled to WASM)

- **Files**: `stringart_wasm.c` (wraps `stringart_core.c`)
- **Runtime**: Browser (loaded via Emscripten JS wrapper)
- **Build**: `emcc -O3 -msimd128`
- **Details**: The optimized C/SIMD implementation compiled to WASM for in-browser execution. Exports `initStringArt()`, `processImage()`, `getLineSequence()`, `cleanup()`. Intelligent cache reuse — only reinitializes when pins/size/minDistance change.
- **Speed**: Slightly slower than native C, but runs in the browser with WASM SIMD128 support
- **Memory**: ~32MB initial, up to 1GB max
- **Tradeoff**: Brought near-native C performance to the browser, but still CPU-bound. Required Emscripten toolchain for builds.

### 6. WebGPU (GPU Compute Shader)

- **Files**: `webgpu-processor.js`, `shader.wgsl`, `lineCache.js`, `preprocessing.js`, `renderer.js`
- **Runtime**: Browser (Chrome/Edge 113+, Safari 18+; Firefox not yet supported)
- **Details**: The entire greedy algorithm runs on the GPU via a WGSL compute shader:
  - 1024-thread workgroups with parallel reduction
  - Evaluates ~270 candidate pins simultaneously
  - Batched iterations (100 per dispatch) to amortize GPU dispatch overhead
  - ~42-50MB GPU memory for typical settings (200px, 300 pins, 4000 lines)
- **Speed**: ~50-100ms with warm cache; 3-7s on cache miss (line cache regeneration)
- **Browser compatibility**: Chrome/Edge 113+, Safari 18+ (macOS Sonoma+). No Firefox support yet.

### 7. 3b1b Explainer (Manim Animations)

- **Files**: `doc/3b1b/string_art_explainer.py`, `doc/3b1b/explainer2.py`
- **Runtime**: Video generation via Manim
- **Purpose**: Educational animations showing the algorithm visually — not a processing implementation.

## Interactive Browser Demo

- **Files**: `index.html`, `stringart.js`, `stringart.css`
- **Features**: Image upload or webcam input, adjustable parameters (pins, lines, size, weight, opacity, min distance), auto line weight (binary search), zoom/pan, PNG download, real-time webcam mode with FPS counter.
- **Backend**: Originally used WASM, migrated to pure WebGPU.

## Performance Progression

```
Go (baseline) → C basic (1.5-2x) → C/SIMD (2-4x) → WASM in browser (~C speed) → WebGPU (~same as WASM, no build toolchain)
```

The C/SIMD version compiled to WASM was the fastest browser option initially. The pure WebGPU version ultimately performed comparably while eliminating the Emscripten build dependency entirely.

## Key Tradeoffs and Learnings

### CPU vs GPU

- The greedy algorithm is inherently sequential (each iteration depends on the previous), which limits GPU parallelism to *within* each iteration (evaluating candidate pins in parallel).
- GPU excels at the candidate evaluation phase but has overhead from dispatch latency and CPU-GPU data transfer.
- Iterations are batched (100 per dispatch, configurable) to reduce the number of GPU dispatches.

### SIMD

- AVX (8-wide) provided meaningful speedup on x86_64; NEON comparable to SSE on Apple Silicon.
- The main bottleneck is memory access (reading error buffer along line pixels), so SIMD helps but doesn't transform performance.

### Memory vs Compute

- Pre-computing all line coordinates (~41MB for 300 pins) is essential — without it, per-iteration coordinate calculation dominates runtime.
- This cache is the primary cost of "cold start" in both WASM and WebGPU versions.

### WASM vs WebGPU

- WASM required Emscripten SDK and a C build toolchain. WebGPU is pure JS + WGSL.
- WASM had more predictable performance. WebGPU performance varies with GPU hardware and driver.
- WebGPU has browser compatibility gaps (no Firefox). WASM works everywhere.
- For this workload, the two approaches ended up roughly equivalent in speed.

### Output Quality

- The Python/aggdraw implementation was the only one producing anti-aliased output. All other implementations use hard-edged line drawing.
- Output line opacity (separate from algorithm weight) significantly affects visual quality and is worth tuning independently.

## Final Architecture

The final version is a pure WebGPU browser application in ``, with no WASM dependency. It supports both static image and webcam input with real-time processing.
