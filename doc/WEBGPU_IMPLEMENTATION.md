# WebGPU Implementation Notes

## Design Decisions and Pitfalls

### WebGPU-Specific Pitfalls

- **WGSL uniform control flow**: `workgroupBarrier()` must be reached by all threads — cannot be inside non-uniform `if`/`break`. Solution: all threads execute all iterations; work is skipped via `if` guards while barriers remain unconditional.
- **Storage buffer visibility**: Writes by one thread to storage buffers are not guaranteed visible to other threads even after `workgroupBarrier()` (which only syncs workgroup memory). State readback from GPU can be stale.
- **Line count detection**: Can't use 0 as empty marker (pin 0 is valid). Solution: fill buffer with sentinel `0xFFFFFFFF`, scan for first sentinel after processing.
- **Buffer reset**: Must fully reset error buffer, state, and line sequence between runs — stale data from previous runs leaks through otherwise.

### Line Count Detection

The shader writes pin indices into a fixed-size `lineSequenceBuffer`. When the algorithm stops early, remaining slots are unwritten:

- **Cannot rely on GPU state readback**: Storage buffer writes are not guaranteed visible to other threads even after `workgroupBarrier()` (which only syncs workgroup memory, not storage memory).
- **Cannot use zero as empty marker**: Pin 0 is a valid pin index.
- **Solution**: Fill buffer with sentinel `0xFFFFFFFF` before each run. JS scans forward to find the first sentinel for exact line count.

### WGSL Uniform Control Flow

`workgroupBarrier()` must be called from uniform control flow — all threads must reach the same barrier:

- Storage buffer reads and workgroup variables are treated as non-uniform by the WGSL validator.
- **Solution**: No `break` statements. All threads execute every loop iteration and hit every barrier. Work is skipped via `if (isActive == 1u)` guards while barriers remain unconditional.

### Buffer Reset Between Runs

Every `processImage` call must reset error buffer (re-upload), state buffer (`[0,0,0,0]`), and line sequence buffer (fill with `0xFFFFFFFF`). Without this, stale data from previous runs leaks through.

### Auto Line Weight Binary Search

Finds the **highest** weight where `lineCount >= targetMaxLines` (range 1-200, ~8 iterations). Higher weight = more per-line impact = fewer lines needed.
