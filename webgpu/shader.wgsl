// WebGPU String Art Compute Shader - Optimized Parallelization
// Evaluates multiple candidates in parallel with smaller reduction groups

// Storage buffers
@group(0) @binding(0) var<storage, read_write> errorBuffer: array<f32>;
@group(0) @binding(1) var<storage, read> lineCoordBuffer: array<u32>;
@group(0) @binding(2) var<storage, read> lineMetadata: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> lineSequence: array<u32>;
@group(0) @binding(4) var<storage, read_write> state: vec2<u32>; // currentPin, iteration

// Uniforms for configuration
struct Config {
    imgSize: u32,
    pins: u32,
    minDistance: u32,
    maxLines: u32,
    lineWeight: f32,
    iterationsPerDispatch: u32,  // Number of iterations to process in one dispatch
}
@group(0) @binding(5) var<uniform> config: Config;

// Workgroup shared memory
// Process 32 candidates in parallel, each using 32 threads for reduction
var<workgroup> partialSums: array<f32, 1024>;  // 32 candidates × 32 threads
var<workgroup> candidateErrors: array<f32, 270>;
var<workgroup> sharedErrors: array<f32, 512>;
var<workgroup> sharedIndices: array<u32, 512>;
var<workgroup> bestPin: u32;

@compute @workgroup_size(1024)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;

    // Process multiple iterations per dispatch
    for (var iter_offset = 0u; iter_offset < config.iterationsPerDispatch; iter_offset++) {
        // All threads read state uniformly
        let currentPin = state.x;
        let iteration = state.y;

        // All threads check the same condition uniformly
        let should_process = iteration < config.maxLines;

        // ==== PHASE 1: Evaluate Candidates ====
    // Process 32 candidates in parallel, each using 32 threads
    // tid 0-31: candidate 0, tid 32-63: candidate 1, etc.

    let numCandidates = config.pins - 2u * config.minDistance;
    let THREADS_PER_CANDIDATE = 32u;
    let CANDIDATES_PER_BATCH = 32u;  // 1024 / 32 = 32 candidates at once

    let num_batches = (numCandidates + CANDIDATES_PER_BATCH - 1u) / CANDIDATES_PER_BATCH;

    for (var batch = 0u; batch < num_batches; batch++) {
        let candidate_in_batch = tid / THREADS_PER_CANDIDATE;  // 0-31
        let thread_in_group = tid % THREADS_PER_CANDIDATE;     // 0-31
        let candidate_idx = batch * CANDIDATES_PER_BATCH + candidate_in_batch;

        var my_sum = 0.0;

        if (should_process && candidate_idx < numCandidates) {
            let test_pin = (currentPin + config.minDistance + candidate_idx) % config.pins;

            // Load line metadata
            let line_idx = test_pin * config.pins + currentPin;
            let offset = lineMetadata[line_idx].x;
            let length = lineMetadata[line_idx].y;

            // Each of the 32 threads in this group sums a subset of pixels
            for (var i = thread_in_group; i < length; i += THREADS_PER_CANDIDATE) {
                let coord_idx = offset + i;
                let x = lineCoordBuffer[coord_idx * 2u];
                let y = lineCoordBuffer[coord_idx * 2u + 1u];
                let pixel_idx = y * config.imgSize + x;
                my_sum += errorBuffer[pixel_idx];
            }
        }

        // Store partial sum
        partialSums[tid] = my_sum;
        workgroupBarrier();

        // Parallel reduction within each 32-thread group
        // Reduce 32 values to 1 for each candidate
        if (thread_in_group < 16u) {
            partialSums[tid] += partialSums[tid + 16u];
        }
        workgroupBarrier();
        if (thread_in_group < 8u) {
            partialSums[tid] += partialSums[tid + 8u];
        }
        workgroupBarrier();
        if (thread_in_group < 4u) {
            partialSums[tid] += partialSums[tid + 4u];
        }
        workgroupBarrier();
        if (thread_in_group < 2u) {
            partialSums[tid] += partialSums[tid + 2u];
        }
        workgroupBarrier();

        // Thread 0 of each group stores the result
        if (should_process && thread_in_group == 0u && candidate_idx < numCandidates) {
            let total_sum = partialSums[tid] + partialSums[tid + 1u];
            let test_pin = (currentPin + config.minDistance + candidate_idx) % config.pins;
            let line_idx = test_pin * config.pins + currentPin;
            let length = lineMetadata[line_idx].y;
            candidateErrors[candidate_idx] = total_sum / f32(length);
        }
        // No barrier needed here - next iteration starts with a barrier at line 82
    }

    // Initialize invalid candidates
    if (should_process && tid < 270u && tid >= numCandidates) {
        candidateErrors[tid] = -999999.0;
    }
    // No barrier needed - Phase 2 has one below before reading candidateErrors

        // ==== PHASE 2: Find Best Candidate ====

        // Load into shared memory for reduction
        if (tid < 270u) {
            if (should_process) {
                sharedErrors[tid] = candidateErrors[tid];
            } else {
                sharedErrors[tid] = -999999.0;
            }
            sharedIndices[tid] = tid;
        } else {
            sharedErrors[tid] = -999999.0;
            sharedIndices[tid] = 0u;
        }
        workgroupBarrier();

        // Parallel reduction to find maximum (now using 512 threads)
        for (var stride = 256u; stride > 0u; stride >>= 1u) {
            if (tid < stride && tid < 512u) {
                let other_idx = tid + stride;
                if (other_idx < 512u && sharedErrors[other_idx] > sharedErrors[tid]) {
                    sharedErrors[tid] = sharedErrors[other_idx];
                    sharedIndices[tid] = sharedIndices[other_idx];
                }
            }
            workgroupBarrier();
        }

        // Thread 0 computes the best pin and stores it
        if (tid == 0u && should_process) {
            let best_candidate_idx = sharedIndices[0];
            bestPin = (currentPin + config.minDistance + best_candidate_idx) % config.pins;
            lineSequence[iteration + 1u] = bestPin;
        }
        workgroupBarrier();

        // ==== PHASE 3: Update Error Buffer ====

        let selected_pin = bestPin;
        let line_idx = selected_pin * config.pins + currentPin;
        let offset = lineMetadata[line_idx].x;
        let length = lineMetadata[line_idx].y;

        // Parallel update: each thread handles subset of pixels (1024 threads now)
        if (should_process) {
            for (var i = tid; i < length; i += 1024u) {
                let coord_idx = offset + i;
                let x = lineCoordBuffer[coord_idx * 2u];
                let y = lineCoordBuffer[coord_idx * 2u + 1u];
                let pixel_idx = y * config.imgSize + x;
                errorBuffer[pixel_idx] -= config.lineWeight;
            }
        }

        // Thread 0 updates state
        workgroupBarrier();
        if (tid == 0u && should_process) {
            state.x = bestPin;  // Update current pin
            state.y = iteration + 1u;  // Increment iteration
        }
        workgroupBarrier();  // Ensure state is updated before next iteration
    }
}
