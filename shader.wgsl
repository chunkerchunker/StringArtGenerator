// WebGPU String Art Compute Shader - Optimized Parallelization
// Evaluates multiple candidates in parallel with smaller reduction groups
//
// NOTE on control flow: WGSL requires workgroupBarrier() to be called from
// uniform control flow. Storage and workgroup variables are considered
// non-uniform. Therefore, ALL threads must execute every barrier on every
// iteration — we use `if` guards to skip work but never `break`.

// Storage buffers
@group(0) @binding(0) var<storage, read_write> errorBuffer: array<f32>;
@group(0) @binding(1) var<storage, read> lineCoordBuffer: array<u32>;
@group(0) @binding(2) var<storage, read> lineMetadata: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> lineSequence: array<u32>;
@group(0) @binding(4) var<storage, read_write> state: vec4<u32>; // currentPin, iteration, stopped, (padding)

// Uniforms for configuration
struct Config {
    imgSize: u32,
    pins: u32,
    minDistance: u32,
    maxLines: u32,
    lineWeight: f32,
    iterationsPerDispatch: u32,
}
@group(0) @binding(5) var<uniform> config: Config;

// Workgroup shared memory
var<workgroup> partialSums: array<f32, 1024>;
var<workgroup> candidateErrors: array<f32, 270>;
var<workgroup> sharedErrors: array<f32, 512>;
var<workgroup> sharedIndices: array<u32, 512>;
var<workgroup> bestPin: u32;
var<workgroup> wgActive: u32;

@compute @workgroup_size(1024)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;

    // Load stop flag from storage into workgroup memory at dispatch start
    if (tid == 0u) {
        wgActive = select(0u, 1u, state.z == 0u && state.y < config.maxLines);
    }
    workgroupBarrier();

    let numCandidates = config.pins - 2u * config.minDistance;
    let THREADS_PER_CANDIDATE = 32u;
    let CANDIDATES_PER_BATCH = 32u;
    let num_batches = (numCandidates + CANDIDATES_PER_BATCH - 1u) / CANDIDATES_PER_BATCH;

    // Process multiple iterations per dispatch
    // ALL threads execute every iteration and every barrier — no early exit
    for (var iter_offset = 0u; iter_offset < config.iterationsPerDispatch; iter_offset++) {

        // Thread 0 reads state and broadcasts via workgroup memory
        if (tid == 0u) {
            // Check if we should still be processing
            if (state.z != 0u || state.y >= config.maxLines) {
                wgActive = 0u;
            }
        }
        workgroupBarrier();

        // Read current state (all threads get same workgroup values)
        let currentPin = state.x;
        let iteration = state.y;
        let isActive = wgActive;

        // ==== PHASE 1: Evaluate Candidates ====

        for (var batch = 0u; batch < num_batches; batch++) {
            let candidate_in_batch = tid / THREADS_PER_CANDIDATE;
            let thread_in_group = tid % THREADS_PER_CANDIDATE;
            let candidate_idx = batch * CANDIDATES_PER_BATCH + candidate_in_batch;

            var my_sum = 0.0;

            if (isActive == 1u && candidate_idx < numCandidates) {
                let test_pin = (currentPin + config.minDistance + candidate_idx) % config.pins;
                let line_idx = test_pin * config.pins + currentPin;
                let offset = lineMetadata[line_idx].x;
                let length = lineMetadata[line_idx].y;

                for (var i = thread_in_group; i < length; i += THREADS_PER_CANDIDATE) {
                    let coord_idx = offset + i;
                    let x = lineCoordBuffer[coord_idx * 2u];
                    let y = lineCoordBuffer[coord_idx * 2u + 1u];
                    let pixel_idx = y * config.imgSize + x;
                    my_sum += errorBuffer[pixel_idx];
                }
            }

            partialSums[tid] = my_sum;
            workgroupBarrier();

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

            if (isActive == 1u && thread_in_group == 0u && candidate_idx < numCandidates) {
                let total_sum = partialSums[tid] + partialSums[tid + 1u];
                let test_pin = (currentPin + config.minDistance + candidate_idx) % config.pins;
                let line_idx = test_pin * config.pins + currentPin;
                let length = lineMetadata[line_idx].y;
                candidateErrors[candidate_idx] = total_sum / f32(length);
            }
        }

        // Initialize invalid candidates
        if (isActive == 1u && tid < 270u && tid >= numCandidates) {
            candidateErrors[tid] = -999999.0;
        }

        // ==== PHASE 2: Find Best Candidate ====

        if (tid < 270u) {
            if (isActive == 1u) {
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

        // Thread 0 decides: accept line or stop
        if (tid == 0u && isActive == 1u) {
            let best_candidate_idx = sharedIndices[0];
            let best_error = sharedErrors[0];

            if (best_error > 0.0) {
                bestPin = (currentPin + config.minDistance + best_candidate_idx) % config.pins;
                lineSequence[iteration + 1u] = bestPin;
            } else {
                state.z = 1u;
                wgActive = 0u;
            }
        }
        workgroupBarrier();

        // Re-read active flag after thread 0 may have changed it
        let stillActive = wgActive;

        // ==== PHASE 3: Update Error Buffer ====

        if (stillActive == 1u) {
            let selected_pin = bestPin;
            let line_idx = selected_pin * config.pins + currentPin;
            let offset = lineMetadata[line_idx].x;
            let length = lineMetadata[line_idx].y;

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
        if (tid == 0u && stillActive == 1u) {
            state.x = bestPin;
            state.y = iteration + 1u;
        }
        workgroupBarrier();
    }
}
