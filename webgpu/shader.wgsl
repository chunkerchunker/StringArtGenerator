// WebGPU String Art Compute Shader - Single Iteration Version
// Processes ONE iteration per dispatch to ensure proper error buffer updates

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
}
@group(0) @binding(5) var<uniform> config: Config;

// Workgroup shared memory
var<workgroup> candidateErrors: array<f32, 270>;
var<workgroup> sharedErrors: array<f32, 256>;
var<workgroup> sharedIndices: array<u32, 256>;
var<workgroup> bestPin: u32;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let currentPin = state.x;
    let iteration = state.y;

    // ==== PHASE 1: Evaluate Candidates ====
    let numCandidates = config.pins - 2u * config.minDistance;

    if (tid < numCandidates) {
        let candidate_idx = tid;
        let test_pin = (currentPin + config.minDistance + candidate_idx) % config.pins;

        // Load line metadata
        let line_idx = test_pin * config.pins + currentPin;
        let offset = lineMetadata[line_idx].x;
        let length = lineMetadata[line_idx].y;

        // Sum error along line
        var error_sum = 0.0;
        for (var i = 0u; i < length; i++) {
            let coord_idx = offset + i;
            let x = lineCoordBuffer[coord_idx * 2u];
            let y = lineCoordBuffer[coord_idx * 2u + 1u];
            let pixel_idx = y * config.imgSize + x;
            error_sum += errorBuffer[pixel_idx];
        }

        // Store average error
        candidateErrors[candidate_idx] = error_sum / f32(length);
    } else if (tid < 270u) {
        candidateErrors[tid] = -999999.0;
    }

    workgroupBarrier();

    // ==== PHASE 2: Find Best Candidate ====

    // Load into shared memory for reduction
    if (tid < 270u) {
        sharedErrors[tid] = candidateErrors[tid];
        sharedIndices[tid] = tid;
    } else {
        sharedErrors[tid] = -999999.0;
        sharedIndices[tid] = 0u;
    }
    workgroupBarrier();

    // Parallel reduction to find maximum
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            let other_idx = tid + stride;
            if (sharedErrors[other_idx] > sharedErrors[tid]) {
                sharedErrors[tid] = sharedErrors[other_idx];
                sharedIndices[tid] = sharedIndices[other_idx];
            }
        }
        workgroupBarrier();
    }

    // Thread 0 updates shared state
    if (tid == 0u) {
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

    // Parallel update: each thread handles subset of pixels
    for (var i = tid; i < length; i += 256u) {
        let coord_idx = offset + i;
        let x = lineCoordBuffer[coord_idx * 2u];
        let y = lineCoordBuffer[coord_idx * 2u + 1u];
        let pixel_idx = y * config.imgSize + x;
        errorBuffer[pixel_idx] -= config.lineWeight;
    }

    // Thread 0 updates state
    workgroupBarrier();
    if (tid == 0u) {
        state.x = bestPin;  // Update current pin
        state.y = iteration + 1u;  // Increment iteration
    }
}
