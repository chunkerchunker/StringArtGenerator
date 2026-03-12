// WebGPU String Art Processor
// Replaces WASM implementation with GPU compute shaders

import { generateLineCache } from './lineCache.js';
import { convertToLuminosity, createErrorBuffer } from './preprocessing.js';

let gpuContext = null;
let lineCache = null;
let currentConfig = null;
let processingLock = null; // Mutex to prevent concurrent processImage calls

export async function initWebGPU(config) {
    console.log('Initializing WebGPU processor...', config);

    // Check WebGPU support
    if (!navigator.gpu) {
        throw new Error('WebGPU not supported in this browser');
    }

    // Request adapter and device
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error('Failed to get WebGPU adapter');
    }

    const device = await adapter.requestDevice({
        requiredLimits: {
            maxComputeWorkgroupSizeX: Math.min(1024, adapter.limits.maxComputeWorkgroupSizeX),
            maxComputeWorkgroupSizeY: Math.min(1024, adapter.limits.maxComputeWorkgroupSizeY),
            maxComputeInvocationsPerWorkgroup: Math.min(1024, adapter.limits.maxComputeInvocationsPerWorkgroup),
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize
        }
    });

    // Load shader
    const shaderResponse = await fetch('shader.wgsl');
    const shaderCode = await shaderResponse.text();

    // Create shader module
    const shaderModule = device.createShaderModule({
        code: shaderCode,
        label: 'String Art Compute Shader'
    });

    // Create compute pipeline
    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        },
        label: 'String Art Pipeline'
    });

    // Generate line cache if config changed
    const configChanged = !currentConfig ||
        currentConfig.imgSize !== config.imgSize ||
        currentConfig.pins !== config.pins ||
        currentConfig.minDistance !== config.minDistance;

    if (configChanged) {
        console.log('Generating line cache...');
        lineCache = generateLineCache(config.imgSize, config.pins, config.minDistance);
        currentConfig = { ...config };
    }

    // Create GPU buffers
    const errorBuffer = device.createBuffer({
        size: config.imgSize * config.imgSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'Error Buffer'
    });

    const lineCoordBuffer = device.createBuffer({
        size: lineCache.lineCoordBuffer.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'Line Coord Buffer'
    });

    const lineMetadataBuffer = device.createBuffer({
        size: lineCache.lineMetadataBuffer.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'Line Metadata Buffer'
    });

    const lineSequenceBuffer = device.createBuffer({
        size: (config.maxLines + 1) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        label: 'Line Sequence Buffer'
    });

    const stateBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        label: 'State Buffer'
    });

    const configBuffer = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        label: 'Config Buffer'
    });

    const readbackBuffer = device.createBuffer({
        size: (config.maxLines + 1) * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        label: 'Readback Buffer'
    });

    const stateReadbackBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        label: 'State Readback Buffer'
    });

    // Upload static data to GPU
    device.queue.writeBuffer(lineCoordBuffer, 0, lineCache.lineCoordBuffer);
    device.queue.writeBuffer(lineMetadataBuffer, 0, lineCache.lineMetadataBuffer);

    // Initialize state buffer
    const initialState = new Uint32Array([0, 0, 0, 0]);
    device.queue.writeBuffer(stateBuffer, 0, initialState);

    // Initialize line sequence with pin 0
    const initialLineSeq = new Uint32Array(config.maxLines + 1);
    initialLineSeq[0] = 0;
    device.queue.writeBuffer(lineSequenceBuffer, 0, initialLineSeq);

    // Upload config
    const configData = new Uint32Array([
        config.imgSize,
        config.pins,
        config.minDistance,
        config.maxLines,
        0, // lineWeight (will be set as f32 below)
        config.iterationsPerDispatch || 100,
        0, // padding
        0  // padding
    ]);
    const configDataF32 = new Float32Array(configData.buffer);
    configDataF32[4] = config.lineWeight;
    device.queue.writeBuffer(configBuffer, 0, configData);

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: errorBuffer } },
            { binding: 1, resource: { buffer: lineCoordBuffer } },
            { binding: 2, resource: { buffer: lineMetadataBuffer } },
            { binding: 3, resource: { buffer: lineSequenceBuffer } },
            { binding: 4, resource: { buffer: stateBuffer } },
            { binding: 5, resource: { buffer: configBuffer } }
        ],
        label: 'String Art Bind Group'
    });

    gpuContext = {
        device,
        pipeline,
        bindGroup,
        buffers: {
            errorBuffer,
            lineSequenceBuffer,
            stateBuffer,
            readbackBuffer,
            stateReadbackBuffer,
            configBuffer
        },
        config
    };

    return gpuContext;
}

export async function processImage(imageData, config) {
    // Wait for any ongoing processing to complete (mutex)
    while (processingLock) {
        await processingLock;
    }

    // Create a new lock for this processing operation
    let releaseLock;
    processingLock = new Promise(resolve => {
        releaseLock = resolve;
    });

    try {
        // Check if we need to reinitialize (critical params changed)
        const needsReinit = !gpuContext ||
            gpuContext.config.imgSize !== config.imgSize ||
            gpuContext.config.pins !== config.pins ||
            gpuContext.config.minDistance !== config.minDistance ||
            gpuContext.config.maxLines !== config.maxLines;

        if (needsReinit) {
            console.log('Reinitializing WebGPU due to config change:', {
                old: gpuContext?.config,
                new: config
            });
            await initWebGPU(config);
        }
        // Update config if only lineWeight or iterationsPerDispatch changed
        else if (config.lineWeight !== gpuContext.config.lineWeight ||
                 (config.iterationsPerDispatch || 100) !== (gpuContext.config.iterationsPerDispatch || 100)) {
            const configData = new Uint32Array([
                config.imgSize,
                config.pins,
                config.minDistance,
                config.maxLines,
                0,
                config.iterationsPerDispatch || 100,
                0,
                0
            ]);
            const configDataF32 = new Float32Array(configData.buffer);
            configDataF32[4] = config.lineWeight;
            gpuContext.device.queue.writeBuffer(gpuContext.buffers.configBuffer, 0, configData);
            gpuContext.config.lineWeight = config.lineWeight;
            gpuContext.config.iterationsPerDispatch = config.iterationsPerDispatch || 100;
        }

        const { device, pipeline, bindGroup, buffers } = gpuContext;

        // Convert image to luminosity and error buffer
        const luminosity = convertToLuminosity(imageData);
        const errorArray = createErrorBuffer(luminosity);

        // Upload error buffer to GPU
        device.queue.writeBuffer(buffers.errorBuffer, 0, errorArray);

        // Reset state buffer
        const initialState = new Uint32Array([0, 0, 0, 0]);
        device.queue.writeBuffer(buffers.stateBuffer, 0, initialState);

        // Reset line sequence buffer with sentinel value (0xFFFFFFFF) so that
        // pin 0 is distinguishable from "not written"
        const SENTINEL = 0xFFFFFFFF;
        const emptyLineSeq = new Uint32Array(config.maxLines + 1);
        emptyLineSeq.fill(SENTINEL);
        emptyLineSeq[0] = 0; // Start at pin 0
        device.queue.writeBuffer(buffers.lineSequenceBuffer, 0, emptyLineSeq);

        // Submit batched iterations
        const numDispatches = Math.ceil(config.maxLines / (config.iterationsPerDispatch || 100));
        for (let batch = 0; batch < numDispatches; batch++) {
            const commandEncoder = device.createCommandEncoder();
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(pipeline);
            computePass.setBindGroup(0, bindGroup);
            computePass.dispatchWorkgroups(1);
            computePass.end();
            device.queue.submit([commandEncoder.finish()]);
        }

        // Copy results to readback buffers
        const readbackEncoder = device.createCommandEncoder();
        readbackEncoder.copyBufferToBuffer(
            buffers.lineSequenceBuffer, 0,
            buffers.readbackBuffer, 0,
            (config.maxLines + 1) * 4
        );
        readbackEncoder.copyBufferToBuffer(
            buffers.stateBuffer, 0,
            buffers.stateReadbackBuffer, 0,
            16
        );
        device.queue.submit([readbackEncoder.finish()]);

        // Readback line sequence
        await buffers.readbackBuffer.mapAsync(GPUMapMode.READ);
        const lineSequenceData = new Uint32Array(buffers.readbackBuffer.getMappedRange());
        const fullSequence = Array.from(lineSequenceData);
        buffers.readbackBuffer.unmap();

        // Determine actual line count from buffer data.
        // The buffer is filled with sentinel (0xFFFFFFFF) before each run.
        // The shader writes valid pin indices sequentially starting at index 1.
        // Find the first sentinel to determine where real data ends.
        const SENTINEL_READ = 0xFFFFFFFF;
        let actualLineCount = fullSequence.length - 1; // default: assume all used
        for (let i = 1; i < fullSequence.length; i++) {
            if (fullSequence[i] === SENTINEL_READ) {
                actualLineCount = i - 1;
                break;
            }
        }

        const lineSequence = fullSequence.slice(0, actualLineCount + 1);

        return {
            lineSequence,
            lineCount: actualLineCount,
            pinCoords: lineCache.pinCoords
        };
    } finally {
        // Release the lock
        releaseLock();
        processingLock = null;
    }
}

export function cleanup() {
    // WebGPU resources are garbage collected, but we can clear references
    gpuContext = null;
}

export function getPinCoords() {
    return lineCache ? lineCache.pinCoords : null;
}
