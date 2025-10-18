// Main WebGPU orchestration for string art generation

import { generateLineCache } from './lineCache.js';
import { captureVideoFrame, convertToLuminosity, createErrorBuffer } from './preprocessing.js';
import { renderStringArt } from './renderer.js';

// Configuration (matching main.go defaults)
const CONFIG = {
    IMG_SIZE: 200,
    OUTPUT_SIZE: 500,
    PINS: 300,
    MIN_DISTANCE: 30,
    MAX_LINES: 4000,
    LINE_WEIGHT: 8.0
};

let gpuContext = null;
let lineCache = null;
let isProcessing = false;
let isPaused = false;
let frameCount = 0;
let lastFrameTime = performance.now();
let fps = 0;

export async function initWebGPU() {
    console.log('Initializing WebGPU...');

    // Check WebGPU support
    if (!navigator.gpu) {
        throw new Error('WebGPU not supported in this browser');
    }

    // Request adapter and device
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error('Failed to get WebGPU adapter');
    }

    // Check adapter limits
    console.log('GPU limits:', {
        maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup
    });

    // Request device with higher workgroup limits
    const device = await adapter.requestDevice({
        requiredLimits: {
            maxComputeWorkgroupSizeX: Math.min(1024, adapter.limits.maxComputeWorkgroupSizeX),
            maxComputeWorkgroupSizeY: Math.min(1024, adapter.limits.maxComputeWorkgroupSizeY),
            maxComputeInvocationsPerWorkgroup: Math.min(1024, adapter.limits.maxComputeInvocationsPerWorkgroup)
        }
    });
    console.log('WebGPU device acquired with extended limits');

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

    console.log('Shader compiled and pipeline created');

    // Generate line cache
    console.log('Generating line cache...');
    lineCache = generateLineCache(CONFIG.IMG_SIZE, CONFIG.PINS, CONFIG.MIN_DISTANCE);

    // Create GPU buffers
    const errorBuffer = device.createBuffer({
        size: CONFIG.IMG_SIZE * CONFIG.IMG_SIZE * 4,
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
        size: (CONFIG.MAX_LINES + 1) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        label: 'Line Sequence Buffer'
    });

    const stateBuffer = device.createBuffer({
        size: 8, // 2 u32 values: currentPin, iteration
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        label: 'State Buffer'
    });

    const configBuffer = device.createBuffer({
        size: 32, // 5 u32 values with padding
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        label: 'Config Buffer'
    });

    const readbackBuffer = device.createBuffer({
        size: (CONFIG.MAX_LINES + 1) * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        label: 'Readback Buffer'
    });

    // Upload static data to GPU
    device.queue.writeBuffer(lineCoordBuffer, 0, lineCache.lineCoordBuffer);
    device.queue.writeBuffer(lineMetadataBuffer, 0, lineCache.lineMetadataBuffer);

    // Initialize state buffer
    const initialState = new Uint32Array([0, 0]); // currentPin=0, iteration=0
    device.queue.writeBuffer(stateBuffer, 0, initialState);

    // Initialize line sequence with pin 0
    const initialLineSeq = new Uint32Array(CONFIG.MAX_LINES + 1);
    initialLineSeq[0] = 0;
    device.queue.writeBuffer(lineSequenceBuffer, 0, initialLineSeq);

    // Upload config
    const configData = new Uint32Array([
        CONFIG.IMG_SIZE,
        CONFIG.PINS,
        CONFIG.MIN_DISTANCE,
        CONFIG.MAX_LINES,
        0, // padding
        0, // padding
        0, // padding
        0  // padding
    ]);
    const configDataF32 = new Float32Array(configData.buffer);
    configDataF32[4] = CONFIG.LINE_WEIGHT;
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

    console.log('GPU buffers created and initialized');

    gpuContext = {
        device,
        pipeline,
        bindGroup,
        buffers: {
            errorBuffer,
            lineSequenceBuffer,
            stateBuffer,
            readbackBuffer
        }
    };

    return gpuContext;
}

export async function processFrame(videoElement, outputCanvas) {
    if (!gpuContext || isProcessing || isPaused) return;
    isProcessing = true;

    const startTime = performance.now();

    try {
        const { device, pipeline, bindGroup, buffers } = gpuContext;

        // 1. Capture and preprocess frame
        const imageData = captureVideoFrame(videoElement, CONFIG.IMG_SIZE);
        const luminosity = convertToLuminosity(imageData);
        const errorArray = createErrorBuffer(luminosity);

        // 2. Upload error buffer to GPU
        device.queue.writeBuffer(buffers.errorBuffer, 0, errorArray);

        // Reset state buffer
        const initialState = new Uint32Array([0, 0]);
        device.queue.writeBuffer(buffers.stateBuffer, 0, initialState);

        // 3. Submit all iterations as separate command buffers
        // GPU queue guarantees ordering - each dispatch sees previous writes
        for (let iter = 0; iter < CONFIG.MAX_LINES; iter++) {
            const commandEncoder = device.createCommandEncoder();
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(pipeline);
            computePass.setBindGroup(0, bindGroup);
            computePass.dispatchWorkgroups(1);
            computePass.end();
            device.queue.submit([commandEncoder.finish()]);
        }

        // 4. Copy result to readback buffer
        const readbackEncoder = device.createCommandEncoder();
        readbackEncoder.copyBufferToBuffer(
            buffers.lineSequenceBuffer, 0,
            buffers.readbackBuffer, 0,
            (CONFIG.MAX_LINES + 1) * 4
        );
        device.queue.submit([readbackEncoder.finish()]);

        // 6. Readback line sequence
        await buffers.readbackBuffer.mapAsync(GPUMapMode.READ);
        const lineSequenceData = new Uint32Array(buffers.readbackBuffer.getMappedRange());
        const lineSequence = Array.from(lineSequenceData);
        buffers.readbackBuffer.unmap();

        // 7. Render to canvas
        renderStringArt(outputCanvas, lineSequence, lineCache.pinCoords, CONFIG.IMG_SIZE, CONFIG.OUTPUT_SIZE);

        // 8. Update FPS
        const endTime = performance.now();
        const frameTime = endTime - startTime;
        frameCount++;

        const timeSinceLastUpdate = endTime - lastFrameTime;
        if (timeSinceLastUpdate >= 1000) {
            fps = (frameCount * 1000) / timeSinceLastUpdate;
            frameCount = 0;
            lastFrameTime = endTime;
        }

        // Update FPS display
        const fpsElement = document.getElementById('fps');
        if (fpsElement) {
            fpsElement.textContent = `FPS: ${fps.toFixed(2)} | Frame time: ${frameTime.toFixed(0)}ms`;
        }

    } catch (error) {
        console.error('Error processing frame:', error);
    } finally {
        isProcessing = false;
    }
}

export function setPaused(paused) {
    isPaused = paused;
}

export function isPausedState() {
    return isPaused;
}

export function getConfig() {
    return CONFIG;
}
