import * as WebGPUProcessor from "./webgpu-processor.js";

let isProcessing = false;
let isAutoCancelled = false;

// Global storage for generated string art data
let currentLineSequence = null;
let currentPinCount = 0;
let currentLineCount = 0;

// Auto-regeneration variables
let regenerationTimeout = null;
let hasInitialImage = false;

// Zoom and pan variables
let zoomLevel = 1;
let panX = 0;
let panY = 0;

// Zoom and pan state for original image
let imgZoomLevel = 1;
let imgPanX = 0;
let imgPanY = 0;

// AbortControllers for event listener cleanup
let zoomPanAbortController = null;
let imgZoomPanAbortController = null;

// FPS tracking
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

// Store original image and display info for cropping
let originalImage = null;
let imageDisplayInfo = null;

// Initialize range input displays
document.addEventListener("DOMContentLoaded", () => {
    const ranges = [
        "pins",
        "maxLines",
        "targetSize",
        "renderLineWidth",
        "lineWeight",
        "minDistance",
        "webcamContrast",
    ];
    ranges.forEach((id) => {
        const input = document.getElementById(id);
        const display = document.getElementById(`${id}Value`);

        input.addEventListener("input", () => {
            display.textContent = input.value;

            // Update canvas dynamically for render line width if we have generated data
            if (currentLineSequence && id === "renderLineWidth") {
                const renderLineWidth = parseFloat(
                    document.getElementById("renderLineWidth").value,
                );

                // Re-render with new line width
                updateCanvasLineWidth(renderLineWidth);
            }

            // Update image canvas when targetSize changes
            if (id === "targetSize" && originalImage) {
                updateImageCanvas();
            }
        });

        // Add change event for auto-regeneration (fires when user releases slider)
        input.addEventListener("change", () => {
            const inputParams = [
                "pins",
                "maxLines",
                "targetSize",
                "lineWeight",
                "minDistance",
            ];
            if (inputParams.includes(id)) {
                scheduleAutoRegeneration();
            }
        });
    });

    // Setup auto line weight checkbox
    const autoLineWeightCheckbox = document.getElementById("autoLineWeight");
    const lineWeightSlider = document.getElementById("lineWeight");
    const lineWeightValue = document.getElementById("lineWeightValue");

    autoLineWeightCheckbox.addEventListener("change", async () => {
        if (autoLineWeightCheckbox.checked) {
            // Special handling for webcam mode
            if (webcamMode && webcamStream) {
                // Pause webcam processing
                if (webcamProcessingLoop) {
                    cancelAnimationFrame(webcamProcessingLoop);
                    webcamProcessingLoop = null;
                }

                // Run auto line weight search once
                await generateWithAutoLineWeight();

                // Uncheck the auto checkbox after completion
                autoLineWeightCheckbox.checked = false;
                lineWeightSlider.disabled = false;
                lineWeightValue.textContent = lineWeightSlider.value;
                lineWeightSlider.classList.remove("opacity-50");
                lineWeightSlider.classList.add("opacity-100");

                // Resume webcam processing
                if (webcamStream && webcamMode) {
                    startWebcamProcessing();
                }

                return;
            }

            // Normal mode (not webcam)
            lineWeightSlider.disabled = true;
            lineWeightValue.textContent = "Auto";
            lineWeightSlider.classList.remove("opacity-100");
            lineWeightSlider.classList.add("opacity-50");
            // Only trigger auto-regeneration when enabling auto mode
            scheduleAutoRegeneration();
        } else {
            lineWeightSlider.disabled = false;
            lineWeightValue.textContent = lineWeightSlider.value;
            lineWeightSlider.classList.remove("opacity-50");
            lineWeightSlider.classList.add("opacity-100");
            // Don't regenerate when disabling auto mode
        }
    });

    // Initialize WebGPU
    initializeWebGPU();

    // Add window resize listener to update canvas size responsively
    let resizeTimeout = null;
    window.addEventListener('resize', () => {
        // Debounce resize events
        if (resizeTimeout) {
            clearTimeout(resizeTimeout);
        }

        resizeTimeout = setTimeout(() => {
            // Only update if we have generated string art and result tab is active
            const resultTab = document.getElementById('resultTab');
            if (currentLineSequence && resultTab && resultTab.classList.contains('active')) {
                // Wait for browser to reflow layout before reading dimensions
                requestAnimationFrame(() => {
                    updateOutputDisplay();
                });
            }
        }, 150); // 150ms debounce delay
    });
});

async function initializeWebGPU() {
    try {
        showStatus("Initializing WebGPU...", "loading");

        // Check WebGPU support
        if (!navigator.gpu) {
            throw new Error(
                "WebGPU not supported in this browser. Please use Chrome 113+ or Safari 18+",
            );
        }

        showStatus("WebGPU initialized successfully!", "success");
        setTimeout(() => hideStatus(), 2000);
    } catch (error) {
        showStatus(`Error initializing WebGPU: ${error.message}`, "error");
        console.error("WebGPU initialization error:", error);
    }
}

// Tab switching function
// biome-ignore lint/correctness/noUnusedVariables: called from html
function switchTab(tabName) {
    switchTabProgrammatically(tabName);
}

// Export functions to window for HTML onclick handlers
window.switchTab = switchTab;
window.generateStringArt = generateStringArt;
window.toggleSidebar = toggleSidebar;
window.adjustZoom = adjustZoom;
window.resetZoom = resetZoom;
window.adjustImgZoom = adjustImgZoom;
window.resetImgZoom = resetImgZoom;
window.toggleInputMode = toggleInputMode;
window.startWebcam = startWebcam;
window.stopWebcam = stopWebcam;

document.getElementById("imageInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
        // Update the file input text
        document.getElementById("fileInputText").textContent = file.name;

        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                displayImageWithZoom(img);
                hasInitialImage = true;
                // Switch to Original Image tab after loading
                switchTabProgrammatically("original");
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

// Display image with zoom and pan controls
function displayImageWithZoom(img) {
    const originalTab = document.getElementById("originalTab");
    originalTab.innerHTML = "";

    // Ensure the tab has relative positioning for absolute positioned controls
    originalTab.style.position = "relative";

    // Store original image for cropping
    originalImage = img;

    // Get target size from input
    const targetSize =
        parseInt(document.getElementById("targetSize").value) || 1000;

    // Create zoomable container
    const zoomableContainer = document.createElement("div");
    zoomableContainer.className = "img-zoomable-container";
    zoomableContainer.style.width = targetSize + "px";
    zoomableContainer.style.height = targetSize + "px";

    // Create content wrapper
    const zoomableContent = document.createElement("div");
    zoomableContent.className = "img-zoomable-content";

    // Calculate scale to fit image in square (this becomes our base scale)
    const baseScale = Math.min(targetSize / img.width, targetSize / img.height);
    const baseScaledWidth = img.width * baseScale;
    const baseScaledHeight = img.height * baseScale;

    // Calculate minimum zoom to fill entire targetSize (both dimensions)
    const minZoomToFillWidth = targetSize / baseScaledWidth;
    const minZoomToFillHeight = targetSize / baseScaledHeight;
    const minZoomLevel = Math.max(minZoomToFillWidth, minZoomToFillHeight);

    // Set initial zoom to minimum required
    imgZoomLevel = minZoomLevel;

    // Center the image
    const offsetX = (targetSize - baseScaledWidth) / 2;
    const offsetY = (targetSize - baseScaledHeight) / 2;

    // Store display info for cropping
    imageDisplayInfo = {
        targetSize,
        baseScale,
        baseScaledWidth,
        baseScaledHeight,
        offsetX,
        offsetY,
        minZoomLevel,
    };

    // Create canvas for the image
    const canvas = document.createElement("canvas");
    canvas.width = targetSize;
    canvas.height = targetSize;
    canvas.style.width = targetSize + "px";
    canvas.style.height = targetSize + "px";

    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, targetSize, targetSize);

    // Apply zoom transformation and draw image
    ctx.save();
    ctx.translate(targetSize / 2, targetSize / 2);
    ctx.translate(imgPanX, imgPanY);
    ctx.scale(imgZoomLevel, imgZoomLevel);
    ctx.translate(-targetSize / 2, -targetSize / 2);
    ctx.drawImage(img, offsetX, offsetY, baseScaledWidth, baseScaledHeight);
    ctx.restore();

    // Add circular lightbox effect
    addCircularLightboxEffect(ctx, targetSize);

    zoomableContent.appendChild(canvas);
    zoomableContent.dataset.originalImg = img.src;
    zoomableContent.dataset.imgWidth = img.width;
    zoomableContent.dataset.imgHeight = img.height;

    // Add zoom controls
    const zoomControls = document.createElement("div");
    zoomControls.className = "zoom-controls";
    zoomControls.innerHTML = `
        <button class="zoom-btn" onclick="adjustImgZoom(0.2)">+</button>
        <button class="zoom-btn" onclick="adjustImgZoom(-0.2)">−</button>
        <button class="zoom-btn" onclick="resetImgZoom()">Reset</button>
    `;

    // Add zoom info
    const zoomInfo = document.createElement("div");
    zoomInfo.className = "zoom-info";
    zoomInfo.id = "imgZoomInfo";
    zoomInfo.textContent = "Zoom: 100% | Cmd+Scroll to zoom, Drag to pan";

    zoomableContainer.appendChild(zoomableContent);
    originalTab.appendChild(zoomableContainer);

    // Append zoom controls and info to the main tab area (not the small container)
    originalTab.appendChild(zoomControls);
    originalTab.appendChild(zoomInfo);

    // Setup zoom and pan for image
    setupImageZoomAndPan(
        zoomableContainer,
        zoomableContent,
        canvas,
        img,
        targetSize,
    );

    // Reset pan when switching tabs or regenerating (zoom is already set to minimum)
    imgPanX = 0;
    imgPanY = 0;
}

// Schedule auto-regeneration with debouncing
function scheduleAutoRegeneration() {
    const autoRegenerate = document.getElementById("autoRegenerate");
    if (!autoRegenerate.checked || !hasInitialImage || isProcessing) {
        return;
    }

    // Only regenerate if the result tab is active
    const resultTab = document.getElementById("resultTab");
    if (!resultTab || !resultTab.classList.contains("active")) {
        return;
    }

    // Clear existing timeout
    if (regenerationTimeout) {
        clearTimeout(regenerationTimeout);
    }

    // Schedule regeneration after a short delay
    regenerationTimeout = setTimeout(() => {
        generateStringArt();
    }, 500); // 500ms delay after user releases slider
}

// Binary search for optimal line weight
async function generateWithAutoLineWeight() {
    if (isProcessing) {
        return;
    }

    const fileInput = document.getElementById("imageInput");

    // Check if we have either a file input OR a saved webcam frame
    if (!fileInput.files[0] && !originalImage && !lastWebcamFrame) {
        alert("Please select an image file or capture a webcam frame first!");
        return;
    }

    // Check WebGPU support
    if (!navigator.gpu) {
        showStatus(
            "WebGPU not supported. Please use a compatible browser.",
            "error",
        );
        return;
    }

    isProcessing = true;
    isAutoCancelled = false;
    const generateBtn = document.getElementById("generateBtn");

    // Set up cancellation handler (define before try block for finally access)
    const cancelHandler = () => {
        isAutoCancelled = true;
        generateBtn.textContent = "Cancelling...";
        generateBtn.disabled = true;
    };

    generateBtn.disabled = false; // Keep enabled for cancellation
    generateBtn.textContent = "Cancel auto-generation";
    generateBtn.addEventListener("click", cancelHandler);

    // Clear previous generation data
    currentLineSequence = null;
    currentPinCount = 0;
    currentLineCount = 0;

    // Clear any pending auto-regeneration
    if (regenerationTimeout) {
        clearTimeout(regenerationTimeout);
        regenerationTimeout = null;
    }

    // Disable all controls during binary search
    const controls = [
        "pins",
        "maxLines",
        "targetSize",
        "minDistance",
        "renderLineWidth",
        "lineWeight",
        "autoLineWeight",
        "autoRegenerate",
        "imageInput",
    ];
    controls.forEach((id) => {
        const element = document.getElementById(id);
        if (element) element.disabled = true;
    });

    try {
        showStatus("Starting auto line weight search...", "loading");
        showProgress(0);

        // Get parameters
        const pins = parseInt(document.getElementById("pins").value);
        const targetMaxLines = parseInt(document.getElementById("maxLines").value);
        const targetSize = parseInt(document.getElementById("targetSize").value);
        const minDistance = parseInt(document.getElementById("minDistance").value);

        // Load image data once
        const file = fileInput.files[0] || null; // null if using webcam frame
        const imageData = await loadImageData(file);

        // Binary search parameters
        // Lower weight = more lines, Higher weight = fewer lines
        // We want to find the minimum weight that produces <= targetMaxLines
        let low = 1;
        let high = 200;
        let bestWeight = 20;
        let iteration = 0;
        const maxIterations = 10;

        console.log(
            `Starting binary search for target ${targetMaxLines} lines (lower weight = more lines)`,
        );

        while (low <= high && iteration < maxIterations && !isAutoCancelled) {
            const mid = Math.floor((low + high) / 2);
            iteration++;

            showStatus(
                `Testing weight ${mid} (iteration ${iteration})...`,
                "loading",
            );
            showProgress((iteration / maxIterations) * 90);

            // Process image with current weight
            await new Promise((resolve) => setTimeout(resolve, 50)); // Small delay for UI update

            const config = {
                imgSize: targetSize,
                pins: pins,
                minDistance: minDistance,
                maxLines: targetMaxLines,
                lineWeight: mid,
                iterationsPerDispatch: 100,
            };

            const result = await WebGPUProcessor.processImage(imageData, config);
            const lineCount = result.lineCount;

            console.log(
                `Weight ${mid}: ${lineCount} lines (target: ${targetMaxLines})`,
            );

            // Update the display with current result
            if (lineCount > 0) {
                currentLineSequence = result.lineSequence;
                currentPinCount = pins;
                currentLineCount = lineCount;
                updateOutputDisplay();
            }

            // Update binary search bounds (lower weight = more lines)
            if (lineCount <= targetMaxLines) {
                // This weight produces acceptable line count, can we go lower (more lines)?
                bestWeight = mid;

                if (lineCount === targetMaxLines) {
                    // Perfect match found
                    console.log(`Perfect match found: weight ${mid}`);
                    break;
                }

                // Try lower weight to get closer to target (more lines)
                high = mid - 1;
            } else {
                // Too many lines, increase weight to reduce line count
                low = mid + 1;
            }
        }

        // Check if cancelled before final generation
        if (isAutoCancelled) {
            showStatus("Auto-generation cancelled", "error");
            throw new Error("Auto-generation cancelled by user");
        }

        // Final generation with best weight
        showStatus(`Finalizing with optimal weight ${bestWeight}...`, "loading");
        showProgress(95);

        const finalConfig = {
            imgSize: targetSize,
            pins: pins,
            minDistance: minDistance,
            maxLines: targetMaxLines,
            lineWeight: bestWeight,
            iterationsPerDispatch: 100,
        };

        const finalResult = await WebGPUProcessor.processImage(
            imageData,
            finalConfig,
        );
        const finalLineCount = finalResult.lineCount;

        // Update with final result
        currentLineSequence = finalResult.lineSequence;
        currentPinCount = pins;
        currentLineCount = finalLineCount;
        updateOutputDisplay();

        // Update the line weight slider to show the found value
        const lineWeightSlider = document.getElementById("lineWeight");
        const lineWeightValue = document.getElementById("lineWeightValue");
        lineWeightSlider.value = bestWeight;
        lineWeightValue.textContent = `Auto (${bestWeight})`;

        showProgress(100);
        showStatus(
            `Found minimum weight: ${bestWeight} for ${finalLineCount} lines (target: ≤${targetMaxLines})`,
            "success",
        );

        setTimeout(() => hideStatus(), 3000);
    } catch (error) {
        if (!isAutoCancelled) {
            showStatus(`Error during auto weight search: ${error.message}`, "error");
        }
        console.error("Auto weight search error:", error);
    } finally {
        isProcessing = false;
        isAutoCancelled = false;

        // Remove the cancellation handler and reset button
        const generateBtn = document.getElementById("generateBtn");
        generateBtn.removeEventListener("click", cancelHandler);
        generateBtn.disabled = false;
        generateBtn.textContent = "Generate String Art";
        hideProgress();

        // Re-enable all controls
        const controls = [
            "pins",
            "maxLines",
            "targetSize",
            "minDistance",
            "renderLineWidth",
            "autoRegenerate",
            "imageInput",
            "autoLineWeight",
        ];
        controls.forEach((id) => {
            const element = document.getElementById(id);
            if (element) element.disabled = false;
        });

        // Re-enable line weight slider based on checkbox state
        const autoLineWeightCheckbox = document.getElementById("autoLineWeight");
        const lineWeightSlider = document.getElementById("lineWeight");
        if (!autoLineWeightCheckbox.checked) {
            lineWeightSlider.disabled = false;
        }
    }
}

async function generateStringArt() {
    console.log("=== generateStringArt called ===");

    if (isProcessing) {
        console.log("Already processing, returning");
        return;
    }

    console.log("Checking for image source...");
    const fileInput = document.getElementById("imageInput");

    // Check if we have either a file input OR a saved webcam frame
    if (!fileInput.files[0] && !originalImage && !lastWebcamFrame) {
        console.log("No image available");
        alert("Please select an image file or capture a webcam frame first!");
        return;
    }

    if (fileInput.files[0]) {
        console.log("File selected:", fileInput.files[0].name);
    } else if (lastWebcamFrame || originalImage) {
        console.log("Using saved webcam frame");
    }

    // Check if auto line weight is enabled
    const autoLineWeightCheckbox = document.getElementById("autoLineWeight");
    if (autoLineWeightCheckbox.checked) {
        console.log("Auto line weight enabled, starting binary search...");
        await generateWithAutoLineWeight();
        return;
    }

    // Check WebGPU support
    if (!navigator.gpu) {
        showStatus(
            "WebGPU not supported. Please use a compatible browser.",
            "error",
        );
        console.error("WebGPU not supported");
        return;
    }

    isProcessing = true;
    const generateBtn = document.getElementById("generateBtn");
    generateBtn.disabled = true;
    generateBtn.textContent = "Processing...";

    // Clear previous generation data
    currentLineSequence = null;
    currentPinCount = 0;
    currentLineCount = 0;

    // Clear any pending auto-regeneration
    if (regenerationTimeout) {
        clearTimeout(regenerationTimeout);
        regenerationTimeout = null;
    }

    try {
        console.log("Starting try block...");
        showStatus("Processing image...", "loading");
        showProgress(0);
        console.log("Status and progress set...");

        // Get parameters
        console.log("Getting parameters...");
        const pins = parseInt(document.getElementById("pins").value);
        const maxLines = parseInt(document.getElementById("maxLines").value);
        const targetSize = parseInt(document.getElementById("targetSize").value);
        const lineWeight = parseInt(document.getElementById("lineWeight").value);
        const minDistance = parseInt(document.getElementById("minDistance").value);
        console.log("Parameters retrieved:", {
            pins,
            maxLines,
            targetSize,
            lineWeight,
            minDistance,
        });

        // Validate parameters
        console.log("Using parameters:", {
            pins,
            maxLines,
            targetSize,
            lineWeight,
            minDistance,
        });

        if (pins < 50 || pins > 1000) throw new Error("Invalid pins count");
        if (maxLines < 10 || maxLines > 50000) throw new Error("Invalid max lines");
        if (targetSize < 50 || targetSize > 2000)
            throw new Error("Invalid target size");

        showProgress(20);

        // Load and process image
        const file = fileInput.files[0] || null; // null if using webcam frame
        console.log("Loading image data...");
        const imageData = await loadImageData(file);
        console.log("Image data loaded:", imageData);

        showProgress(40);
        showStatus("Calculating string art lines...", "loading");

        // Process the image (this is the heavy computation)
        console.log("Starting image processing...");

        // Use setTimeout to yield control back to browser for UI updates
        await new Promise((resolve) => setTimeout(resolve, 100));

        const startTime = Date.now();

        // Process with WebGPU
        const config = {
            imgSize: targetSize,
            pins: pins,
            minDistance: minDistance,
            maxLines: maxLines,
            lineWeight: lineWeight,
            iterationsPerDispatch: 100,
        };

        const result = await WebGPUProcessor.processImage(imageData, config);

        const endTime = Date.now();
        console.log("Processing completed. Line count:", result.lineCount);
        console.log("Processing time:", (endTime - startTime) / 1000, "seconds");

        showProgress(80);
        showStatus("Generating output...", "loading");

        // Store data globally for dynamic updates
        currentLineSequence = result.lineSequence;
        currentPinCount = pins;
        currentLineCount = result.lineCount;

        // Create and display initial output
        updateOutputDisplay();

        showProgress(100);
        showStatus(
            `String art generated successfully! Used ${currentLineCount} lines.`,
            "success",
        );

        setTimeout(() => hideStatus(), 3000);
    } catch (error) {
        showStatus(`Error generating string art: ${error.message}`, "error");
        console.error("Generation error:", error);
    } finally {
        isProcessing = false;
        generateBtn.disabled = false;
        generateBtn.textContent = "Generate String Art";
        hideProgress();
    }
}

async function loadImageData(file) {
    return new Promise((resolve, reject) => {
        // If we have an original image and display info, use the cropped version
        if (originalImage && imageDisplayInfo) {
            try {
                const croppedImageData = getCroppedImageDataForWebGPU();
                resolve(croppedImageData);
                return;
            } catch (error) {
                console.warn(
                    "Failed to get cropped image data, falling back to full image:",
                    error,
                );
            }
        }

        // If no file provided but we have a saved webcam frame, use it
        if (!file && (lastWebcamFrame || originalImage)) {
            const img = lastWebcamFrame || originalImage;
            try {
                const canvas = document.createElement("canvas");
                const ctx = canvas.getContext("2d");
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);

                const imageData = ctx.getImageData(0, 0, img.width, img.height);
                resolve(imageData);
                return;
            } catch (error) {
                reject(error);
                return;
            }
        }

        // Fallback to loading from file
        if (!file) {
            reject(new Error("No image source available"));
            return;
        }

        const img = new Image();
        img.onload = () => {
            try {
                const canvas = document.createElement("canvas");
                const ctx = canvas.getContext("2d");
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);

                const imageData = ctx.getImageData(0, 0, img.width, img.height);
                resolve(imageData);
            } catch (error) {
                reject(error);
            }
        };
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
    });
}

function showStatus(message, type) {
    const status = document.getElementById("status");
    status.textContent = message;
    status.className = `status ${type}`;
    status.style.display = "block";
}

function hideStatus() {
    document.getElementById("status").style.display = "none";
}

function showProgress(percent) {
    const progressBar = document.getElementById("progressBar");
    const progressFill = document.getElementById("progressFill");
    progressBar.style.display = "block";
    progressFill.style.width = `${percent}%`;
}

function hideProgress() {
    document.getElementById("progressBar").style.display = "none";
}

// Calculate pin coordinates based on stringart_core.c logic
function calculatePinCoordinates(pinCount, outputSize) {
    const center = outputSize * 0.5;
    const radius = center - 1.0;
    const angleStep = (2.0 * Math.PI) / pinCount;
    const pinCoords = [];

    for (let i = 0; i < pinCount; i++) {
        const angle = angleStep * i;
        const x = Math.floor(center + radius * Math.cos(angle));
        const y = Math.floor(center + radius * Math.sin(angle));
        pinCoords.push({ x, y });
    }

    return pinCoords;
}

// Update canvas line width efficiently without full re-render
function updateCanvasLineWidth(lineWidth) {
    const canvas = document.getElementById("stringArtCanvas");
    if (!canvas || !currentLineSequence) return false;

    // Use the actual canvas size (not the display size)
    const canvasSize = canvas.width;

    // Re-render with new line width
    renderStringsToCanvas(
        canvas,
        currentLineSequence,
        currentPinCount,
        canvasSize,
        lineWidth,
    );

    return true; // Successfully updated in-place
}

// Update canvas output using current slider values
function updateOutputDisplay() {
    if (!currentLineSequence) return;

    // Reset zoom and pan
    zoomLevel = 1;
    panX = 0;
    panY = 0;

    // Calculate available space in the tab content
    const resultTab = document.getElementById("resultTab");
    const availableWidth = Math.max(resultTab.clientWidth - 40, 200); // Account for padding, min 200
    const availableHeight = Math.max(resultTab.clientHeight - 40, 200); // Account for padding, min 200

    // Use the smaller dimension to keep it square and fit within container, capped at 2000px
    const displaySize = Math.min(availableWidth, availableHeight, 2000);

    // Render at 2x for better quality (high DPI displays), minimum 400px
    const canvasSize = Math.max(displaySize * 2, 400);

    const renderLineWidth = parseFloat(
        document.getElementById("renderLineWidth").value,
    );

    // Create canvas element
    const canvas = document.createElement("canvas");
    canvas.id = "stringArtCanvas";
    canvas.width = canvasSize;
    canvas.height = canvasSize;

    // Set display size
    canvas.style.width = `${displaySize}px`;
    canvas.style.height = `${displaySize}px`;

    // Clear result tab content
    resultTab.innerHTML = "";

    // Create zoomable container
    const zoomableContainer = document.createElement("div");
    zoomableContainer.className = "zoomable-container";

    const zoomableContent = document.createElement("div");
    zoomableContent.className = "zoomable-content";

    zoomableContent.appendChild(canvas);

    // Add zoom controls (only if not in webcam mode)
    if (!webcamMode) {
        const zoomControls = document.createElement("div");
        zoomControls.className = "zoom-controls";
        zoomControls.innerHTML = `
            <button class="zoom-btn" onclick="adjustZoom(0.2)">+</button>
            <button class="zoom-btn" onclick="adjustZoom(-0.2)">−</button>
            <button class="zoom-btn" onclick="resetZoom()">Reset</button>
        `;
        zoomableContainer.appendChild(zoomControls);
    }

    // Add zoom info (only if not in webcam mode)
    if (!webcamMode) {
        const zoomInfo = document.createElement("div");
        zoomInfo.className = "zoom-info";
        zoomInfo.id = "zoomInfo";
        zoomInfo.textContent = "Zoom: 100% | Cmd+Scroll to zoom, Drag to pan";
        zoomableContainer.appendChild(zoomInfo);
    }

    // Add FPS indicator (only in webcam mode)
    if (webcamMode) {
        const fpsIndicator = document.createElement("div");
        fpsIndicator.className = "fps-indicator";
        fpsIndicator.id = "fpsIndicator";
        fpsIndicator.textContent = "FPS: --";
        zoomableContainer.appendChild(fpsIndicator);
    }

    zoomableContainer.appendChild(zoomableContent);

    // Add download section with line count and button
    const downloadSection = document.createElement("div");
    downloadSection.className = "download-section";

    // Add line count message
    const lineCountMsg = document.createElement("div");
    lineCountMsg.className = "line-count-msg";
    lineCountMsg.textContent = `${currentLineCount - 1} lines`;
    downloadSection.appendChild(lineCountMsg);

    const downloadBtn = document.createElement("button");
    downloadBtn.className = "download-btn";
    downloadBtn.textContent = "Download as PNG";
    downloadBtn.onclick = () => {
        downloadCanvasAsPNG(canvas);
    };
    downloadSection.appendChild(downloadBtn);

    const downloadPinsBtn = document.createElement("button");
    downloadPinsBtn.className = "download-btn";
    downloadPinsBtn.textContent = "Download Pin List";
    downloadPinsBtn.onclick = () => {
        downloadPinList();
    };
    downloadSection.appendChild(downloadPinsBtn);
    zoomableContainer.appendChild(downloadSection);

    resultTab.appendChild(zoomableContainer);

    // Add event listeners for zoom and pan (only if not in webcam mode)
    if (!webcamMode) {
        setupZoomAndPan(zoomableContainer, zoomableContent);
    }

    // NOW render lines to canvas (after FPS indicator is in the DOM)
    renderStringsToCanvas(
        canvas,
        currentLineSequence,
        currentPinCount,
        canvasSize,
        renderLineWidth,
    );

    // Switch to result tab when canvas is updated
    switchTabProgrammatically("result");
}

// Programmatic tab switching (without event)
function switchTabProgrammatically(tabName) {
    // Remove active class from all tabs and tab contents
    document
        .querySelectorAll(".tab")
        .forEach((tab) => tab.classList.remove("active"));
    document
        .querySelectorAll(".tab-content")
        .forEach((content) => content.classList.remove("active"));

    // Add active class to target tab and corresponding content
    const tabs = document.querySelectorAll(".tab");
    if (tabName === "original") {
        tabs[0].classList.add("active");
    } else if (tabName === "result") {
        tabs[1].classList.add("active");
    }
    document.getElementById(`${tabName}Tab`).classList.add("active");
}

// Toggle sidebar expand/collapse
// biome-ignore lint/correctness/noUnusedVariables: called from html
function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    const toggleIcon = document.getElementById("toggleIcon");

    sidebar.classList.toggle("collapsed");

    if (sidebar.classList.contains("collapsed")) {
        toggleIcon.textContent = "▶";
    } else {
        toggleIcon.textContent = "◀";
    }

    // Update canvas size after sidebar toggle (if we have generated string art)
    const resultTab = document.getElementById('resultTab');
    if (currentLineSequence && resultTab && resultTab.classList.contains('active')) {
        // Wait for CSS transition to complete and browser to reflow
        setTimeout(() => {
            requestAnimationFrame(() => {
                updateOutputDisplay();
            });
        }, 300); // Match the sidebar transition duration from CSS
    }
}

// Setup zoom and pan functionality
function setupZoomAndPan(container, content) {
    // Clean up previous event listeners
    if (zoomPanAbortController) {
        zoomPanAbortController.abort();
    }
    zoomPanAbortController = new AbortController();
    const { signal } = zoomPanAbortController;

    container.classList.add("cursor-grab");

    // Wheel event for zoom (Ctrl+scroll) and pan (normal scroll)
    container.addEventListener(
        "wheel",
        (e) => {
            e.preventDefault();

            if (e.ctrlKey || e.metaKey) {
                // Zoom functionality
                const delta = e.deltaY > 0 ? -0.1 : 0.1;
                const rect = container.getBoundingClientRect();
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;

                adjustZoomAt(delta, centerX, centerY);
            } else {
                // Pan functionality
                const panSpeed = 2;
                panX -= e.deltaX * panSpeed;
                panY -= e.deltaY * panSpeed;
                updateTransform(content);
            }
        },
        { signal },
    );

    // Mouse drag for panning
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let startPanX = 0;
    let startPanY = 0;

    container.addEventListener(
        "mousedown",
        (e) => {
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            startPanX = panX;
            startPanY = panY;
            container.classList.remove("cursor-grab");
            container.classList.add("cursor-grabbing");
        },
        { signal },
    );

    document.addEventListener(
        "mousemove",
        (e) => {
            if (!isDragging) return;

            const deltaX = e.clientX - startX;
            const deltaY = e.clientY - startY;

            panX = startPanX + deltaX;
            panY = startPanY + deltaY;
            updateTransform(content);
        },
        { signal },
    );

    document.addEventListener(
        "mouseup",
        () => {
            if (isDragging) {
                isDragging = false;
                container.classList.remove("cursor-grabbing");
                container.classList.add("cursor-grab");
            }
        },
        { signal },
    );
}

// Adjust zoom level
// biome-ignore lint/correctness/noUnusedVariables: called from html
function adjustZoom(delta) {
    zoomLevel = Math.max(0.1, Math.min(5, zoomLevel + delta));
    const content = document.querySelector(".zoomable-content");
    if (content) {
        updateTransform(content);
    }
}

// Adjust zoom at specific point
function adjustZoomAt(delta, clientX, clientY) {
    const oldZoom = zoomLevel;
    zoomLevel = Math.max(0.1, Math.min(5, zoomLevel + delta));

    // Adjust pan to zoom into the cursor position
    const zoomFactor = zoomLevel / oldZoom;
    const rect = document
        .querySelector(".zoomable-container")
        .getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    panX = (panX - (clientX - centerX)) * zoomFactor + (clientX - centerX);
    panY = (panY - (clientY - centerY)) * zoomFactor + (clientY - centerY);

    const content = document.querySelector(".zoomable-content");
    if (content) {
        updateTransform(content);
    }
}

// Reset zoom and pan
// biome-ignore lint/correctness/noUnusedVariables: called from html
function resetZoom() {
    zoomLevel = 1;
    panX = 0;
    panY = 0;
    const content = document.querySelector(".zoomable-content");
    if (content) {
        updateTransform(content);
    }
}

// Setup zoom and pan for image
function setupImageZoomAndPan(container, _content, canvas, img, targetSize) {
    // Clean up previous event listeners
    if (imgZoomPanAbortController) {
        imgZoomPanAbortController.abort();
    }
    imgZoomPanAbortController = new AbortController();
    const { signal } = imgZoomPanAbortController;

    const ctx = canvas.getContext("2d");

    // Function to redraw image with current zoom and pan
    const redrawImage = () => {
        ctx.save();
        ctx.clearRect(0, 0, targetSize, targetSize);
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, targetSize, targetSize);

        // Apply transformations
        ctx.translate(targetSize / 2, targetSize / 2);
        ctx.translate(imgPanX, imgPanY);
        ctx.scale(imgZoomLevel, imgZoomLevel);
        ctx.translate(-targetSize / 2, -targetSize / 2);

        // Use stored display info for consistent scaling
        if (imageDisplayInfo) {
            const { baseScaledWidth, baseScaledHeight, offsetX, offsetY } =
                imageDisplayInfo;
            ctx.drawImage(img, offsetX, offsetY, baseScaledWidth, baseScaledHeight);
        }
        ctx.restore();

        // Add circular lightbox effect
        addCircularLightboxEffect(ctx, targetSize);

        // Update zoom info
        const zoomInfo = document.getElementById("imgZoomInfo");
        if (zoomInfo) {
            zoomInfo.textContent = `Zoom: ${Math.round(imgZoomLevel * 100)}% | Cmd+Scroll to zoom, Drag to pan`;
        }
    };

    // Wheel event for zoom (Ctrl+scroll) and pan (normal scroll)
    container.addEventListener(
        "wheel",
        (e) => {
            e.preventDefault();

            if (e.ctrlKey || e.metaKey) {
                // Zoom functionality
                const delta = e.deltaY > 0 ? -0.1 : 0.1;
                const rect = container.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;

                adjustImgZoomAt(delta, mouseX, mouseY, redrawImage);
            } else {
                // Pan functionality
                const panSpeed = 2;
                const newPanX = imgPanX - (e.deltaX * panSpeed) / imgZoomLevel;
                const newPanY = imgPanY - (e.deltaY * panSpeed) / imgZoomLevel;

                // Apply pan constraints
                if (imageDisplayInfo) {
                    const { baseScaledWidth, baseScaledHeight } = imageDisplayInfo;
                    const constrained = constrainPan(
                        newPanX,
                        newPanY,
                        targetSize,
                        imgZoomLevel,
                        baseScaledWidth,
                        baseScaledHeight,
                    );
                    imgPanX = constrained.x;
                    imgPanY = constrained.y;
                } else {
                    imgPanX = newPanX;
                    imgPanY = newPanY;
                }

                redrawImage();
            }
        },
        { signal },
    );

    // Mouse drag for panning
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let startPanX = 0;
    let startPanY = 0;

    container.addEventListener(
        "mousedown",
        (e) => {
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            startPanX = imgPanX;
            startPanY = imgPanY;
            container.classList.remove("cursor-grab");
            container.classList.add("cursor-grabbing");
        },
        { signal },
    );

    document.addEventListener(
        "mousemove",
        (e) => {
            if (!isDragging || !container.contains(e.target)) return;

            const deltaX = e.clientX - startX;
            const deltaY = e.clientY - startY;

            const newPanX = startPanX + deltaX / imgZoomLevel;
            const newPanY = startPanY + deltaY / imgZoomLevel;

            // Apply pan constraints
            if (imageDisplayInfo) {
                const { baseScaledWidth, baseScaledHeight } = imageDisplayInfo;
                const constrained = constrainPan(
                    newPanX,
                    newPanY,
                    targetSize,
                    imgZoomLevel,
                    baseScaledWidth,
                    baseScaledHeight,
                );
                imgPanX = constrained.x;
                imgPanY = constrained.y;
            } else {
                imgPanX = newPanX;
                imgPanY = newPanY;
            }

            redrawImage();
        },
        { signal },
    );

    document.addEventListener(
        "mouseup",
        () => {
            if (isDragging) {
                isDragging = false;
                container.classList.remove("cursor-grabbing");
                container.classList.add("cursor-grab");
            }
        },
        { signal },
    );
}

// Adjust image zoom level
// biome-ignore lint/correctness/noUnusedVariables: called from html
function adjustImgZoom(delta) {
    const minZoom = imageDisplayInfo ? imageDisplayInfo.minZoomLevel : 1;
    imgZoomLevel = Math.max(minZoom, Math.min(5, imgZoomLevel + delta));

    // Find the canvas and redraw
    const canvas = document.querySelector(".img-zoomable-content canvas");
    if (canvas) {
        const img = new Image();
        const content = canvas.parentElement;
        img.onload = () => {
            const ctx = canvas.getContext("2d");
            const targetSize = parseInt(canvas.width);

            ctx.save();
            ctx.clearRect(0, 0, targetSize, targetSize);
            ctx.fillStyle = "white";
            ctx.fillRect(0, 0, targetSize, targetSize);

            ctx.translate(targetSize / 2, targetSize / 2);
            ctx.translate(imgPanX, imgPanY);
            ctx.scale(imgZoomLevel, imgZoomLevel);
            ctx.translate(-targetSize / 2, -targetSize / 2);

            // Use stored display info for consistent scaling
            if (imageDisplayInfo) {
                const { baseScaledWidth, baseScaledHeight, offsetX, offsetY } =
                    imageDisplayInfo;
                ctx.drawImage(img, offsetX, offsetY, baseScaledWidth, baseScaledHeight);
            }
            ctx.restore();

            // Add circular lightbox effect
            addCircularLightboxEffect(ctx, targetSize);

            // Update zoom info
            const zoomInfo = document.getElementById("imgZoomInfo");
            if (zoomInfo) {
                zoomInfo.textContent = `Zoom: ${Math.round(imgZoomLevel * 100)}% | Cmd+Scroll to zoom, Drag to pan`;
            }
        };
        img.src = content.dataset.originalImg;
    }
}

// Adjust image zoom at specific point
function adjustImgZoomAt(delta, mouseX, mouseY, redrawCallback) {
    const oldZoom = imgZoomLevel;
    const minZoom = imageDisplayInfo ? imageDisplayInfo.minZoomLevel : 1;
    imgZoomLevel = Math.max(minZoom, Math.min(5, imgZoomLevel + delta));

    // Adjust pan to zoom into the mouse position
    const zoomFactor = imgZoomLevel / oldZoom;
    const canvas = document.querySelector(".img-zoomable-content canvas");
    if (canvas) {
        const targetSize = parseInt(canvas.width);
        const centerX = targetSize / 2;
        const centerY = targetSize / 2;

        const newPanX =
            (imgPanX - (mouseX - centerX) / oldZoom) * zoomFactor +
            (mouseX - centerX) / imgZoomLevel;
        const newPanY =
            (imgPanY - (mouseY - centerY) / oldZoom) * zoomFactor +
            (mouseY - centerY) / imgZoomLevel;

        // Apply pan constraints after zoom adjustment
        if (imageDisplayInfo) {
            const { baseScaledWidth, baseScaledHeight } = imageDisplayInfo;
            const constrained = constrainPan(
                newPanX,
                newPanY,
                targetSize,
                imgZoomLevel,
                baseScaledWidth,
                baseScaledHeight,
            );
            imgPanX = constrained.x;
            imgPanY = constrained.y;
        } else {
            imgPanX = newPanX;
            imgPanY = newPanY;
        }
    }

    if (redrawCallback) {
        redrawCallback();
    }
}

// Reset image zoom and pan
// biome-ignore lint/correctness/noUnusedVariables: called from html
function resetImgZoom() {
    // Reset to minimum zoom level (fills targetSize dimensions)
    imgZoomLevel = imageDisplayInfo ? imageDisplayInfo.minZoomLevel : 1;
    imgPanX = 0;
    imgPanY = 0;

    // Redraw the image
    const canvas = document.querySelector(".img-zoomable-content canvas");
    if (canvas) {
        adjustImgZoom(0); // This will trigger a redraw with current values
    }
}

// Update image canvas when targetSize changes
function updateImageCanvas() {
    if (!originalImage) return;

    const newTargetSize =
        parseInt(document.getElementById("targetSize").value) || 1000;

    // Update stored display info
    if (imageDisplayInfo) {
        const baseScale = Math.min(
            newTargetSize / originalImage.width,
            newTargetSize / originalImage.height,
        );
        const baseScaledWidth = originalImage.width * baseScale;
        const baseScaledHeight = originalImage.height * baseScale;

        // Recalculate minimum zoom for new target size
        const minZoomToFillWidth = newTargetSize / baseScaledWidth;
        const minZoomToFillHeight = newTargetSize / baseScaledHeight;
        const minZoomLevel = Math.max(minZoomToFillWidth, minZoomToFillHeight);

        const offsetX = (newTargetSize - baseScaledWidth) / 2;
        const offsetY = (newTargetSize - baseScaledHeight) / 2;

        imageDisplayInfo = {
            targetSize: newTargetSize,
            baseScale,
            baseScaledWidth,
            baseScaledHeight,
            offsetX,
            offsetY,
            minZoomLevel,
        };

        // Ensure current zoom level meets new minimum
        if (imgZoomLevel < minZoomLevel) {
            imgZoomLevel = minZoomLevel;
        }
    }

    // Find the current image container and update it
    const container = document.querySelector(".img-zoomable-container");
    const canvas = document.querySelector(".img-zoomable-content canvas");

    if (container && canvas) {
        // Update container size
        container.style.width = `${newTargetSize}px`;
        container.style.height = `${newTargetSize}px`;

        // Update canvas size
        canvas.width = newTargetSize;
        canvas.height = newTargetSize;
        canvas.style.width = `${newTargetSize}px`;
        canvas.style.height = `${newTargetSize}px`;

        // Redraw the image with new size
        const ctx = canvas.getContext("2d");
        ctx.save();
        ctx.clearRect(0, 0, newTargetSize, newTargetSize);
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, newTargetSize, newTargetSize);

        // Apply current transformations
        ctx.translate(newTargetSize / 2, newTargetSize / 2);
        ctx.translate(imgPanX, imgPanY);
        ctx.scale(imgZoomLevel, imgZoomLevel);
        ctx.translate(-newTargetSize / 2, -newTargetSize / 2);

        // Draw image with new scaling
        if (imageDisplayInfo) {
            const { baseScaledWidth, baseScaledHeight, offsetX, offsetY } =
                imageDisplayInfo;
            ctx.drawImage(
                originalImage,
                offsetX,
                offsetY,
                baseScaledWidth,
                baseScaledHeight,
            );
        }
        ctx.restore();

        // Add circular lightbox effect
        addCircularLightboxEffect(ctx, newTargetSize);

        // Update zoom info
        const zoomInfo = document.getElementById("imgZoomInfo");
        if (zoomInfo) {
            zoomInfo.textContent = `Zoom: ${Math.round(imgZoomLevel * 100)}% | Cmd+Scroll to zoom, Drag to pan`;
        }
    }
}

// Calculate pan limits to ensure image always covers the canvas
function calculatePanLimits(
    targetSize,
    zoomLevel,
    baseScaledWidth,
    baseScaledHeight,
) {
    // Calculate the actual displayed size of the image at current zoom
    const displayedWidth = baseScaledWidth * zoomLevel;
    const displayedHeight = baseScaledHeight * zoomLevel;

    // Calculate how much the image can move while still covering the canvas
    const maxPanX = Math.max(0, (displayedWidth - targetSize) / 2);
    const maxPanY = Math.max(0, (displayedHeight - targetSize) / 2);

    return {
        minX: -maxPanX,
        maxX: maxPanX,
        minY: -maxPanY,
        maxY: maxPanY,
    };
}

// Constrain pan values to ensure full canvas coverage
function constrainPan(
    panX,
    panY,
    targetSize,
    zoomLevel,
    baseScaledWidth,
    baseScaledHeight,
) {
    const limits = calculatePanLimits(
        targetSize,
        zoomLevel,
        baseScaledWidth,
        baseScaledHeight,
    );

    return {
        x: Math.max(limits.minX, Math.min(limits.maxX, panX)),
        y: Math.max(limits.minY, Math.min(limits.maxY, panY)),
    };
}

// Add circular lightbox effect to highlight the string art generation area
function addCircularLightboxEffect(ctx, size) {
    // Save the current context state
    ctx.save();

    // Create a circular clipping path for the lightbox effect
    const centerX = size / 2;
    const centerY = size / 2;
    const radius = size / 2 - 1; // Slightly smaller than half to match string art circle

    // Create overlay that covers everything
    ctx.fillStyle = "rgba(0, 0, 0, 0.3)"; // Semi-transparent dark overlay
    ctx.fillRect(0, 0, size, size);

    // Use composite operation to "cut out" the circle
    ctx.globalCompositeOperation = "destination-out";
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.fill();

    // Restore the context state
    ctx.restore();
}

// Get cropped image data based on current pan/zoom state (for WebGPU)
function getCroppedImageDataForWebGPU() {
    if (!originalImage || !imageDisplayInfo) {
        throw new Error("No original image or display info available");
    }

    const { targetSize, baseScaledWidth, baseScaledHeight, offsetX, offsetY } =
        imageDisplayInfo;

    // Create a temporary canvas to extract the visible portion
    const tempCanvas = document.createElement("canvas");
    const tempCtx = tempCanvas.getContext("2d");

    // Set canvas size to target size (square)
    tempCanvas.width = targetSize;
    tempCanvas.height = targetSize;

    // Fill with white background
    tempCtx.fillStyle = "white";
    tempCtx.fillRect(0, 0, targetSize, targetSize);

    // Apply the same transformations as the display
    tempCtx.save();
    tempCtx.translate(targetSize / 2, targetSize / 2);
    tempCtx.translate(imgPanX, imgPanY);
    tempCtx.scale(imgZoomLevel, imgZoomLevel);
    tempCtx.translate(-targetSize / 2, -targetSize / 2);

    // Draw the image with the same scaling and positioning as display
    tempCtx.drawImage(
        originalImage,
        offsetX,
        offsetY,
        baseScaledWidth,
        baseScaledHeight,
    );
    tempCtx.restore();

    // Get and return the image data from the canvas
    const imageData = tempCtx.getImageData(0, 0, targetSize, targetSize);
    return imageData;
}

// Update CSS transform
function updateTransform(content) {
    if (content) {
        const canvas = content.querySelector("#stringArtCanvas");
        if (canvas) {
            // Apply transform directly to canvas
            canvas.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomLevel})`;
            canvas.style.transformOrigin = "center center";
        }

        // Update zoom info
        const zoomInfo = document.getElementById("zoomInfo");
        if (zoomInfo) {
            zoomInfo.textContent = `Zoom: ${Math.round(zoomLevel * 100)}% | Cmd+Scroll to zoom, Drag to pan`;
        }
    }
}

// Download pin list as text file
function downloadPinList() {
    if (!currentLineSequence || currentLineSequence.length === 0) {
        alert("No pin data available to download");
        return;
    }

    // Generate the output filename
    const fileInput = document.getElementById("imageInput");
    let baseName = "stringart";

    if (fileInput.files && fileInput.files.length > 0) {
        const parts = fileInput.files[0].name.split(".");
        baseName = parts.slice(0, -1).join(".");
    } else if (lastWebcamFrame) {
        baseName = "webcam-capture";
    }

    const outname = `${baseName}-pins.txt`;

    // Create text content with header and one pin number per line
    const pinListContent = `# ${currentPinCount} pins\n${currentLineSequence.join("\n")}`;

    // Create and download the text file
    const blob = new Blob([pinListContent], { type: "text/plain" });
    const link = document.createElement("a");
    const blobUrl = URL.createObjectURL(blob);
    link.download = outname;
    link.href = blobUrl;
    link.click();

    // Revoke the blob URL after a short delay to prevent memory leak
    setTimeout(() => URL.revokeObjectURL(blobUrl), 100);
}

// Download canvas as PNG
function downloadCanvasAsPNG(canvas) {
    // Generate the output filename
    const fileInput = document.getElementById("imageInput");
    let baseName = "stringart";

    if (fileInput.files && fileInput.files.length > 0) {
        const parts = fileInput.files[0].name.split(".");
        baseName = parts.slice(0, -1).join(".");
    } else if (lastWebcamFrame) {
        baseName = "webcam-capture";
    }

    const outname = `${baseName}-strings.png`;

    // Convert canvas to blob and download
    canvas.toBlob((blob) => {
        const link = document.createElement("a");
        const blobUrl = URL.createObjectURL(blob);
        link.download = outname;
        link.href = blobUrl;
        link.click();

        // Revoke the blob URL after a short delay to prevent memory leak
        setTimeout(() => URL.revokeObjectURL(blobUrl), 100);
    });
}

// Update FPS counter
function updateFPS() {
    const currentTime = performance.now();
    const deltaTime = currentTime - lastFrameTime;

    frameCount++;

    // Update FPS display
    const fpsIndicator = document.getElementById("fpsIndicator");
    if (!fpsIndicator) return;

    // Update FPS every second
    if (deltaTime >= 1000) {
        fps = Math.round((frameCount * 1000) / deltaTime);
        fpsIndicator.textContent = `FPS: ${fps}`;
        frameCount = 0;
        lastFrameTime = currentTime;
    } else {
        // Show tentative FPS even before 1 second passes
        const tentativeFps = Math.round((frameCount * 1000) / deltaTime);
        fpsIndicator.textContent = `FPS: ${tentativeFps}`;
    }
}

// Render string art lines to canvas
function renderStringsToCanvas(
    canvas,
    lineSequence,
    pinCount,
    canvasSize,
    lineWidth,
) {
    // Validate canvas size before rendering
    if (canvasSize < 100 || !Number.isFinite(canvasSize)) {
        console.warn(`Invalid canvas size: ${canvasSize}, skipping render`);
        return;
    }

    const ctx = canvas.getContext("2d");

    // Clear canvas
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvasSize, canvasSize);

    // Calculate pin coordinates
    const pinCoords = calculatePinCoordinates(pinCount, canvasSize);

    // Draw circle border
    const centerOut = canvasSize / 2;
    const radiusOut = canvasSize / 2 - 1;
    ctx.strokeStyle = "black";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerOut, centerOut, radiusOut, 0, 2 * Math.PI);
    ctx.stroke();

    // Draw pins
    ctx.fillStyle = "black";
    for (let i = 0; i < pinCoords.length; i++) {
        ctx.beginPath();
        ctx.arc(pinCoords[i].x, pinCoords[i].y, 2, 0, 2 * Math.PI);
        ctx.fill();
    }

    // Draw lines with alpha blending for smooth, natural string art effect
    // Line width parameter controls opacity (higher = darker)
    // This creates cumulative darkening where lines overlap
    // Scale opacity based on canvas size to maintain consistent appearance
    const baseAlpha = lineWidth / 100;  // Convert slider value (3-60) to base opacity (0.03-0.6)
    const scaleFactor = canvasSize / 500;  // Scale relative to reference size (500px from webgpu)
    const alpha = baseAlpha * scaleFactor;  // Increase opacity proportionally with canvas size
    ctx.strokeStyle = `rgba(0, 0, 0, ${Math.min(alpha, 1.0)})`;  // Clamp to max 1.0
    ctx.lineWidth = 1;

    for (let i = 0; i < lineSequence.length - 1; i++) {
        const fromPin = lineSequence[i];
        const toPin = lineSequence[i + 1];

        if (
            fromPin < 0 ||
            fromPin >= pinCoords.length ||
            toPin < 0 ||
            toPin >= pinCoords.length
        ) {
            continue; // Skip invalid pin indices
        }

        ctx.beginPath();
        ctx.moveTo(pinCoords[fromPin].x, pinCoords[fromPin].y);
        ctx.lineTo(pinCoords[toPin].x, pinCoords[toPin].y);
        ctx.stroke();
    }
}

// ============= WEBCAM MODE ============= //

let webcamMode = false;
let webcamStream = null;
let webcamProcessingLoop = null;
let webcamProcessing = false;
let lastWebcamFrame = null; // Store last frame as Image object
let webcamInitialCalibrationDone = false; // Track if we've done initial auto line weight

// biome-ignore lint/correctness/noUnusedVariables: called from html
function toggleInputMode() {
    const modeToggle = document.getElementById("modeToggle");
    const imageInputGroup = document.getElementById("imageInputGroup");
    const webcamGroup = document.getElementById("webcamGroup");

    // Update mode based on checkbox state
    webcamMode = modeToggle.checked;

    if (webcamMode) {
        imageInputGroup.style.display = "none";
        webcamGroup.style.display = "block";

        // Update parameters for webcam mode
        const targetSizeInput = document.getElementById("targetSize");
        const targetSizeValue = document.getElementById("targetSizeValue");
        const maxLinesInput = document.getElementById("maxLines");
        const maxLinesValue = document.getElementById("maxLinesValue");

        targetSizeInput.value = 200;
        targetSizeValue.textContent = 200;
        maxLinesInput.value = 4000;
        maxLinesValue.textContent = 4000;

        // Reset calibration flag
        webcamInitialCalibrationDone = false;

        // Automatically start webcam
        startWebcam();
    } else {
        imageInputGroup.style.display = "block";
        webcamGroup.style.display = "none";

        // Stop webcam if running
        if (webcamStream) {
            stopWebcam();
        }

        // Re-enable pan & zoom controls and hide FPS when switching to static mode
        // If there's already generated output, update it to show controls
        if (currentLineSequence) {
            updateOutputDisplay();
        }
    }
}

// biome-ignore lint/correctness/noUnusedVariables: called from html
async function startWebcam() {
    try {
        // Ensure auto line weight checkbox is unchecked
        const autoLineWeightCheckbox = document.getElementById("autoLineWeight");
        const lineWeightSlider = document.getElementById("lineWeight");
        const lineWeightValue = document.getElementById("lineWeightValue");

        if (autoLineWeightCheckbox.checked) {
            autoLineWeightCheckbox.checked = false;
            lineWeightSlider.disabled = false;
            lineWeightValue.textContent = lineWeightSlider.value;
            lineWeightSlider.classList.remove("opacity-50");
            lineWeightSlider.classList.add("opacity-100");
        }

        const video = document.getElementById("webcamVideo");
        const startBtn = document.getElementById("startWebcamBtn");
        const stopBtn = document.getElementById("stopWebcamBtn");

        showStatus("Starting webcam...", "loading");

        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
            },
        });

        video.srcObject = webcamStream;

        // Show canvas instead of video (canvas will display contrast-adjusted frames)
        const canvas = document.getElementById("webcamCanvas");
        canvas.style.display = "block";

        startBtn.style.display = "none";
        stopBtn.style.display = "block";

        showStatus("Webcam started!", "success");
        setTimeout(() => hideStatus(), 2000);

        // Start processing loop
        startWebcamProcessing();
    } catch (error) {
        showStatus(`Error starting webcam: ${error.message}`, "error");
        console.error("Webcam error:", error);
    }
}

// biome-ignore lint/correctness/noUnusedVariables: called from html
function stopWebcam() {
    const video = document.getElementById("webcamVideo");
    const startBtn = document.getElementById("startWebcamBtn");
    const stopBtn = document.getElementById("stopWebcamBtn");
    const canvas = document.getElementById("webcamCanvas");

    // Capture last frame before stopping
    if (video.videoWidth && video.videoHeight) {
        const targetSize = parseInt(document.getElementById("targetSize").value);
        canvas.width = targetSize;
        canvas.height = targetSize;

        const ctx = canvas.getContext("2d");
        const size = Math.min(video.videoWidth, video.videoHeight);
        const sx = (video.videoWidth - size) / 2;
        const sy = (video.videoHeight - size) / 2;

        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, targetSize, targetSize);
        ctx.drawImage(video, sx, sy, size, size, 0, 0, targetSize, targetSize);

        // Convert canvas to image and save
        const img = new Image();
        img.onload = () => {
            lastWebcamFrame = img;
            originalImage = img; // Set as original image for reprocessing
            hasInitialImage = true;
            displayImageWithZoom(img);
            showStatus("Last webcam frame saved", "success");
            setTimeout(() => hideStatus(), 2000);
            // Clear the onload handler to allow GC
            img.onload = null;
        };
        img.onerror = () => {
            console.warn("Failed to capture last webcam frame");
            // Clear the error handler to allow GC
            img.onerror = null;
        };
        img.src = canvas.toDataURL();
    }

    if (webcamStream) {
        webcamStream.getTracks().forEach((track) => track.stop());
        webcamStream = null;
    }

    if (webcamProcessingLoop) {
        cancelAnimationFrame(webcamProcessingLoop);
        webcamProcessingLoop = null;
    }

    canvas.style.display = "none";
    startBtn.style.display = "block";
    stopBtn.style.display = "none";
    webcamProcessing = false;
}

function startWebcamProcessing() {
    const processFrame = async () => {
        if (!webcamStream || !webcamMode) {
            return;
        }

        // Only process if not already processing
        if (!webcamProcessing) {
            webcamProcessing = true;

            try {
                await processWebcamFrame();
                // Update FPS after entire frame processing (WASM + rendering)
                updateFPS();
            } catch (error) {
                console.error("Frame processing error:", error);
                showStatus(`Processing error: ${error.message}`, "error");
            } finally {
                webcamProcessing = false;
            }
        }

        // Schedule next frame
        webcamProcessingLoop = requestAnimationFrame(processFrame);
    };

    processFrame();
}

async function processWebcamFrame() {
    const video = document.getElementById("webcamVideo");
    const canvas = document.getElementById("webcamCanvas");

    if (!video.videoWidth || !video.videoHeight) {
        return; // Video not ready
    }

    // Check if we need to do initial calibration
    if (!webcamInitialCalibrationDone && lastWebcamFrame) {
        webcamInitialCalibrationDone = true;

        // Pause the processing loop
        if (webcamProcessingLoop) {
            cancelAnimationFrame(webcamProcessingLoop);
            webcamProcessingLoop = null;
        }

        // Run auto line weight calibration
        await generateWithAutoLineWeight();

        // Resume processing loop
        if (webcamStream && webcamMode) {
            startWebcamProcessing();
        }

        return;
    }

    // Get target size
    const targetSize = parseInt(document.getElementById("targetSize").value);

    // Set canvas to target size
    canvas.width = targetSize;
    canvas.height = targetSize;

    const ctx = canvas.getContext("2d");

    // Calculate crop to fit square
    const size = Math.min(video.videoWidth, video.videoHeight);
    const sx = (video.videoWidth - size) / 2;
    const sy = (video.videoHeight - size) / 2;

    // Draw cropped and scaled frame
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, targetSize, targetSize);
    ctx.drawImage(video, sx, sy, size, size, 0, 0, targetSize, targetSize);

    // Get image data
    const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
    const data = imageData.data;

    // Apply contrast adjustment
    const contrastValue = parseInt(
        document.getElementById("webcamContrast").value,
    );
    const contrastFactor = contrastValue / 100;

    for (let i = 0; i < data.length; i += 4) {
        // Apply contrast to R, G, B channels (skip alpha channel at i+3)
        data[i] = Math.max(
            0,
            Math.min(255, (data[i] - 128) * contrastFactor + 128),
        ); // R
        data[i + 1] = Math.max(
            0,
            Math.min(255, (data[i + 1] - 128) * contrastFactor + 128),
        ); // G
        data[i + 2] = Math.max(
            0,
            Math.min(255, (data[i + 2] - 128) * contrastFactor + 128),
        ); // B
    }

    // Put adjusted image data back to canvas
    ctx.putImageData(imageData, 0, 0);

    // Save current frame for later reprocessing (non-blocking)
    const dataURL = canvas.toDataURL();
    if (!lastWebcamFrame || lastWebcamFrame.src !== dataURL) {
        const img = new Image();
        img.onload = () => {
            lastWebcamFrame = img;
            originalImage = img;
            hasInitialImage = true;
            // Clear the onload handler to allow GC
            img.onload = null;
        };
        img.onerror = () => {
            console.warn("Failed to load webcam frame");
            // Clear the error handler to allow GC
            img.onerror = null;
        };
        img.src = dataURL;
    }

    // Get the adjusted image data for WASM processing
    const adjustedImageData = ctx.getImageData(0, 0, targetSize, targetSize);
    const adjustedData = adjustedImageData.data;

    // Get parameters
    const pins = parseInt(document.getElementById("pins").value);
    const maxLines = parseInt(document.getElementById("maxLines").value);
    const lineWeight = parseInt(document.getElementById("lineWeight").value);
    const minDistance = parseInt(document.getElementById("minDistance").value);

    // Process frame with WebGPU
    const config = {
        imgSize: targetSize,
        pins: pins,
        minDistance: minDistance,
        maxLines: maxLines,
        lineWeight: lineWeight,
        iterationsPerDispatch: 100,
    };

    try {
        const result = await WebGPUProcessor.processImage(
            adjustedImageData,
            config,
        );

        if (result && result.lineCount > 0) {
            currentLineSequence = result.lineSequence;
            currentPinCount = pins;
            currentLineCount = result.lineCount;
            updateOutputDisplay();
        }
    } catch (error) {
        console.error("WebGPU processing error:", error);
    }
}
