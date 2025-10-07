var StringArtModule = null;
let stringArtModule = null;
let isProcessing = false;
let runCount = 0;
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

// Store original image and display info for cropping
let originalImage = null;
let imageDisplayInfo = null;

// Initialize range input displays
document.addEventListener("DOMContentLoaded", () => {
    const ranges = [
        "pins",
        "maxLines",
        "targetSize",
        "outputSize",
        "lineWeight",
        "minDistance",
    ];
    ranges.forEach((id) => {
        const input = document.getElementById(id);
        const display = document.getElementById(`${id}Value`);

        input.addEventListener("input", () => {
            display.textContent = input.value;

            // Update SVG dynamically for output parameters if we have generated data
            if (currentLineSequence && id === "outputSize") {
                const outputSize = parseInt(
                    document.getElementById("outputSize").value,
                );

                // Try efficient update first, fallback to full re-render if needed
                if (!updateSVGParameters(outputSize)) {
                    updateSVGOutput();
                }
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

    autoLineWeightCheckbox.addEventListener("change", () => {
        if (autoLineWeightCheckbox.checked) {
            lineWeightSlider.disabled = true;
            lineWeightValue.textContent = "Auto";
            lineWeightSlider.style.opacity = "0.5";
            // Only trigger auto-regeneration when enabling auto mode
            scheduleAutoRegeneration();
        } else {
            lineWeightSlider.disabled = false;
            lineWeightValue.textContent = lineWeightSlider.value;
            lineWeightSlider.style.opacity = "1";
            // Don't regenerate when disabling auto mode
        }
    });

    // Load the WASM module
    loadWasmModule();
});

async function loadWasmModule() {
    try {
        showStatus("Loading WASM module...", "loading");

        // Wait for StringArtModule to be available
        let attempts = 0;
        while (typeof StringArtModule === "undefined" && attempts < 50) {
            await new Promise((resolve) => setTimeout(resolve, 100));
            attempts++;
        }

        if (typeof StringArtModule !== "undefined") {
            console.log("StringArtModule found, initializing...");
            stringArtModule = await StringArtModule();

            showStatus("WASM module loaded successfully!", "success");
            setTimeout(() => hideStatus(), 2000);
        } else {
            throw new Error(
                "StringArtModule not found. Make sure stringart.wasm.js is loaded.",
            );
        }
    } catch (error) {
        showStatus(`Error loading WASM module: ${error.message}`, "error");
        console.error("WASM loading error:", error);
    }
}

// Tab switching function
// biome-ignore lint/correctness/noUnusedVariables: called from html
function switchTab(tabName) {
    // Remove active class from all tabs and tab contents
    document
        .querySelectorAll(".tab")
        .forEach((tab) => tab.classList.remove("active"));
    document
        .querySelectorAll(".tab-content")
        .forEach((content) => content.classList.remove("active"));

    // Add active class to clicked tab and corresponding content
    event.target.classList.add("active");
    document.getElementById(`${tabName}Tab`).classList.add("active");
}

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
    zoomableContainer.style.position = "relative";
    zoomableContainer.style.width = targetSize + "px";
    zoomableContainer.style.height = targetSize + "px";
    zoomableContainer.style.margin = "20px auto";
    zoomableContainer.style.overflow = "hidden";
    zoomableContainer.style.cursor = "grab";
    zoomableContainer.style.border = "1px solid #ddd";
    zoomableContainer.style.borderRadius = "8px";

    // Create content wrapper
    const zoomableContent = document.createElement("div");
    zoomableContent.className = "img-zoomable-content";
    zoomableContent.style.position = "absolute";
    zoomableContent.style.transformOrigin = "center center";

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

    // Only regenerate if the SVG tab is active
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
    if (!fileInput.files[0]) {
        alert("Please select an image file first!");
        return;
    }

    if (!stringArtModule || !stringArtModule._malloc || !stringArtModule.ccall) {
        showStatus("WASM module not ready. Please wait and try again.", "error");
        return;
    }

    // Ensure HEAPU8 is ready
    if (!stringArtModule.HEAPU8) {
        try {
            const testPtr = stringArtModule._malloc(1);
            if (testPtr !== 0) {
                stringArtModule._free(testPtr);
            }
        } catch (e) {
            console.log("Error during memory test:", e);
        }

        if (!stringArtModule.HEAPU8) {
            showStatus("WASM memory not ready. Please try again.", "error");
            return;
        }
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
        "pins", "maxLines", "targetSize", "minDistance", "outputSize",
        "lineWeight", "autoLineWeight", "autoRegenerate", "imageInput"
    ];
    controls.forEach(id => {
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
        const file = fileInput.files[0];
        const imageData = await loadImageData(file);

        // Binary search parameters
        // Lower weight = more lines, Higher weight = fewer lines
        // We want to find the minimum weight that produces <= targetMaxLines
        let low = 1;
        let high = 200;
        let bestWeight = 20;
        let iteration = 0;
        const maxIterations = 10;

        console.log(`Starting binary search for target ${targetMaxLines} lines (lower weight = more lines)`);

        while (low <= high && iteration < maxIterations && !isAutoCancelled) {
            const mid = Math.floor((low + high) / 2);
            iteration++;

            showStatus(`Testing weight ${mid} (iteration ${iteration})...`, "loading");
            showProgress((iteration / maxIterations) * 90);

            // Initialize with current weight
            const initResult = stringArtModule.ccall(
                "initStringArt",
                "number",
                ["number", "number", "number", "number", "number", "number"],
                [pins, targetMaxLines, targetSize, mid, mid, minDistance],
            );

            if (initResult < 1) {
                throw new Error("Failed to initialize string art generator");
            }

            // Process image with current weight
            await new Promise(resolve => setTimeout(resolve, 50)); // Small delay for UI update

            const lineCount = stringArtModule.ccall(
                "processImage",
                "number",
                ["number", "number", "number", "number"],
                [imageData.dataPtr, imageData.width, imageData.height, imageData.channels],
            );

            console.log(`Weight ${mid}: ${lineCount} lines (target: ${targetMaxLines})`);

            // Update the display with current result
            if (lineCount > 0) {
                currentLineSequence = null; // Clear to force update
                const lineSequencePtr = stringArtModule.ccall("getLineSequence", "number", [], []);
                const totalLineCount = stringArtModule.ccall("getLineCount", "number", [], []);

                if (lineSequencePtr !== 0 && totalLineCount > 0) {
                    const lineSequence = new Int32Array(
                        stringArtModule.HEAPU8.buffer,
                        lineSequencePtr,
                        totalLineCount,
                    );
                    currentLineSequence = Array.from(lineSequence);
                    currentPinCount = pins;
                    currentLineCount = totalLineCount;
                    updateSVGOutput();
                }
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

        const initResult = stringArtModule.ccall(
            "initStringArt",
            "number",
            ["number", "number", "number", "number", "number", "number"],
            [pins, targetMaxLines, targetSize, bestWeight, bestWeight, minDistance],
        );

        if (initResult < 1) {
            throw new Error("Failed to initialize with optimal weight");
        }

        const finalLineCount = stringArtModule.ccall(
            "processImage",
            "number",
            ["number", "number", "number", "number"],
            [imageData.dataPtr, imageData.width, imageData.height, imageData.channels],
        );

        // Get final result
        const lineSequencePtr = stringArtModule.ccall("getLineSequence", "number", [], []);
        const totalLineCount = stringArtModule.ccall("getLineCount", "number", [], []);

        if (lineSequencePtr !== 0 && totalLineCount > 0) {
            const lineSequence = new Int32Array(
                stringArtModule.HEAPU8.buffer,
                lineSequencePtr,
                totalLineCount,
            );
            currentLineSequence = Array.from(lineSequence);
            currentPinCount = pins;
            currentLineCount = totalLineCount;
            updateSVGOutput();
        }

        // Update the line weight slider to show the found value
        const lineWeightSlider = document.getElementById("lineWeight");
        const lineWeightValue = document.getElementById("lineWeightValue");
        lineWeightSlider.value = bestWeight;
        lineWeightValue.textContent = `Auto (${bestWeight})`;

        showProgress(100);
        showStatus(
            `Found minimum weight: ${bestWeight} for ${finalLineCount} lines (target: ≤${targetMaxLines})`,
            "success"
        );

        // Cleanup
        stringArtModule._free(imageData.dataPtr);

        setTimeout(() => hideStatus(), 3000);
    } catch (error) {
        if (!isAutoCancelled) {
            showStatus(`Error during auto weight search: ${error.message}`, "error");
        }
        console.error("Auto weight search error:", error);

        try {
            stringArtModule.ccall("cleanup", null, [], []);
        } catch (e) {
            console.warn("Emergency cleanup failed:", e);
        }
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
            "pins", "maxLines", "targetSize", "minDistance", "outputSize",
            "autoRegenerate", "imageInput", "autoLineWeight"
        ];
        controls.forEach(id => {
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

    console.log("Checking for file input...");
    const fileInput = document.getElementById("imageInput");
    if (!fileInput.files[0]) {
        console.log("No file selected");
        alert("Please select an image file first!");
        return;
    }
    console.log("File selected:", fileInput.files[0].name);

    // Check if auto line weight is enabled
    const autoLineWeightCheckbox = document.getElementById("autoLineWeight");
    if (autoLineWeightCheckbox.checked) {
        console.log("Auto line weight enabled, starting binary search...");
        await generateWithAutoLineWeight();
        return;
    }

    if (!stringArtModule || !stringArtModule._malloc || !stringArtModule.ccall) {
        showStatus("WASM module not ready. Please wait and try again.", "error");
        console.error("WASM module readiness check failed");
        return;
    }

    // Check if HEAPU8 is available, if not, try to create it
    if (!stringArtModule.HEAPU8) {
        console.log("HEAPU8 not ready, trying to initialize memory...");
        showStatus("Initializing WASM memory...", "loading");

        // Try to find memory and create HEAPU8
        let memory = null;

        // Look for memory in various possible locations
        if (stringArtModule.wasmMemory) {
            memory = stringArtModule.wasmMemory;
        } else if (stringArtModule.memory) {
            memory = stringArtModule.memory;
        } else {
            // Try to access memory through WebAssembly exports
            console.log("Attempting to access memory through other means...");
            // Sometimes we need to allocate some memory first to trigger initialization
            try {
                const testPtr = stringArtModule._malloc(1);
                if (testPtr !== 0) {
                    stringArtModule._free(testPtr);
                }
                // Check again after malloc
                if (stringArtModule.HEAPU8) {
                    console.log("HEAPU8 became available after malloc");
                }
            } catch (e) {
                console.log("Error during memory test:", e);
            }
        }

        if (memory?.buffer && !stringArtModule.HEAPU8) {
            console.log("Creating HEAPU8 view manually...");
            stringArtModule.HEAPU8 = new Uint8Array(memory.buffer);
        }

        if (!stringArtModule.HEAPU8) {
            showStatus("WASM memory not ready. Please try again.", "error");
            console.error("HEAPU8 still not available after all attempts");
            console.log(
                "Available memory-related properties:",
                Object.keys(stringArtModule).filter(
                    (k) => k.toLowerCase().includes("mem") || k.includes("HEAP"),
                ),
            );
            return;
        }
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

        // Initialize the generator
        console.log("Initializing string art generator...");
        const initResult = stringArtModule.ccall(
            "initStringArt",
            "number",
            ["number", "number", "number", "number", "number", "number"],
            [pins, maxLines, targetSize, lineWeight, lineWeight, minDistance],
        );
        console.log("Init result:", initResult);

        if (initResult === 2) {
            console.log("Full reinitialization performed (pin/line cache rebuilt)");
        } else if (initResult === 1) {
            console.log("Cache reused (pin/line data preserved)");
        }

        if (initResult < 1) {
            throw new Error("Failed to initialize string art generator");
        }

        showProgress(20);

        // Load and process image
        const file = fileInput.files[0];
        console.log("Loading image data...");
        const imageData = await loadImageData(file);
        console.log("Image data loaded:", imageData);

        showProgress(40);
        showStatus("Calculating string art lines...", "loading");

        // Process the image (this is the heavy computation)
        console.log("Starting image processing...");
        console.log("Parameters:", {
            dataPtr: imageData.dataPtr,
            width: imageData.width,
            height: imageData.height,
            channels: imageData.channels,
        });

        // Check memory usage before processing
        const memUsedBefore = stringArtModule.wasmMemory.buffer.byteLength;
        console.log(
            "WASM memory usage before processing:",
            (memUsedBefore / 1024 / 1024).toFixed(2),
            "MB",
        );
        let memUsedAfter = memUsedBefore; // Initialize to avoid scope issues

        // Use setTimeout to yield control back to browser for UI updates
        await new Promise((resolve) => setTimeout(resolve, 100));

        // Add timeout to catch hanging
        const startTime = Date.now();
        let lineCount;

        try {
            // Set a timeout for the processing
            const processPromise = new Promise((resolve, reject) => {
                try {
                    const result = stringArtModule.ccall(
                        "processImage",
                        "number",
                        ["number", "number", "number", "number"],
                        [
                            imageData.dataPtr,
                            imageData.width,
                            imageData.height,
                            imageData.channels,
                        ],
                    );
                    resolve(result);
                } catch (error) {
                    reject(error);
                }
            });

            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(
                    () => reject(new Error("Processing timeout after 10 seconds")),
                    10000,
                );
            });

            lineCount = await Promise.race([processPromise, timeoutPromise]);

            const endTime = Date.now();
            console.log("Processing completed. Line count:", lineCount);
            console.log("Processing time:", (endTime - startTime) / 1000, "seconds");

            // Check memory usage after processing
            memUsedAfter = stringArtModule.wasmMemory.buffer.byteLength;
            console.log(
                "WASM memory usage after processing:",
                (memUsedAfter / 1024 / 1024).toFixed(2),
                "MB",
            );
            console.log(
                "Memory increase during processing:",
                ((memUsedAfter - memUsedBefore) / 1024 / 1024).toFixed(2),
                "MB",
            );
        } catch (error) {
            console.error("Processing error:", error);
            throw error;
        }

        showProgress(80);
        showStatus("Generating SVG output...", "loading");

        // Get the line sequence instead of rendered image
        const lineSequencePtr = stringArtModule.ccall(
            "getLineSequence",
            "number",
            [],
            [],
        );
        const totalLineCount = stringArtModule.ccall(
            "getLineCount",
            "number",
            [],
            [],
        );

        if (lineSequencePtr === 0 || totalLineCount === 0) {
            throw new Error("Failed to generate line sequence");
        }

        // Extract line sequence from WASM memory
        const lineSequence = new Int32Array(
            stringArtModule.HEAPU8.buffer,
            lineSequencePtr,
            totalLineCount,
        );

        // Store data globally for dynamic updates
        currentLineSequence = Array.from(lineSequence); // Copy to regular array
        currentPinCount = pins;
        currentLineCount = totalLineCount;

        // Create and display initial SVG
        updateSVGOutput();

        showProgress(100);
        showStatus(
            `String art generated successfully! Used ${currentLineCount} lines.`,
            "success",
        );

        // Cleanup
        try {
            // Force garbage collection if available
            if (window.gc) {
                window.gc();
            }
        } catch (e) {
            console.warn("Cleanup failed:", e);
        }

        try {
            stringArtModule._free(imageData.dataPtr);
        } catch (e) {
            console.warn("Free imageData failed:", e);
        }

        // Check memory usage after cleanup
        const memUsedAfterCleanup = stringArtModule.wasmMemory.buffer.byteLength;
        console.log(
            "WASM memory usage after cleanup:",
            (memUsedAfterCleanup / 1024 / 1024).toFixed(2),
            "MB",
        );
        console.log(
            "Memory freed by cleanup:",
            ((memUsedAfter - memUsedAfterCleanup) / 1024 / 1024).toFixed(2),
            "MB",
        );

        // Increment run count and reload module if memory is getting high
        runCount++;
        console.log("Run count:", runCount);

        if (memUsedAfterCleanup > 600 * 1024 * 1024) {
            // 600MB threshold
            console.log(
                "Memory usage high or run limit reached, reloading WASM module...",
            );
            showStatus("Refreshing memory...", "loading");

            try {
                await loadWasmModule();
                runCount = 0;
                showStatus("Memory refreshed successfully!", "success");
                setTimeout(() => hideStatus(), 2000);
            } catch (e) {
                console.error("Failed to reload WASM module:", e);
                showStatus("Memory refresh failed", "error");
            }
        }

        setTimeout(() => hideStatus(), 3000);
    } catch (error) {
        showStatus(`Error generating string art: ${error.message}`, "error");
        console.error("Generation error:", error);

        // Ensure cleanup even on error
        try {
            stringArtModule.ccall("cleanup", null, [], []);
        } catch (e) {
            console.warn("Emergency cleanup failed:", e);
        }
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
                const croppedImageData = getCroppedImageData();
                resolve(croppedImageData);
                return;
            } catch (error) {
                console.warn(
                    "Failed to get cropped image data, falling back to full image:",
                    error,
                );
            }
        }

        // Fallback to original behavior
        const img = new Image();
        img.onload = () => {
            try {
                const canvas = document.createElement("canvas");
                const ctx = canvas.getContext("2d");
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);

                const imageData = ctx.getImageData(0, 0, img.width, img.height);
                const data = imageData.data;

                // Check if WASM module is ready
                if (!stringArtModule || !stringArtModule._malloc) {
                    throw new Error("WASM module not ready");
                }

                // Allocate memory in WASM heap
                const dataPtr = stringArtModule._malloc(data.length);
                if (dataPtr === 0) {
                    throw new Error("Failed to allocate WASM memory");
                }

                // Copy data to WASM memory - handle if HEAPU8 is not available
                let wasmArray;
                if (stringArtModule.HEAPU8) {
                    wasmArray = new Uint8Array(
                        stringArtModule.HEAPU8.buffer,
                        dataPtr,
                        data.length,
                    );
                } else {
                    // Fallback: create a view directly from the exported memory
                    const memory = stringArtModule.wasmMemory || stringArtModule.memory;
                    if (memory?.buffer) {
                        wasmArray = new Uint8Array(memory.buffer, dataPtr, data.length);
                    } else {
                        throw new Error("Cannot access WASM memory");
                    }
                }
                wasmArray.set(data);

                resolve({
                    dataPtr: dataPtr,
                    width: img.width,
                    height: img.height,
                    channels: 4, // RGBA
                });
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

// Update SVG parameters efficiently without full re-render
function updateSVGParameters(outputSize) {
    const svg = document.getElementById("stringArtSVG");
    if (!svg || !currentLineSequence) return false;

    const currentSize = parseInt(svg.getAttribute("width"));
    const sizeChanged = currentSize !== outputSize;

    // Update SVG dimensions if size changed
    if (sizeChanged) {
        svg.setAttribute("width", outputSize);
        svg.setAttribute("height", outputSize);
        svg.setAttribute("viewBox", `0 0 ${outputSize} ${outputSize}`);

        // Recalculate pin coordinates for new size
        const pinCoords = calculatePinCoordinates(currentPinCount, outputSize);

        // Update pin positions
        const pins = svg.querySelectorAll("circle");
        pins.forEach((pin, index) => {
            if (index < pinCoords.length) {
                pin.setAttribute("cx", pinCoords[index].x);
                pin.setAttribute("cy", pinCoords[index].y);
            }
        });

        // Update line positions
        const lines = svg.querySelectorAll("line");
        lines.forEach((line, index) => {
            if (index < currentLineSequence.length - 1) {
                const fromPin = currentLineSequence[index];
                const toPin = currentLineSequence[index + 1];

                if (
                    fromPin >= 0 &&
                    fromPin < pinCoords.length &&
                    toPin >= 0 &&
                    toPin < pinCoords.length
                ) {
                    line.setAttribute("x1", pinCoords[fromPin].x);
                    line.setAttribute("y1", pinCoords[fromPin].y);
                    line.setAttribute("x2", pinCoords[toPin].x);
                    line.setAttribute("y2", pinCoords[toPin].y);
                }
            }
        });
    }

    return true; // Successfully updated in-place
}

// Update SVG output using current slider values
function updateSVGOutput() {
    if (!currentLineSequence) return;

    // Reset zoom and pan
    zoomLevel = 1;
    panX = 0;
    panY = 0;

    // Get current slider values
    const outputSize = parseInt(document.getElementById("outputSize").value);

    // Create SVG element
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", outputSize);
    svg.setAttribute("height", outputSize);
    svg.setAttribute("viewBox", `0 0 ${outputSize} ${outputSize}`);
    svg.style.background = "white";

    // Calculate pin coordinates for current output size
    const pinCoords = calculatePinCoordinates(currentPinCount, outputSize);

    // Render lines using SVG
    renderStringsToSVG(svg, currentLineSequence, pinCoords);

    // Update display in result tab
    const resultTab = document.getElementById("resultTab");
    resultTab.innerHTML = "";

    // Create zoomable container
    const zoomableContainer = document.createElement("div");
    zoomableContainer.className = "zoomable-container";
    zoomableContainer.style.position = "relative";
    zoomableContainer.style.width = "100%";
    zoomableContainer.style.height = "100%";

    const zoomableContent = document.createElement("div");
    zoomableContent.className = "zoomable-content";
    zoomableContent.style.display = "flex";
    zoomableContent.style.alignItems = "center";
    zoomableContent.style.justifyContent = "center";
    zoomableContent.style.height = "100%";

    zoomableContent.appendChild(svg);

    // Add zoom controls
    const zoomControls = document.createElement("div");
    zoomControls.className = "zoom-controls";
    zoomControls.innerHTML = `
        <button class="zoom-btn" onclick="adjustZoom(0.2)">+</button>
        <button class="zoom-btn" onclick="adjustZoom(-0.2)">−</button>
        <button class="zoom-btn" onclick="resetZoom()">Reset</button>
    `;

    // Add zoom info
    const zoomInfo = document.createElement("div");
    zoomInfo.className = "zoom-info";
    zoomInfo.id = "zoomInfo";
    zoomInfo.textContent = "Zoom: 100% | Cmd+Scroll to zoom, Scroll to pan";

    zoomableContainer.appendChild(zoomableContent);
    zoomableContainer.appendChild(zoomControls);
    zoomableContainer.appendChild(zoomInfo);

    // Add download section with line count and button
    const downloadSection = document.createElement("div");
    downloadSection.className = "download-section";
    downloadSection.style.position = "absolute";
    downloadSection.style.bottom = "20px";
    downloadSection.style.right = "20px";
    downloadSection.style.zIndex = "100";
    downloadSection.style.textAlign = "center";

    // Add line count message
    const lineCountMsg = document.createElement("div");
    lineCountMsg.style.color = "#666";
    lineCountMsg.style.fontSize = "14px";
    lineCountMsg.style.marginBottom = "8px";
    lineCountMsg.textContent = `${currentLineCount - 1} lines`;
    downloadSection.appendChild(lineCountMsg);

    const downloadBtn = document.createElement("button");
    downloadBtn.className = "download-btn";
    downloadBtn.textContent = "Download as SVG";
    downloadBtn.onclick = () => {
        downloadSVGWithStyles(svg);
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

    // Add event listeners for zoom and pan
    setupZoomAndPan(zoomableContainer, zoomableContent);

    // Switch to result tab when SVG is updated
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
}

// Setup zoom and pan functionality
function setupZoomAndPan(container, content) {
    // Wheel event for zoom (Ctrl+scroll) and pan (normal scroll)
    container.addEventListener("wheel", (e) => {
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
    });

    // Mouse drag for panning
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let startPanX = 0;
    let startPanY = 0;

    container.addEventListener("mousedown", (e) => {
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        startPanX = panX;
        startPanY = panY;
        container.style.cursor = "grabbing";
    });

    document.addEventListener("mousemove", (e) => {
        if (!isDragging) return;

        const deltaX = e.clientX - startX;
        const deltaY = e.clientY - startY;

        panX = startPanX + deltaX;
        panY = startPanY + deltaY;
        updateTransform(content);
    });

    document.addEventListener("mouseup", () => {
        if (isDragging) {
            isDragging = false;
            container.style.cursor = "grab";
        }
    });
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
    container.addEventListener("wheel", (e) => {
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
    });

    // Mouse drag for panning
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let startPanX = 0;
    let startPanY = 0;

    container.addEventListener("mousedown", (e) => {
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        startPanX = imgPanX;
        startPanY = imgPanY;
        container.style.cursor = "grabbing";
    });

    document.addEventListener("mousemove", (e) => {
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
    });

    document.addEventListener("mouseup", () => {
        if (isDragging) {
            isDragging = false;
            container.style.cursor = "grab";
        }
    });
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

// Get cropped image data based on current pan/zoom state
function getCroppedImageData() {
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

    // Get the image data from the canvas
    const imageData = tempCtx.getImageData(0, 0, targetSize, targetSize);
    const data = imageData.data;

    // Check if WASM module is ready
    if (!stringArtModule || !stringArtModule._malloc) {
        throw new Error("WASM module not ready");
    }

    // Allocate memory in WASM heap
    const dataPtr = stringArtModule._malloc(data.length);
    if (dataPtr === 0) {
        throw new Error("Failed to allocate WASM memory");
    }

    // Copy data to WASM memory
    let wasmArray;
    if (stringArtModule.HEAPU8) {
        wasmArray = new Uint8Array(
            stringArtModule.HEAPU8.buffer,
            dataPtr,
            data.length,
        );
    } else {
        // Fallback: create a view directly from the exported memory
        const memory = stringArtModule.wasmMemory || stringArtModule.memory;
        if (memory?.buffer) {
            wasmArray = new Uint8Array(memory.buffer, dataPtr, data.length);
        } else {
            throw new Error("Cannot access WASM memory");
        }
    }
    wasmArray.set(data);

    return {
        dataPtr: dataPtr,
        width: targetSize,
        height: targetSize,
        channels: 4, // RGBA
    };
}

// Update CSS transform
function updateTransform(content) {
    if (content) {
        content.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomLevel})`;

        // Update zoom info
        const zoomInfo = document.getElementById("zoomInfo");
        if (zoomInfo) {
            zoomInfo.textContent = `Zoom: ${Math.round(zoomLevel * 100)}% | Ctrl+Scroll to zoom, Scroll to pan`;
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
    if (!fileInput.files || fileInput.files.length === 0) {
        alert("No input file selected");
        return;
    }

    const parts = fileInput.files[0].name.split(".");
    const baseName = parts.slice(0, -1).join(".");
    const outname = `${baseName}-pins.txt`;

    // Create text content with one pin number per line
    const pinListContent = currentLineSequence.join("\n");

    // Create and download the text file
    const blob = new Blob([pinListContent], { type: "text/plain" });
    const link = document.createElement("a");
    link.download = outname;
    link.href = URL.createObjectURL(blob);
    link.click();
}

// Download SVG with embedded styles for standalone viewing
function downloadSVGWithStyles(originalSvg) {
    // Clone the SVG to avoid modifying the original
    const svgClone = originalSvg.cloneNode(true);

    // Create a style element with the necessary CSS
    const styleElement = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "style",
    );
    styleElement.textContent = `
        line {
            stroke: black;
            stroke-width: 1;
        }
        circle {
            fill: black;
            r: 2;
        }
        svg {
            width: 100%;
            height: auto;
            max-height: 100vh;
        }
    `;

    // Insert style element as the first child of the SVG
    svgClone.insertBefore(styleElement, svgClone.firstChild);

    // generate the output filename
    const fileInput = document.getElementById("imageInput");
    const parts = fileInput.files[0].name.split(".");
    const baseName = parts.slice(0, -1).join(".");
    const outname = `${baseName}-strings.svg`;

    // Serialize and download
    const svgData = new XMLSerializer().serializeToString(svgClone);
    const blob = new Blob([svgData], { type: "image/svg+xml" });
    const link = document.createElement("a");
    link.download = outname;
    link.href = URL.createObjectURL(blob);
    link.click();
}

// Render string art lines to SVG
function renderStringsToSVG(svg, lineSequence, pinCoords) {
    // Add SVG ID for later reference
    svg.setAttribute("id", "stringArtSVG");

    // Add pins as small circles with minimal attributes (styling via CSS)
    for (let i = 0; i < pinCoords.length; i++) {
        const circle = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "circle",
        );
        circle.setAttribute("cx", pinCoords[i].x);
        circle.setAttribute("cy", pinCoords[i].y);
        svg.appendChild(circle);
    }

    // Add lines between pins with minimal attributes (styling via CSS)
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

        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", pinCoords[fromPin].x);
        line.setAttribute("y1", pinCoords[fromPin].y);
        line.setAttribute("x2", pinCoords[toPin].x);
        line.setAttribute("y2", pinCoords[toPin].y);
        svg.appendChild(line);
    }
}
