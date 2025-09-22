var StringArtModule = null;
let stringArtModule = null;
let isProcessing = false;
let runCount = 0;

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
                const originalTab = document.getElementById("originalTab");
                originalTab.innerHTML = "";
                originalTab.appendChild(img);
                hasInitialImage = true;
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

// Schedule auto-regeneration with debouncing
function scheduleAutoRegeneration() {
    const autoRegenerate = document.getElementById("autoRegenerate");
    if (!autoRegenerate.checked || !hasInitialImage || isProcessing) {
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
        const outputSize = parseInt(document.getElementById("outputSize").value);
        const lineWeight = parseInt(document.getElementById("lineWeight").value);
        const minDistance = parseInt(document.getElementById("minDistance").value);
        console.log("Parameters retrieved:", {
            pins,
            maxLines,
            targetSize,
            outputSize,
            lineWeight,
            minDistance,
        });

        // Validate parameters
        console.log("Using parameters:", {
            pins,
            maxLines,
            targetSize,
            outputSize,
            lineWeight,
            minDistance,
        });

        if (pins < 50 || pins > 1000) throw new Error("Invalid pins count");
        if (maxLines < 10 || maxLines > 50000) throw new Error("Invalid max lines");
        if (targetSize < 100 || targetSize > 2000)
            throw new Error("Invalid target size");

        // Initialize the generator
        console.log("Initializing string art generator...");
        const initResult = stringArtModule.ccall(
            "initStringArt",
            "number",
            ["number", "number", "number", "number", "number", "number", "number"],
            [
                pins,
                maxLines,
                targetSize,
                outputSize,
                lineWeight,
                lineWeight,
                minDistance,
            ],
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
