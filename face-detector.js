// face-detector.js
// Lazy-loaded MediaPipe Face Landmarker for face detection, alignment, and cropping.

const MEDIAPIPE_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

let landmarker = null;
let initPromise = null;

// Smoothed transform state (EMA)
// smoothingFactor 0.3 means ~75% of target reached after 4 frames: 1 - (1-0.3)^4 ≈ 0.76
const SMOOTHING = 0.3;
let smooth = null; // { centerX, centerY, roll, size }

// User transform adjustments (pan/zoom overlay, controlled from result pane UI)
let userPanX = 0;
let userPanY = 0;
let userZoom = 1.0;

/**
 * Initialize the face landmarker. Called automatically on first use.
 * Loads ~10MB of WASM + model from CDN.
 */
export async function init() {
    if (landmarker) return;
    if (initPromise) return initPromise;

    initPromise = (async () => {
        const { FaceLandmarker, FilesetResolver } = await import(
            `${MEDIAPIPE_CDN}/vision_bundle.mjs`
        );

        const vision = await FilesetResolver.forVisionTasks(
            `${MEDIAPIPE_CDN}/wasm`
        );

        landmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: MODEL_URL,
                delegate: "GPU",
            },
            runningMode: "IMAGE",
            numFaces: 1,
        });
    })();

    return initPromise;
}

/**
 * Reset smoothing and user transform state (e.g. when switching from webcam to static image).
 */
export function resetSmoothing() {
    smooth = null;
    userPanX = 0;
    userPanY = 0;
    userZoom = 1.0;
}

export function adjustUserPan(dx, dy) { userPanX += dx; userPanY += dy; }
export function adjustUserZoom(delta) { userZoom = Math.max(0.2, Math.min(5, userZoom + delta)); }
export function resetUserTransform() { userPanX = 0; userPanY = 0; userZoom = 1.0; }
export function getUserZoom() { return userZoom; }

/**
 * Detect the largest face, align it vertically (undo roll), and crop to a square.
 *
 * @param {HTMLImageElement|HTMLCanvasElement|HTMLVideoElement} source
 * @param {number} outputSize - Square output dimension in pixels
 * @param {object} [options]
 * @param {number} [options.padding=1.5] - Multiplier around the face bounding box
 * @param {boolean} [options.smoothed=false] - Apply temporal smoothing (for video/webcam)
 * @returns {Promise<ImageData|null>} Aligned, cropped face or null if no face found
 */
export async function detectAndCrop(source, outputSize, options = {}) {
    const { padding = 1.5, smoothed = false } = options;

    await init();

    const result = landmarker.detect(source);

    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
        return null;
    }

    const landmarks = result.faceLandmarks[0];

    // Source dimensions
    const width = source.naturalWidth || source.width || source.videoWidth;
    const height = source.naturalHeight || source.height || source.videoHeight;

    // Compute roll angle from eye positions
    // Landmark 33: left eye inner corner
    // Landmark 263: right eye inner corner
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];

    const rawRoll = Math.atan2(
        (rightEye.y - leftEye.y) * height,
        (rightEye.x - leftEye.x) * width,
    );

    // Compute face bounding box from all landmarks
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const lm of landmarks) {
        const px = lm.x * width;
        const py = lm.y * height;
        if (px < minX) minX = px;
        if (py < minY) minY = py;
        if (px > maxX) maxX = px;
        if (py > maxY) maxY = py;
    }

    const rawCenterX = (minX + maxX) / 2;
    const faceHeight = maxY - minY;
    // Shift crop upward so chin sits near the bottom and forehead/skull fills the top.
    // Landmarks don't include the top of the skull, so we bias upward.
    const rawCenterY = (minY + maxY) / 2 - faceHeight * 0.15;
    const rawSize = Math.max(maxX - minX, faceHeight) * padding;

    // Apply temporal smoothing (EMA) for video/webcam
    let centerX, centerY, roll, faceSize;
    if (smoothed) {
        if (!smooth) {
            // First frame: snap to detected values
            smooth = { centerX: rawCenterX, centerY: rawCenterY, roll: rawRoll, size: rawSize };
        } else {
            smooth.centerX += SMOOTHING * (rawCenterX - smooth.centerX);
            smooth.centerY += SMOOTHING * (rawCenterY - smooth.centerY);
            smooth.roll += SMOOTHING * (rawRoll - smooth.roll);
            smooth.size += SMOOTHING * (rawSize - smooth.size);
        }
        centerX = smooth.centerX;
        centerY = smooth.centerY;
        roll = smooth.roll;
        faceSize = smooth.size;
    } else {
        centerX = rawCenterX;
        centerY = rawCenterY;
        roll = rawRoll;
        faceSize = rawSize;
    }

    // Draw rotated + cropped face onto output canvas
    const canvas = document.createElement("canvas");
    canvas.width = outputSize;
    canvas.height = outputSize;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, outputSize, outputSize);

    // Transform: center output on face, rotate to align, scale to fit
    // User pan/zoom is applied on top (in output pixel space)
    const scale = (outputSize / faceSize) * userZoom;
    ctx.translate(outputSize / 2 + userPanX, outputSize / 2 + userPanY);
    ctx.rotate(-roll);
    ctx.scale(scale, scale);
    ctx.translate(-centerX, -centerY);
    ctx.drawImage(source, 0, 0);

    return ctx.getImageData(0, 0, outputSize, outputSize);
}
